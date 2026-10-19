// Command buffers, recording and submission.
//
// This is where ADR-1 is paid for.  A Vulkan command buffer is an object:
// the application may record it once and submit it many times, record it on one
// thread and submit from another, and reset and re-record it in between.  An
// OpenGL command stream has none of those properties -- commands are issued
// immediately and belong to whichever context is current on the calling thread.
//
// So Kniz records into the small IR in kniz_objects.h and replays it inside
// vkQueueSubmit, on the submitting thread, with the context made current.  The
// consequence worth stating plainly: nothing executes until submission, and a
// submission blocks until the GPU is done, because vkQueueSubmit has to return
// with the fence signalled for the ordering guarantees to hold.  That is
// correct but it is not fast, and it is why batching at submission matters more
// here than it would in a driver that could queue work.
//
// The alternative designs and why they were rejected are recorded in
// docs/DESIGN.md (ADR-1).

#include "kniz_common.h"
#include "kniz_image.h"
#include "kniz_memory.h"
#include "kniz_objects.h"

#include <algorithm>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace kniz {
namespace {

// vkCmdUpdateBuffer and vkCmdFillBuffer both cap what they may touch, and both
// are honoured by writing from the host.  The chunk keeps a fill of a large
// buffer from needing a matching host allocation.
constexpr uint64_t kFillChunkBytes = 1u << 20;
constexpr uint64_t kUpdateBufferMax = 65536;

bool recording(CommandBuffer* cmd, const char* what) {
  if (cmd == nullptr) {
    KNIZ_ERR("%s was called with something that is not a Kniz command buffer",
             what);
    return false;
  }
  if (!cmd->recording) {
    KNIZ_ERR("%s was called outside vkBeginCommandBuffer/vkEndCommandBuffer",
             what);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

void replay_dispatch(const RecordedCommand& command) {
  Pipeline* pipeline = command.pipeline;
  gl::f.UseProgram(pipeline->program);

  gl::report_errors("after glUseProgram");

  // The pipeline layout decided, at creation, which GL binding point each
  // (set, binding) got; the shader was generated from the same table.  So this
  // loop only has to read the table and bind what the sets hold.
  for (const GlBinding& mapping : pipeline->bindings) {
    if (mapping.set >= command.sets.size()) {
      KNIZ_ERR("dispatch needs set %u, which the pipeline layout declares, but "
               "the command buffer bound fewer sets",
               mapping.set);
      continue;
    }

    DescriptorSet* set = command.sets[mapping.set];
    if (set == nullptr) {
      KNIZ_ERR("dispatch reads set %u binding %u, but no descriptor set is "
               "bound there",
               mapping.set, mapping.binding);
      continue;
    }

    if (mapping.binding >= set->bindings.size()) {
      KNIZ_ERR("the bound set has no binding %u", mapping.binding);
      continue;
    }

    const DescriptorBinding& binding = set->bindings[mapping.binding];
    if (!binding.bound || binding.buffer == nullptr) {
      KNIZ_ERR("set %u binding %u is never written; the dispatch would read "
               "undefined data",
               mapping.set, mapping.binding);
      continue;
    }

    Buffer* buffer = binding.buffer;
    uint64_t range = binding.range;
    if (range == VK_WHOLE_SIZE || range == 0 || range > buffer->size - binding.offset)
      range = buffer->size - binding.offset;

    // GL requires indexed storage buffer ranges to start at a multiple of
    // GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT.  Vulkan caps the equivalent
    // limit at 256, so on a driver that wants more this is the point where the
    // two cannot be reconciled.
    const int alignment = static_cast<int>(buffer->device->instance->gl.ssbo_offset_alignment);
    if (alignment > 0 && binding.offset % static_cast<uint64_t>(alignment) != 0) {
      KNIZ_ERR("buffer offset %llu is not a multiple of this driver's shader "
               "storage buffer alignment (%d); the descriptor cannot be bound",
               static_cast<unsigned long long>(binding.offset), alignment);
      continue;
    }

    KNIZ_TRACE("dispatch: set %u binding %u -> GL_SHADER_STORAGE_BUFFER[%u] = "
               "buffer %u, offset %llu, range %llu",
               mapping.set, mapping.binding, mapping.gl_index, buffer->gl_name,
               static_cast<unsigned long long>(binding.offset),
               static_cast<unsigned long long>(range));

    gl::f.BindBufferRange(gl::kShaderStorageBuffer, mapping.gl_index,
                          buffer->gl_name,
                          static_cast<gl::IntPtr>(binding.offset),
                          gl::as_size_ptr(range));
  }

  gl::report_errors("after binding descriptor sets");

  gl::f.DispatchCompute(command.group_count[0], command.group_count[1],
                        command.group_count[2]);

  gl::report_errors("after glDispatchCompute");

  // Make the writes visible to everything that follows: a later fill, a copy,
  // the read back to the host, or the next dispatch.
  gl::f.MemoryBarrier(gl::kAllBarrierBits);
}

void replay_fill(const RecordedCommand& command) {
  std::vector<uint32_t> scratch(static_cast<size_t>(kFillChunkBytes / 4),
                                command.data);
  gl::f.BindBuffer(gl::kArrayBuffer, command.dst->gl_name);

  uint64_t written = 0;
  while (written < command.size) {
    const uint64_t chunk = std::min(command.size - written, kFillChunkBytes);
    gl::f.BufferSubData(gl::kArrayBuffer,
                        static_cast<gl::IntPtr>(command.offset + written),
                        gl::as_size_ptr(chunk), scratch.data());
    written += chunk;
  }

  gl::f.BindBuffer(gl::kArrayBuffer, 0);
}

void replay_update(const RecordedCommand& command) {
  gl::f.BindBuffer(gl::kArrayBuffer, command.dst->gl_name);
  gl::f.BufferSubData(gl::kArrayBuffer, static_cast<gl::IntPtr>(command.offset),
                      gl::as_size_ptr(command.size), command.bytes.data());
  gl::f.BindBuffer(gl::kArrayBuffer, 0);
}

void replay_copy(const RecordedCommand& command) {
  if (command.src == command.dst) {
    // GL forbids the same buffer object being bound to both the read and write
    // side of a copy, so an in-buffer move has no direct GL equivalent.
    KNIZ_ERR("copying within a single buffer is not implemented (source and "
             "destination are the same buffer)");
    return;
  }

  gl::f.BindBuffer(gl::kCopyReadBuffer, command.src->gl_name);
  gl::f.BindBuffer(gl::kCopyWriteBuffer, command.dst->gl_name);
  gl::f.CopyBufferSubData(
      gl::kCopyReadBuffer, gl::kCopyWriteBuffer,
      static_cast<gl::IntPtr>(command.src_offset),
      static_cast<gl::IntPtr>(command.offset), gl::as_size_ptr(command.size));
  gl::f.BindBuffer(gl::kCopyReadBuffer, 0);
  gl::f.BindBuffer(gl::kCopyWriteBuffer, 0);

  gl::f.MemoryBarrier(gl::kAllBarrierBits);
}

// The image commands.  Each reports its own failures internally; a copy that
// fails partway stops, because writing half an image from data that is not
// trustworthy is worse than not writing it.

void replay_clear_image(const RecordedCommand& command) {
  if (command.image == nullptr)
    return;
  image_clear(command.image, command.clear_range, command.clear_value);
}

void replay_copy_buffer_to_image(const RecordedCommand& command) {
  for (const VkBufferImageCopy& region : command.regions) {
    if (!image_write(command.src, command.image, region))
      return;
  }
}

void replay_copy_image_to_buffer(const RecordedCommand& command) {
  for (const VkBufferImageCopy& region : command.regions) {
    if (!image_read(command.dst, command.image, region))
      return;
  }
}

void replay(const CommandBuffer& cmd) {
  for (const RecordedCommand& command : cmd.commands) {
    switch (command.kind) {
      case RecordedCommand::Kind::Barrier:
        // A Vulkan barrier orders memory access between stages.  GL reaches the
        // same guarantee by keeping commands in order and inserting an explicit
        // barrier; all-bits is over-broad, and being over-broad is safe.
        gl::f.MemoryBarrier(gl::kAllBarrierBits);
        break;
      case RecordedCommand::Kind::Dispatch:
        replay_dispatch(command);
        break;
      case RecordedCommand::Kind::FillBuffer:
        replay_fill(command);
        break;
      case RecordedCommand::Kind::UpdateBuffer:
        replay_update(command);
        break;
      case RecordedCommand::Kind::CopyBuffer:
        replay_copy(command);
        break;
      case RecordedCommand::Kind::ClearColorImage:
        replay_clear_image(command);
        break;
      case RecordedCommand::Kind::CopyBufferToImage:
        replay_copy_buffer_to_image(command);
        break;
      case RecordedCommand::Kind::CopyImageToBuffer:
        replay_copy_image_to_buffer(command);
        break;
    }
  }
}

void clear_recording_state(CommandBuffer* cmd) {
  cmd->commands.clear();
  cmd->bound_pipeline = nullptr;
  cmd->bound_layout = nullptr;
  cmd->bound_sets.clear();
}

}  // namespace

// ---------------------------------------------------------------------------
// Command pools
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateCommandPool(
    VkDevice deviceHandle, const VkCommandPoolCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkCommandPool* pCommandPool) {
  (void)pAllocator;
  if (pCommandPool == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pCommandPool = VK_NULL_HANDLE;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr || pCreateInfo == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  if (pCreateInfo->queueFamilyIndex != 0) {
    KNIZ_ERR("this driver has one queue family (index 0), but a command pool "
             "was requested for family %u",
             pCreateInfo->queueFamilyIndex);
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  CommandPool* pool = new (std::nothrow) CommandPool();
  if (pool == nullptr)
    return VK_ERROR_OUT_OF_HOST_MEMORY;

  pool->device = device;
  pool->family_index = pCreateInfo->queueFamilyIndex;
  pool->resettable =
      (pCreateInfo->flags & VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT) != 0;

  *pCommandPool = to_handle(pool);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyCommandPool(
    VkDevice deviceHandle, VkCommandPool commandPoolHandle,
    const VkAllocationCallbacks* pAllocator) {
  (void)pAllocator;
  if (from_handle(deviceHandle) == nullptr)
    return;
  CommandPool* pool = from_handle(commandPoolHandle);
  if (pool == nullptr)
    return;

  for (CommandBuffer* cmd : pool->buffers)
    delete cmd;
  delete pool;
}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandPool(
    VkDevice deviceHandle, VkCommandPool commandPoolHandle,
    VkCommandPoolResetFlags flags) {
  (void)flags;
  if (from_handle(deviceHandle) == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  CommandPool* pool = from_handle(commandPoolHandle);
  if (pool == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  for (CommandBuffer* cmd : pool->buffers) {
    clear_recording_state(cmd);
    cmd->recording = false;
    cmd->executable = false;
  }
  return VK_SUCCESS;
}

// ---------------------------------------------------------------------------
// Command buffers
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(
    VkDevice deviceHandle, const VkCommandBufferAllocateInfo* pAllocateInfo,
    VkCommandBuffer* pCommandBuffers) {
  if (pCommandBuffers == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr || pAllocateInfo == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  CommandPool* pool = from_handle(pAllocateInfo->commandPool);
  if (pool == nullptr) {
    KNIZ_ERR("the command pool is not a Kniz command pool");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  if (pAllocateInfo->level != VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
    // Secondary command buffers exist to be recorded in parallel and executed
    // from a primary one.  Kniz executes at submission, so there is nothing for
    // them to buy and a great deal of state to inherit.
    KNIZ_ERR("secondary command buffers are not implemented; allocate primary "
             "buffers");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  for (uint32_t i = 0; i < pAllocateInfo->commandBufferCount; ++i) {
    CommandBuffer* cmd = new (std::nothrow) CommandBuffer();
    if (cmd == nullptr)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

    // The loader writes its own dispatch table into the first word of every
    // dispatchable handle it hands out, and vkAllocateCommandBuffers is one of
    // the places it does so.  Marking the reservation is what makes that legal.
    set_loader_magic_value(cmd);

    cmd->device = device;
    cmd->pool = pool;
    pool->buffers.push_back(cmd);
    pCommandBuffers[i] = to_handle(cmd);
  }

  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FreeCommandBuffers(
    VkDevice deviceHandle, VkCommandPool commandPoolHandle,
    uint32_t commandBufferCount, const VkCommandBuffer* pCommandBuffers) {
  Device* device = from_handle(deviceHandle);
  CommandPool* pool = from_handle(commandPoolHandle);
  if (device == nullptr || pool == nullptr)
    return;

  for (uint32_t i = 0; i < commandBufferCount; ++i) {
    CommandBuffer* cmd = from_handle(pCommandBuffers[i]);
    if (cmd == nullptr)
      continue;
    for (auto it = pool->buffers.begin(); it != pool->buffers.end(); ++it) {
      if (*it == cmd) {
        pool->buffers.erase(it);
        break;
      }
    }
    delete cmd;
  }
}

VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(
    VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo* pBeginInfo) {
  (void)pBeginInfo;
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (cmd == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  if (cmd->recording) {
    KNIZ_ERR("vkBeginCommandBuffer on a command buffer that is already "
             "recording");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  clear_recording_state(cmd);
  cmd->recording = true;
  cmd->executable = false;
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL EndCommandBuffer(VkCommandBuffer commandBuffer) {
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (cmd == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (!cmd->recording) {
    KNIZ_ERR("vkEndCommandBuffer without a matching vkBeginCommandBuffer");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  cmd->recording = false;
  cmd->executable = true;
  KNIZ_TRACE("command buffer %p recorded %zu command(s)",
             static_cast<void*>(commandBuffer), cmd->commands.size());
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandBuffer(
    VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags) {
  (void)flags;
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (cmd == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  // vkResetCommandBuffer itself is the only way to re-record a buffer whose
  // pool was created without VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
  // so it has to obey that flag even though nothing here is pooled.
  if (cmd->pool != nullptr && !cmd->pool->resettable) {
    KNIZ_ERR("this command pool was created without "
             "VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, so its buffers "
             "cannot be reset individually");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  clear_recording_state(cmd);
  cmd->recording = false;
  cmd->executable = false;
  return VK_SUCCESS;
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

VKAPI_ATTR void VKAPI_CALL CmdBindPipeline(VkCommandBuffer commandBuffer,
                                           VkPipelineBindPoint pipelineBindPoint,
                                           VkPipeline pipelineHandle) {
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdBindPipeline"))
    return;

  if (pipelineBindPoint != VK_PIPELINE_BIND_POINT_COMPUTE) {
    KNIZ_ERR("vkCmdBindPipeline only supports compute bind points until M2");
    return;
  }

  Pipeline* pipeline = from_handle(pipelineHandle);
  if (pipeline == nullptr) {
    KNIZ_ERR("the pipeline is not a Kniz pipeline");
    return;
  }

  cmd->bound_pipeline = pipeline;
  cmd->bound_layout = pipeline->layout;
}

VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSets(
    VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
    VkPipelineLayout layoutHandle, uint32_t firstSet, uint32_t descriptorSetCount,
    const VkDescriptorSet* pDescriptorSets, uint32_t dynamicOffsetCount,
    const uint32_t* pDynamicOffsets) {
  (void)pDynamicOffsets;
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdBindDescriptorSets"))
    return;

  if (pipelineBindPoint != VK_PIPELINE_BIND_POINT_COMPUTE) {
    KNIZ_ERR("vkCmdBindDescriptorSets only supports compute bind points until M2");
    return;
  }

  if (dynamicOffsetCount > 0) {
    KNIZ_ERR("dynamic descriptor offsets are not implemented");
    return;
  }

  PipelineLayout* layout = from_handle(layoutHandle);
  if (layout == nullptr) {
    KNIZ_ERR("the pipeline layout is not a Kniz pipeline layout");
    return;
  }
  if (cmd->bound_layout != nullptr && cmd->bound_layout != layout) {
    KNIZ_ERR("vkCmdBindDescriptorSets was given a layout that differs from the "
             "bound pipeline's; the sets would not match the shader");
    return;
  }

  if (firstSet + descriptorSetCount > cmd->bound_sets.size())
    cmd->bound_sets.resize(firstSet + descriptorSetCount, nullptr);

  for (uint32_t i = 0; i < descriptorSetCount; ++i) {
    DescriptorSet* set = from_handle(pDescriptorSets[i]);
    if (set == nullptr) {
      KNIZ_ERR("descriptor set %u is not a Kniz descriptor set", firstSet + i);
      continue;
    }
    cmd->bound_sets[firstSet + i] = set;
  }
}

VKAPI_ATTR void VKAPI_CALL CmdPushConstants(
    VkCommandBuffer commandBuffer, VkPipelineLayout layout, VkShaderStageFlags stageFlags,
    uint32_t offset, uint32_t size, const void* pValues) {
  (void)layout;
  (void)stageFlags;
  (void)offset;
  (void)size;
  (void)pValues;
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdPushConstants"))
    return;
  // GL has no push constants.  A pipeline whose shader actually reads them is
  // refused at creation, so if recording reaches here the ranges are declared
  // but unused and dropping the values changes nothing.
  KNIZ_DBG("vkCmdPushConstants ignored: OpenGL has no push constant equivalent");
}

VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier(
    VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask,
    VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags,
    uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers,
    uint32_t bufferMemoryBarrierCount,
    const VkBufferMemoryBarrier* pBufferMemoryBarriers,
    uint32_t imageMemoryBarrierCount,
    const VkImageMemoryBarrier* pImageMemoryBarriers) {
  (void)srcStageMask;
  (void)dstStageMask;
  (void)dependencyFlags;
  (void)memoryBarrierCount;
  (void)pMemoryBarriers;
  (void)bufferMemoryBarrierCount;
  (void)pBufferMemoryBarriers;

  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdPipelineBarrier"))
    return;

  // The layout transitions are recorded on the images for diagnostics and
  // otherwise dropped: OpenGL has no image layouts, and the ordering a barrier
  // provides is what the recorded memory barrier below is for.
  for (uint32_t i = 0; i < imageMemoryBarrierCount; ++i) {
    Image* image = from_handle(pImageMemoryBarriers[i].image);
    if (image != nullptr && image->device == cmd->device)
      image->layout = pImageMemoryBarriers[i].newLayout;
  }

  RecordedCommand command;
  command.kind = RecordedCommand::Kind::Barrier;
  cmd->commands.push_back(std::move(command));
}

VKAPI_ATTR void VKAPI_CALL CmdDispatch(VkCommandBuffer commandBuffer,
                                       uint32_t groupCountX, uint32_t groupCountY,
                                       uint32_t groupCountZ) {
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdDispatch"))
    return;

  if (cmd->bound_pipeline == nullptr) {
    KNIZ_ERR("vkCmdDispatch with no compute pipeline bound");
    return;
  }

  RecordedCommand command;
  command.kind = RecordedCommand::Kind::Dispatch;
  command.pipeline = cmd->bound_pipeline;
  command.group_count[0] = groupCountX;
  command.group_count[1] = groupCountY;
  command.group_count[2] = groupCountZ;

  // Resolve the sets now rather than at replay: Vulkan says the descriptors a
  // dispatch uses are the ones bound when it is recorded, and capturing them
  // here makes the recorded command independent of what happens later.
  PipelineLayout* layout = cmd->bound_pipeline->layout;
  command.sets.assign(layout->sets.size(), nullptr);
  for (size_t set = 0; set < command.sets.size() && set < cmd->bound_sets.size();
       ++set) {
    DescriptorSet* bound = cmd->bound_sets[set];
    if (bound != nullptr && bound->layout != layout->sets[set]) {
      KNIZ_ERR("descriptor set %zu was created from a different layout than the "
               "pipeline layout declares for that set; refusing to record the "
               "dispatch",
               set);
      return;
    }
    command.sets[set] = bound;
  }

  cmd->commands.push_back(std::move(command));
}

VKAPI_ATTR void VKAPI_CALL CmdFillBuffer(VkCommandBuffer commandBuffer,
                                         VkBuffer dstBuffer,
                                         VkDeviceSize dstOffset,
                                         VkDeviceSize size, uint32_t data) {
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdFillBuffer"))
    return;

  Buffer* buffer = from_handle(dstBuffer);
  if (buffer == nullptr) {
    KNIZ_ERR("vkCmdFillBuffer was given something that is not a Kniz buffer");
    return;
  }
  if ((buffer->usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) == 0) {
    KNIZ_ERR("vkCmdFillBuffer needs VK_BUFFER_USAGE_TRANSFER_DST_BIT");
    return;
  }

  if (size == VK_WHOLE_SIZE)
    size = buffer->size - dstOffset;

  if (dstOffset % 4 != 0 || size % 4 != 0) {
    KNIZ_ERR("vkCmdFillBuffer needs an offset and size that are multiples of 4 "
             "(got %llu and %llu)",
             static_cast<unsigned long long>(dstOffset),
             static_cast<unsigned long long>(size));
    return;
  }
  if (dstOffset + size > buffer->size) {
    KNIZ_ERR("vkCmdFillBuffer range [%llu, %llu) does not fit in a %llu byte "
             "buffer",
             static_cast<unsigned long long>(dstOffset),
             static_cast<unsigned long long>(dstOffset + size),
             static_cast<unsigned long long>(buffer->size));
    return;
  }

  RecordedCommand command;
  command.kind = RecordedCommand::Kind::FillBuffer;
  command.dst = buffer;
  command.offset = dstOffset;
  command.size = size;
  command.data = data;
  cmd->commands.push_back(std::move(command));
}

VKAPI_ATTR void VKAPI_CALL CmdUpdateBuffer(VkCommandBuffer commandBuffer,
                                           VkBuffer dstBuffer,
                                           VkDeviceSize dstOffset,
                                           VkDeviceSize dataSize,
                                           const void* pData) {
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdUpdateBuffer"))
    return;

  Buffer* buffer = from_handle(dstBuffer);
  if (buffer == nullptr) {
    KNIZ_ERR("vkCmdUpdateBuffer was given something that is not a Kniz buffer");
    return;
  }
  if ((buffer->usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) == 0) {
    KNIZ_ERR("vkCmdUpdateBuffer needs VK_BUFFER_USAGE_TRANSFER_DST_BIT");
    return;
  }
  if (dataSize == 0 || dataSize % 4 != 0 || dataSize > kUpdateBufferMax) {
    KNIZ_ERR("vkCmdUpdateBuffer takes between 4 and %llu bytes, in multiples of "
             "4 (got %llu)",
             static_cast<unsigned long long>(kUpdateBufferMax),
             static_cast<unsigned long long>(dataSize));
    return;
  }
  if (pData == nullptr || dstOffset + dataSize > buffer->size) {
    KNIZ_ERR("vkCmdUpdateBuffer range [%llu, %llu) does not fit in a %llu byte "
             "buffer",
             static_cast<unsigned long long>(dstOffset),
             static_cast<unsigned long long>(dstOffset + dataSize),
             static_cast<unsigned long long>(buffer->size));
    return;
  }

  RecordedCommand command;
  command.kind = RecordedCommand::Kind::UpdateBuffer;
  command.dst = buffer;
  command.offset = dstOffset;
  command.size = dataSize;
  // Copied, not referenced: the application is free to reuse pData as soon as
  // this call returns, and there is no guarantee the memory still exists at
  // submission.
  const auto* bytes = static_cast<const uint8_t*>(pData);
  command.bytes.assign(bytes, bytes + dataSize);
  cmd->commands.push_back(std::move(command));
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBuffer(VkCommandBuffer commandBuffer,
                                         VkBuffer srcBuffer, VkBuffer dstBuffer,
                                         uint32_t regionCount,
                                         const VkBufferCopy* pRegions) {
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdCopyBuffer"))
    return;

  Buffer* src = from_handle(srcBuffer);
  Buffer* dst = from_handle(dstBuffer);
  if (src == nullptr || dst == nullptr) {
    KNIZ_ERR("vkCmdCopyBuffer was given something that is not a Kniz buffer");
    return;
  }

  for (uint32_t i = 0; i < regionCount; ++i) {
    const VkBufferCopy& region = pRegions[i];
    if (region.size == 0)
      continue;
    if (region.srcOffset + region.size > src->size ||
        region.dstOffset + region.size > dst->size) {
      KNIZ_ERR("copy region %u does not fit in the buffers it names", i);
      continue;
    }

    RecordedCommand command;
    command.kind = RecordedCommand::Kind::CopyBuffer;
    command.src = src;
    command.dst = dst;
    command.src_offset = region.srcOffset;
    command.offset = region.dstOffset;
    command.size = region.size;
    cmd->commands.push_back(std::move(command));
  }
}

// ---------------------------------------------------------------------------
// Image commands
//
// Each of these records what it needs and defers the GL work to replay, matching
// the buffer commands.  The one interesting difference is that they capture data
// rather than just references: the clear value is copied out of the caller's
// union, and a copy captures its whole region list, because Vulkan lets the
// caller free both as soon as the recording call returns.
// ---------------------------------------------------------------------------

VKAPI_ATTR void VKAPI_CALL CmdClearColorImage(
    VkCommandBuffer commandBuffer, VkImage imageHandle, VkImageLayout imageLayout,
    const VkClearColorValue* pColor, uint32_t rangeCount,
    const VkImageSubresourceRange* pRanges) {
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdClearColorImage"))
    return;

  Image* image = from_handle(imageHandle);
  if (image == nullptr) {
    KNIZ_ERR("vkCmdClearColorImage was given something that is not a Kniz image");
    return;
  }
  if ((image->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0) {
    KNIZ_ERR("vkCmdClearColorImage needs VK_IMAGE_USAGE_TRANSFER_DST_BIT on the "
             "image");
    return;
  }
  if (pColor == nullptr || pRanges == nullptr || rangeCount == 0)
    return;

  image->layout = imageLayout;

  for (uint32_t i = 0; i < rangeCount; ++i) {
    RecordedCommand command;
    command.kind = RecordedCommand::Kind::ClearColorImage;
    command.image = image;
    command.clear_range = pRanges[i];

    // Which member of VkClearColorValue the application filled in depends on the
    // image's format's component class, so the bytes are taken as they are and
    // interpreted at replay.  That is also why the buffer is exactly 16 bytes
    // rather than a VkClearColorValue: replay has to be able to read it without
    // knowing which member is live.
    static_assert(sizeof(VkClearColorValue) == sizeof(command.clear_value),
                  "a clear value must fit the recorded bytes");
    std::memcpy(command.clear_value, pColor, sizeof(command.clear_value));

    cmd->commands.push_back(std::move(command));
  }
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBufferToImage(
    VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImageHandle,
    VkImageLayout dstImageLayout, uint32_t regionCount,
    const VkBufferImageCopy* pRegions) {
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdCopyBufferToImage"))
    return;

  Buffer* src = from_handle(srcBuffer);
  Image* dst = from_handle(dstImageHandle);
  if (src == nullptr || dst == nullptr) {
    KNIZ_ERR("vkCmdCopyBufferToImage was given something that is not a Kniz "
             "buffer or image");
    return;
  }
  if ((src->usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) == 0) {
    KNIZ_ERR("vkCmdCopyBufferToImage needs VK_BUFFER_USAGE_TRANSFER_SRC_BIT on the "
             "buffer");
    return;
  }
  if ((dst->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0) {
    KNIZ_ERR("vkCmdCopyBufferToImage needs VK_IMAGE_USAGE_TRANSFER_DST_BIT on the "
             "image");
    return;
  }
  if (pRegions == nullptr || regionCount == 0)
    return;

  dst->layout = dstImageLayout;

  RecordedCommand command;
  command.kind = RecordedCommand::Kind::CopyBufferToImage;
  command.src = src;
  command.image = dst;
  command.regions.assign(pRegions, pRegions + regionCount);
  cmd->commands.push_back(std::move(command));
}

VKAPI_ATTR void VKAPI_CALL CmdCopyImageToBuffer(
    VkCommandBuffer commandBuffer, VkImage srcImageHandle, VkImageLayout srcImageLayout,
    VkBuffer dstBuffer, uint32_t regionCount, const VkBufferImageCopy* pRegions) {
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdCopyImageToBuffer"))
    return;

  Image* src = from_handle(srcImageHandle);
  Buffer* dst = from_handle(dstBuffer);
  if (src == nullptr || dst == nullptr) {
    KNIZ_ERR("vkCmdCopyImageToBuffer was given something that is not a Kniz "
             "buffer or image");
    return;
  }
  if ((src->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0) {
    KNIZ_ERR("vkCmdCopyImageToBuffer needs VK_IMAGE_USAGE_TRANSFER_SRC_BIT on the "
             "image");
    return;
  }
  if ((dst->usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) == 0) {
    KNIZ_ERR("vkCmdCopyImageToBuffer needs VK_BUFFER_USAGE_TRANSFER_DST_BIT on the "
             "buffer");
    return;
  }
  if (pRegions == nullptr || regionCount == 0)
    return;

  src->layout = srcImageLayout;

  RecordedCommand command;
  command.kind = RecordedCommand::Kind::CopyImageToBuffer;
  command.dst = dst;
  command.image = src;
  command.regions.assign(pRegions, pRegions + regionCount);
  cmd->commands.push_back(std::move(command));
}

// ---------------------------------------------------------------------------
// Submission
// ---------------------------------------------------------------------------
VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(VkQueue queueHandle, uint32_t submitCount,
                                           const VkSubmitInfo* pSubmits,
                                           VkFence fenceHandle) {
  Queue* queue = from_handle(queueHandle);
  if (queue == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  Device* device = queue->device;
  Instance* instance = device->instance;
  if (!instance->gl_ready) {
    KNIZ_ERR("cannot submit: no OpenGL context is available");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  Fence* fence = nullptr;
  if (fenceHandle != VK_NULL_HANDLE) {
    fence = from_handle(fenceHandle);
    if (fence == nullptr) {
      KNIZ_ERR("the fence is not a Kniz fence");
      return VK_ERROR_INITIALIZATION_FAILED;
    }
  }

  // GL state is per-thread, and Vulkan lets the application submit from any
  // thread, so the context has to be made current before anything is issued.
  if (!gl::make_current()) {
    KNIZ_ERR("could not make the OpenGL context current on this thread");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  for (uint32_t i = 0; i < submitCount; ++i) {
    const VkSubmitInfo& submit = pSubmits[i];

    if (submit.waitSemaphoreCount > 0)
      KNIZ_DBG("ignoring %u wait semaphore(s): submission already completes "
               "before vkQueueSubmit returns",
               submit.waitSemaphoreCount);
    if (submit.signalSemaphoreCount > 0)
      KNIZ_DBG("ignoring %u signal semaphore(s): submission already completes "
               "before vkQueueSubmit returns",
               submit.signalSemaphoreCount);

    // Application writes reach the GL buffer objects first, so that everything
    // the commands read is the data the application last wrote.
    sync_buffers_to_gl(device);

    for (uint32_t j = 0; j < submit.commandBufferCount; ++j) {
      CommandBuffer* cmd = from_handle(submit.pCommandBuffers[j]);
      if (cmd == nullptr) {
        KNIZ_ERR("submit %u references something that is not a Kniz command "
                 "buffer",
                 i);
        continue;
      }
      if (cmd->recording) {
        KNIZ_ERR("submit %u references a command buffer that is still being "
                 "recorded",
                 i);
        continue;
      }
      replay(*cmd);
    }
  }

  // One barrier and one wait for the whole submission.  The barrier makes every
  // write visible to the host; the finish is what lets vkQueueSubmit return a
  // fence that can honestly be reported as signalled.
  gl::f.MemoryBarrier(gl::kAllBarrierBits);
  gl::f.Finish();

  sync_buffers_from_gl(device);

  if (fence != nullptr)
    fence->signaled = true;

  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateFence(VkDevice deviceHandle,
                                           const VkFenceCreateInfo* pCreateInfo,
                                           const VkAllocationCallbacks* pAllocator,
                                           VkFence* pFence) {
  (void)pAllocator;
  if (pFence == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pFence = VK_NULL_HANDLE;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  Fence* fence = new (std::nothrow) Fence();
  if (fence == nullptr)
    return VK_ERROR_OUT_OF_HOST_MEMORY;

  fence->device = device;
  fence->signaled =
      pCreateInfo != nullptr &&
      (pCreateInfo->flags & VK_FENCE_CREATE_SIGNALED_BIT) != 0;

  *pFence = to_handle(fence);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyFence(VkDevice deviceHandle, VkFence fenceHandle,
                                        const VkAllocationCallbacks* pAllocator) {
  (void)pAllocator;
  if (from_handle(deviceHandle) == nullptr)
    return;
  delete from_handle(fenceHandle);
}

VKAPI_ATTR VkResult VKAPI_CALL ResetFences(VkDevice deviceHandle,
                                           uint32_t fenceCount,
                                           const VkFence* pFences) {
  if (from_handle(deviceHandle) == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  for (uint32_t i = 0; i < fenceCount; ++i) {
    Fence* fence = from_handle(pFences[i]);
    if (fence != nullptr)
      fence->signaled = false;
  }
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL GetFenceStatus(VkDevice deviceHandle,
                                              VkFence fenceHandle) {
  if (from_handle(deviceHandle) == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  Fence* fence = from_handle(fenceHandle);
  if (fence == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  return fence->signaled ? VK_SUCCESS : VK_NOT_READY;
}

VKAPI_ATTR VkResult VKAPI_CALL WaitForFences(VkDevice deviceHandle,
                                             uint32_t fenceCount,
                                             const VkFence* pFences,
                                             VkBool32 waitAll,
                                             uint64_t timeout) {
  if (from_handle(deviceHandle) == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  uint32_t signaled = 0;
  for (uint32_t i = 0; i < fenceCount; ++i) {
    Fence* fence = from_handle(pFences[i]);
    if (fence != nullptr && fence->signaled)
      ++signaled;
  }

  // Every submission has already completed by the time vkQueueSubmit returns,
  // so an unsignalled fence means either the application is waiting on work it
  // never submitted or a fence it reset and did not resubmit.  Blocking would
  // hang forever and say nothing; reporting VK_TIMEOUT surfaces the bug.
  const bool satisfied = waitAll ? signaled == fenceCount : signaled > 0;
  if (satisfied)
    return VK_SUCCESS;

  KNIZ_WARN("vkWaitForFences found %u of %u fences unsignalled; submission is "
            "synchronous in Kniz, so this fence was never submitted",
            fenceCount - signaled, fenceCount);
  (void)timeout;
  return VK_TIMEOUT;
}

}  // namespace kniz
