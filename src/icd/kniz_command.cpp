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
#include "kniz_render.h"

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
      case RecordedCommand::Kind::BeginRenderPass:
        render_begin_pass(command);
        break;
      case RecordedCommand::Kind::EndRenderPass:
        render_end_pass(command);
        break;
      case RecordedCommand::Kind::Draw:
        render_draw(command);
        break;
    }
  }
}

void clear_recording_state(CommandBuffer* cmd) {
  cmd->commands.clear();

  cmd->compute = CommandBuffer::BindState{};
  cmd->graphics = CommandBuffer::BindState{};

  cmd->viewport_set = false;
  cmd->viewport = VkViewport{};
  cmd->scissor_set = false;
  cmd->scissor = VkRect2D{};

  cmd->vertex_buffers.clear();
  cmd->index_buffer = nullptr;
  cmd->index_offset = 0;
  cmd->index_type = VK_INDEX_TYPE_UINT16;

  cmd->active_framebuffer = nullptr;
  cmd->active_render_area = VkRect2D{};
}

// The intersection of two rectangles, which is what a Vulkan scissor and a render
// area amount to together.  An empty result is legal and means nothing draws.
VkRect2D intersect(const VkRect2D& a, const VkRect2D& b) {
  const int32_t left = std::max(a.offset.x, b.offset.x);
  const int32_t top = std::max(a.offset.y, b.offset.y);
  const int32_t right =
      std::min(a.offset.x + static_cast<int32_t>(a.extent.width),
               b.offset.x + static_cast<int32_t>(b.extent.width));
  const int32_t bottom =
      std::min(a.offset.y + static_cast<int32_t>(a.extent.height),
               b.offset.y + static_cast<int32_t>(b.extent.height));

  VkRect2D result{};
  result.offset = {left, top};
  result.extent.width = right > left ? static_cast<uint32_t>(right - left) : 0u;
  result.extent.height = bottom > top ? static_cast<uint32_t>(bottom - top) : 0u;
  return result;
}

// Capture the descriptor sets a dispatch or a draw uses.  Vulkan says the
// descriptors a command uses are the ones bound when it is *recorded*, so they
// are resolved here and stored in the recorded command rather than looked up at
// replay.
bool resolve_descriptor_sets(const CommandBuffer::BindState& state,
                             Pipeline* pipeline,
                             std::vector<DescriptorSet*>* out) {
  PipelineLayout* layout = pipeline->layout;
  out->assign(layout->sets.size(), nullptr);

  for (size_t set = 0; set < out->size() && set < state.sets.size(); ++set) {
    DescriptorSet* bound = state.sets[set];
    if (bound != nullptr && bound->layout != layout->sets[set]) {
      KNIZ_ERR("descriptor set %zu was created from a different layout than the "
               "pipeline layout declares for that set; refusing to record the "
               "command",
               set);
      return false;
    }
    (*out)[set] = bound;
  }
  return true;
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

  CommandBuffer::BindState* state = nullptr;
  switch (pipelineBindPoint) {
    case VK_PIPELINE_BIND_POINT_COMPUTE:
      state = &cmd->compute;
      break;
    case VK_PIPELINE_BIND_POINT_GRAPHICS:
      state = &cmd->graphics;
      break;
    default:
      KNIZ_ERR("bind point %d is not implemented; only compute and graphics are",
               static_cast<int>(pipelineBindPoint));
      return;
  }

  Pipeline* pipeline = from_handle(pipelineHandle);
  if (pipeline == nullptr) {
    KNIZ_ERR("the pipeline is not a Kniz pipeline");
    return;
  }
  if (pipeline->bind_point != pipelineBindPoint) {
    KNIZ_ERR("a %s pipeline was bound at the %s bind point",
             pipeline->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS ? "graphics"
                                                                     : "compute",
             pipelineBindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS ? "graphics"
                                                                  : "compute");
    return;
  }

  state->pipeline = pipeline;
  state->layout = pipeline->layout;
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

  CommandBuffer::BindState* state = nullptr;
  switch (pipelineBindPoint) {
    case VK_PIPELINE_BIND_POINT_COMPUTE:
      state = &cmd->compute;
      break;
    case VK_PIPELINE_BIND_POINT_GRAPHICS:
      state = &cmd->graphics;
      break;
    default:
      KNIZ_ERR("bind point %d is not implemented; only compute and graphics are",
               static_cast<int>(pipelineBindPoint));
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
  if (state->layout != nullptr && state->layout != layout) {
    KNIZ_ERR("vkCmdBindDescriptorSets was given a layout that differs from the "
             "bound pipeline's; the sets would not match the shader");
    return;
  }

  if (firstSet + descriptorSetCount > state->sets.size())
    state->sets.resize(firstSet + descriptorSetCount, nullptr);

  for (uint32_t i = 0; i < descriptorSetCount; ++i) {
    DescriptorSet* set = from_handle(pDescriptorSets[i]);
    if (set == nullptr) {
      KNIZ_ERR("descriptor set %u is not a Kniz descriptor set", firstSet + i);
      continue;
    }
    state->sets[firstSet + i] = set;
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

  if (cmd->compute.pipeline == nullptr) {
    KNIZ_ERR("vkCmdDispatch with no compute pipeline bound");
    return;
  }

  Pipeline* pipeline = cmd->compute.pipeline;
  if (pipeline->bind_point != VK_PIPELINE_BIND_POINT_COMPUTE) {
    KNIZ_ERR("vkCmdDispatch was given a pipeline built for another bind point");
    return;
  }

  RecordedCommand command;
  command.kind = RecordedCommand::Kind::Dispatch;
  command.pipeline = pipeline;
  command.group_count[0] = groupCountX;
  command.group_count[1] = groupCountY;
  command.group_count[2] = groupCountZ;

  if (!resolve_descriptor_sets(cmd->compute, pipeline, &command.sets))
    return;

  cmd->commands.push_back(std::move(command));
}

// ---------------------------------------------------------------------------
// Graphics recording
// ---------------------------------------------------------------------------

// Everything vkCmdDraw and vkCmdDrawIndexed have in common.  A draw captures the
// pipeline, its descriptor sets, the vertex buffers and the resolved viewport and
// scissor, because Vulkan defines each of those as the state in force when the
// draw is *recorded* -- not when it is submitted.
bool prepare_draw(CommandBuffer* cmd, const char* name, RecordedCommand* out) {
  if (cmd->active_framebuffer == nullptr) {
    KNIZ_ERR("%s outside a render pass; Vulkan allows a draw only inside one, and "
             "dynamic rendering is not implemented",
             name);
    return false;
  }
  if (cmd->graphics.pipeline == nullptr) {
    KNIZ_ERR("%s with no graphics pipeline bound", name);
    return false;
  }

  Pipeline* pipeline = cmd->graphics.pipeline;
  if (pipeline->render_pass != cmd->active_framebuffer->render_pass) {
    KNIZ_ERR("%s: the bound pipeline was created for a different render pass than "
             "the one the bound framebuffer belongs to",
             name);
    return false;
  }

  out->kind = RecordedCommand::Kind::Draw;
  out->pipeline = pipeline;
  out->vertex_buffers = cmd->vertex_buffers;
  out->index_buffer = cmd->index_buffer;
  out->index_offset = cmd->index_offset;
  out->index_type = cmd->index_type;

  if (pipeline->dynamic_viewport) {
    if (!cmd->viewport_set) {
      KNIZ_ERR("%s: the pipeline declares dynamic viewport state, but "
               "vkCmdSetViewport was never called; Kniz refuses rather than "
               "drawing with a viewport Vulkan leaves undefined",
               name);
      return false;
    }
    out->viewport = cmd->viewport;
  } else {
    out->viewport = pipeline->viewport;
  }

  if (pipeline->dynamic_scissor) {
    if (!cmd->scissor_set) {
      KNIZ_ERR("%s: the pipeline declares dynamic scissor state, but "
               "vkCmdSetScissor was never called",
               name);
      return false;
    }
    out->scissor = intersect(cmd->scissor, cmd->active_render_area);
  } else {
    out->scissor = intersect(pipeline->scissor, cmd->active_render_area);
  }

  return resolve_descriptor_sets(cmd->graphics, pipeline, &out->sets);
}

VKAPI_ATTR void VKAPI_CALL CmdBeginRenderPass(
    VkCommandBuffer commandBuffer, const VkRenderPassBeginInfo* pRenderPassBegin,
    VkSubpassContents contents) {
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdBeginRenderPass"))
    return;
  if (pRenderPassBegin == nullptr) {
    KNIZ_ERR("vkCmdBeginRenderPass was given no begin info");
    return;
  }
  if (cmd->active_framebuffer != nullptr) {
    KNIZ_ERR("vkCmdBeginRenderPass inside another render pass; Vulkan forbids "
             "nesting and Kniz has no way to express it on OpenGL");
    return;
  }
  if (contents != VK_SUBPASS_CONTENTS_INLINE) {
    KNIZ_ERR("secondary command buffers are not implemented");
    return;
  }

  Framebuffer* framebuffer = from_handle(pRenderPassBegin->framebuffer);
  if (framebuffer == nullptr) {
    KNIZ_ERR("vkCmdBeginRenderPass was given something that is not a Kniz "
             "framebuffer");
    return;
  }
  const RenderPass* pass = framebuffer->render_pass;

  // The render area must be inside the framebuffer.  Letting it escape would
  // hand GL coordinates the attachment does not have.
  const VkRect2D& area = pRenderPassBegin->renderArea;
  if (area.offset.x < 0 || area.offset.y < 0 || area.extent.width == 0 ||
      area.extent.height == 0 ||
      static_cast<uint32_t>(area.offset.x) + area.extent.width >
          framebuffer->width ||
      static_cast<uint32_t>(area.offset.y) + area.extent.height >
          framebuffer->height) {
    KNIZ_ERR("the render area (%d,%d %ux%u) is not inside the %ux%u framebuffer",
             area.offset.x, area.offset.y, area.extent.width, area.extent.height,
             framebuffer->width, framebuffer->height);
    return;
  }
  if (area.offset.x != 0 || area.offset.y != 0 ||
      area.extent.width != framebuffer->width ||
      area.extent.height != framebuffer->height) {
    KNIZ_DBG("the render area is a strict subset of the framebuffer; it is "
             "honoured through the viewport and scissor, and the load-op clear is "
             "confined to it the same way");
  }

  // How many clear values the render pass actually needs: one past the highest
  // attachment it clears.
  uint32_t required = 0;
  for (size_t i = 0; i < pass->attachments.size(); ++i) {
    if (pass->attachments[i].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
      required = static_cast<uint32_t>(i) + 1;
  }
  if (pRenderPassBegin->clearValueCount < required) {
    KNIZ_ERR("vkCmdBeginRenderPass supplied %u clear value(s) but the render pass "
             "clears up to attachment %u, so %u are needed",
             pRenderPassBegin->clearValueCount, required - 1, required);
    return;
  }

  RecordedCommand command;
  command.kind = RecordedCommand::Kind::BeginRenderPass;
  command.framebuffer = framebuffer;
  command.render_area = area;

  // Copied because Vulkan allows the caller to free pClearValues as soon as the
  // call returns, and nothing is replayed until submit.
  const uint32_t copy_count =
      std::min(pRenderPassBegin->clearValueCount,
               static_cast<uint32_t>(pass->attachments.size()));
  command.clear_values.resize(copy_count * sizeof(VkClearValue));
  if (copy_count > 0 && pRenderPassBegin->pClearValues != nullptr) {
    std::memcpy(command.clear_values.data(), pRenderPassBegin->pClearValues,
                copy_count * sizeof(VkClearValue));
  }

  cmd->active_framebuffer = framebuffer;
  cmd->active_render_area = area;
  cmd->commands.push_back(std::move(command));
}

VKAPI_ATTR void VKAPI_CALL CmdEndRenderPass(VkCommandBuffer commandBuffer) {
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdEndRenderPass"))
    return;
  if (cmd->active_framebuffer == nullptr) {
    KNIZ_ERR("vkCmdEndRenderPass without a matching vkCmdBeginRenderPass");
    return;
  }

  RecordedCommand command;
  command.kind = RecordedCommand::Kind::EndRenderPass;
  cmd->commands.push_back(std::move(command));
  cmd->active_framebuffer = nullptr;
  cmd->active_render_area = VkRect2D{};
}

VKAPI_ATTR void VKAPI_CALL CmdNextSubpass(VkCommandBuffer commandBuffer,
                                          VkSubpassContents contents) {
  (void)contents;
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdNextSubpass"))
    return;
  KNIZ_ERR("vkCmdNextSubpass is not implemented: a render pass has exactly one "
           "subpass, because OpenGL cannot keep a subpass's results on chip");
}

VKAPI_ATTR void VKAPI_CALL CmdBindVertexBuffers(
    VkCommandBuffer commandBuffer, uint32_t firstBinding, uint32_t bindingCount,
    const VkBuffer* pBuffers, const VkDeviceSize* pOffsets) {
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdBindVertexBuffers"))
    return;
  if (pBuffers == nullptr || pOffsets == nullptr) {
    KNIZ_ERR("vkCmdBindVertexBuffers was given no buffers or no offsets");
    return;
  }

  if (firstBinding + bindingCount > cmd->vertex_buffers.size())
    cmd->vertex_buffers.resize(firstBinding + bindingCount);

  for (uint32_t i = 0; i < bindingCount; ++i) {
    Buffer* buffer = from_handle(pBuffers[i]);
    if (buffer == nullptr) {
      // Vulkan permits a null handle here to reserve a binding, and a draw that
      // never reads it is legal, so this is not an error.
      cmd->vertex_buffers[firstBinding + i] = BoundVertexBuffer{};
      continue;
    }
    if ((buffer->usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) == 0) {
      KNIZ_ERR("binding %u is a buffer created without "
               "VK_BUFFER_USAGE_VERTEX_BUFFER_BIT",
               firstBinding + i);
      continue;
    }
    cmd->vertex_buffers[firstBinding + i] =
        BoundVertexBuffer{buffer, static_cast<uint64_t>(pOffsets[i])};
  }
}

VKAPI_ATTR void VKAPI_CALL CmdBindIndexBuffer(VkCommandBuffer commandBuffer,
                                              VkBuffer bufferHandle,
                                              VkDeviceSize offset,
                                              VkIndexType indexType) {
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdBindIndexBuffer"))
    return;

  Buffer* buffer = from_handle(bufferHandle);
  if (buffer == nullptr) {
    KNIZ_ERR("vkCmdBindIndexBuffer was given something that is not a Kniz buffer");
    return;
  }
  if ((buffer->usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) == 0) {
    KNIZ_ERR("the index buffer was created without "
             "VK_BUFFER_USAGE_INDEX_BUFFER_BIT");
    return;
  }

  cmd->index_buffer = buffer;
  cmd->index_offset = static_cast<uint64_t>(offset);
  cmd->index_type = indexType;
}

VKAPI_ATTR void VKAPI_CALL CmdSetViewport(VkCommandBuffer commandBuffer,
                                          uint32_t firstViewport,
                                          uint32_t viewportCount,
                                          const VkViewport* pViewports) {
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdSetViewport"))
    return;
  if (firstViewport != 0 || viewportCount != 1 || pViewports == nullptr) {
    KNIZ_ERR("exactly one viewport is implemented (first %u, count %u)",
             firstViewport, viewportCount);
    return;
  }
  cmd->viewport = pViewports[0];
  cmd->viewport_set = true;
}

VKAPI_ATTR void VKAPI_CALL CmdSetScissor(VkCommandBuffer commandBuffer,
                                         uint32_t firstScissor,
                                         uint32_t scissorCount,
                                         const VkRect2D* pScissors) {
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdSetScissor"))
    return;
  if (firstScissor != 0 || scissorCount != 1 || pScissors == nullptr) {
    KNIZ_ERR("exactly one scissor is implemented (first %u, count %u)",
             firstScissor, scissorCount);
    return;
  }
  cmd->scissor = pScissors[0];
  cmd->scissor_set = true;
}

VKAPI_ATTR void VKAPI_CALL CmdDraw(VkCommandBuffer commandBuffer,
                                   uint32_t vertexCount, uint32_t instanceCount,
                                   uint32_t firstVertex, uint32_t firstInstance) {
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdDraw"))
    return;

  if (instanceCount != 1 || firstInstance != 0) {
    KNIZ_ERR("instanced drawing is not implemented (%u instance(s) starting at "
             "%u); it needs glDrawArraysInstanced",
             instanceCount, firstInstance);
    return;
  }
  if (vertexCount == 0)  // Vulkan defines a zero count as a no-op
    return;

  RecordedCommand command;
  if (!prepare_draw(cmd, "vkCmdDraw", &command))
    return;
  command.indexed = false;
  command.first_vertex = firstVertex;
  command.vertex_count = vertexCount;
  cmd->commands.push_back(std::move(command));
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexed(
    VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount,
    uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdDrawIndexed"))
    return;

  if (instanceCount != 1 || firstInstance != 0) {
    KNIZ_ERR("instanced drawing is not implemented (%u instance(s) starting at "
             "%u); it needs glDrawElementsInstanced",
             instanceCount, firstInstance);
    return;
  }
  if (indexCount == 0)
    return;

  RecordedCommand command;
  if (!prepare_draw(cmd, "vkCmdDrawIndexed", &command))
    return;
  command.indexed = true;
  command.first_index = firstIndex;
  command.index_count = indexCount;
  command.vertex_offset = vertexOffset;
  cmd->commands.push_back(std::move(command));
}

VKAPI_ATTR void VKAPI_CALL CmdClearAttachments(
    VkCommandBuffer commandBuffer, uint32_t attachmentCount,
    const VkClearAttachment* pAttachments, uint32_t rectCount,
    const VkClearRect* pRects) {
  (void)attachmentCount;
  (void)pAttachments;
  (void)rectCount;
  (void)pRects;
  CommandBuffer* cmd = from_handle(commandBuffer);
  if (!recording(cmd, "vkCmdClearAttachments"))
    return;
  // A clear of a rectangle inside a render pass has no single GL equivalent: the
  // scissor would have to be set and the write mask forced, and restoring the
  // state afterwards is the part that goes wrong.  Squaring that with the
  // pipeline's own scissor is worth doing deliberately rather than here.
  KNIZ_ERR("vkCmdClearAttachments is not implemented; use a render pass load-op "
           "clear, or vkCmdClearColorImage outside the render pass");
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
