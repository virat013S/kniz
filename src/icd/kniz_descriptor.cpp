// Descriptor set layouts, pipeline layouts, pools, sets and updates.
//
// Two descriptor types are supported: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, which
// arrived with compute in M1, and VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, which came
// with the rest of M2c.  They are the same shape -- a buffer, an offset and a
// range -- and differ only in which GL binding namespace they are indexed by,
// which is why supporting the second one is a small change here and a larger one
// in the shader translation, where the two namespaces have to be kept apart.
//
// Every other type has a different GL binding point, a different layout rule or
// both, so they are refused at the earliest point the application reveals its
// intent.  That produces one clear error instead of a later mysterious one, and
// it is what makes tests/uniform.cpp's assertion that a sampled image is refused
// a statement about the driver rather than about a bug.

#include "kniz_common.h"
#include "kniz_descriptor.h"
#include "kniz_objects.h"

#include <algorithm>
#include <cstring>
#include <new>

namespace kniz {
namespace {

bool descriptor_type_supported(VkDescriptorType type) {
  return type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
         type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
}

const char* descriptor_type_name(VkDescriptorType type) {
  switch (type) {
    case VK_DESCRIPTOR_TYPE_SAMPLER:
      return "VK_DESCRIPTOR_TYPE_SAMPLER";
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      return "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER";
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      return "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE";
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      return "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE";
    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      return "VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER";
    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
      return "VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER";
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      return "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER";
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      return "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER";
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      return "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC";
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
      return "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC";
    case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
      return "VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT";
    default:
      return "an unrecognised descriptor type";
  }
}

}  // namespace

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorSetLayout(
    VkDevice deviceHandle, const VkDescriptorSetLayoutCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDescriptorSetLayout* pSetLayout) {
  (void)pAllocator;
  if (pSetLayout == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pSetLayout = VK_NULL_HANDLE;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (pCreateInfo == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

#ifdef VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR
  if ((pCreateInfo->flags &
       VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR) != 0) {
    KNIZ_ERR("push descriptor sets are not implemented");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
#endif

  DescriptorSetLayout* layout = new (std::nothrow) DescriptorSetLayout();
  if (layout == nullptr)
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  layout->device = device;

  for (uint32_t i = 0; i < pCreateInfo->bindingCount; ++i) {
    const VkDescriptorSetLayoutBinding& binding = pCreateInfo->pBindings[i];

    if (!descriptor_type_supported(binding.descriptorType)) {
      KNIZ_ERR("binding %u uses %s; Kniz implements "
               "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER and "
               "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER",
               binding.binding, descriptor_type_name(binding.descriptorType));
      delete layout;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (binding.descriptorCount != 1) {
      KNIZ_ERR("binding %u declares %u descriptors; arrays of descriptors are "
               "not implemented",
               binding.binding, binding.descriptorCount);
      delete layout;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    DescriptorSetLayoutBinding stored{};
    stored.binding = binding.binding;
    stored.type = binding.descriptorType;
    stored.count = binding.descriptorCount;
    stored.stages = binding.stageFlags;
    layout->bindings.push_back(stored);
    if (binding.binding + 1 > layout->binding_slots)
      layout->binding_slots = binding.binding + 1;
  }

  // Binding order, so that the GL binding points derived from this layout do not
  // depend on the order the application declared its bindings in.
  std::sort(layout->bindings.begin(), layout->bindings.end(),
            [](const DescriptorSetLayoutBinding& a,
               const DescriptorSetLayoutBinding& b) {
              return a.binding < b.binding;
            });

  *pSetLayout = to_handle(layout);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorSetLayout(
    VkDevice deviceHandle, VkDescriptorSetLayout setLayoutHandle,
    const VkAllocationCallbacks* pAllocator) {
  (void)pAllocator;
  if (from_handle(deviceHandle) == nullptr)
    return;
  delete from_handle(setLayoutHandle);
}

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineLayout(
    VkDevice deviceHandle, const VkPipelineLayoutCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkPipelineLayout* pPipelineLayout) {
  (void)pAllocator;
  if (pPipelineLayout == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pPipelineLayout = VK_NULL_HANDLE;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr || pCreateInfo == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  if (pCreateInfo->setLayoutCount > 4) {
    KNIZ_ERR("pipeline layout declares %u sets; this driver reports a limit of "
             "4 (maxBoundDescriptorSets)",
             pCreateInfo->setLayoutCount);
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  if (pCreateInfo->pushConstantRangeCount > 0 &&
      pCreateInfo->pPushConstantRanges == nullptr) {
    KNIZ_ERR("pipeline layout declares %u push constant range(s) but provides "
             "no array of them",
             pCreateInfo->pushConstantRangeCount);
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  PipelineLayout* layout = new (std::nothrow) PipelineLayout();
  if (layout == nullptr)
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  layout->device = device;

  for (uint32_t i = 0; i < pCreateInfo->setLayoutCount; ++i) {
    DescriptorSetLayout* set_layout =
        from_handle(pCreateInfo->pSetLayouts[i]);
    if (set_layout == nullptr) {
      KNIZ_ERR("set layout %u is not a Kniz descriptor set layout", i);
      delete layout;
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    // Order is significant: the index in this vector is the set number.
    layout->sets.push_back(set_layout);
  }

  // ---- push constant ranges ----------------------------------------------
  //
  // Vulkan's rules are checked rather than assumed, because every one of them
  // has a consequence here: a range that runs past the limit cannot be uploaded,
  // an unaligned offset has no GLSL equivalent, and two ranges that overlap
  // within one stage would need the shader's two halves of the block to be
  // uploaded from different places.
  for (uint32_t i = 0; i < pCreateInfo->pushConstantRangeCount; ++i) {
    const VkPushConstantRange& range = pCreateInfo->pPushConstantRanges[i];

    if (range.stageFlags == 0) {
      KNIZ_ERR("push constant range %u names no shader stages", i);
      delete layout;
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (range.size == 0) {
      KNIZ_ERR("push constant range %u is empty", i);
      delete layout;
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (range.offset % 4 != 0) {
      KNIZ_ERR("push constant range %u starts at offset %u, which is not a "
               "multiple of 4; Vulkan requires one",
               i, range.offset);
      delete layout;
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (range.offset + range.size > kMaxPushConstantsSize) {
      KNIZ_ERR("push constant range %u covers bytes %u..%u, but this driver "
               "reports maxPushConstantsSize = %u",
               i, range.offset, range.offset + range.size,
               kMaxPushConstantsSize);
      delete layout;
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    for (uint32_t previous = 0; previous < i; ++previous) {
      const VkPushConstantRange& other = pCreateInfo->pPushConstantRanges[previous];
      if ((other.stageFlags & range.stageFlags) == 0)
        continue;
      if (range.offset < other.offset + other.size &&
          other.offset < range.offset + range.size) {
        KNIZ_ERR("push constant ranges %u and %u overlap for a common stage",
                 previous, i);
        delete layout;
        return VK_ERROR_INITIALIZATION_FAILED;
      }
    }

    layout->push_constant_ranges.push_back(range);
    layout->has_push_constants = true;
    layout->push_constant_size =
        std::max(layout->push_constant_size, range.offset + range.size);
  }

  // ---- the flat GL binding map -------------------------------------------
  //
  // One namespace per resource class, each numbered from zero, because GL
  // indexes uniform buffers and shader storage buffers independently.  The
  // binding number alone cannot be used: (set 0, binding 0) and (set 1,
  // binding 0) are both legal and would collide.
  //
  // Each set layout keeps its bindings in binding order -- sorted when the set
  // layout was created -- so the numbering depends on which bindings exist
  // rather than on the order an application happened to declare them in.
  {
    uint32_t next_uniform = 0;
    uint32_t next_storage = 0;

    // Push constants take uniform binding point 0, ahead of every descriptor.
    // The reason is not that they deserve the first slot: it is that the index
    // has to be the *same* in every layout that can share a push, and Vulkan
    // lets an application write push constants with one layout and draw with
    // another that is merely "compatible for push constants" -- which constrains
    // the push constant ranges and nothing else, so the descriptor set layouts,
    // and with them the number of uniform buffers, may differ.  Numbering the
    // descriptors first would make the index depend on how many uniform buffers
    // a layout happens to declare, and the values would land where the drawing
    // shader is not looking.  Numbering the push constants first makes the index
    // depend only on whether the layout declares any ranges at all, which every
    // compatible layout agrees about.
    if (layout->has_push_constants) {
      layout->push_constant_gl_index = 0;
      next_uniform = 1;
    }

    for (uint32_t set = 0; set < layout->sets.size(); ++set) {
      for (const DescriptorSetLayoutBinding& binding :
           layout->sets[set]->bindings) {
        GlBinding mapped;
        mapped.set = set;
        mapped.binding = binding.binding;
        mapped.type = binding.type;
        mapped.gl_index = descriptor_is_uniform(binding.type) ? next_uniform++
                                                             : next_storage++;
        layout->bindings.push_back(mapped);
      }
    }
    layout->uniform_count = next_uniform;
  }

  if (layout->has_push_constants) {
    KNIZ_DBG("pipeline layout: %zu descriptor binding(s), push constants %u "
             "byte(s) at GL uniform binding %u, uniform descriptors from %u",
             layout->bindings.size(), layout->push_constant_size,
             layout->push_constant_gl_index,
             layout->push_constant_gl_index + 1);
  }

  *pPipelineLayout = to_handle(layout);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipelineLayout(
    VkDevice deviceHandle, VkPipelineLayout pipelineLayoutHandle,
    const VkAllocationCallbacks* pAllocator) {
  (void)pAllocator;
  if (from_handle(deviceHandle) == nullptr)
    return;
  delete from_handle(pipelineLayoutHandle);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorPool(
    VkDevice deviceHandle, const VkDescriptorPoolCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDescriptorPool* pDescriptorPool) {
  (void)pAllocator;
  if (pDescriptorPool == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pDescriptorPool = VK_NULL_HANDLE;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr || pCreateInfo == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  DescriptorPool* pool = new (std::nothrow) DescriptorPool();
  if (pool == nullptr)
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  pool->device = device;
  pool->max_sets = pCreateInfo->maxSets;

  *pDescriptorPool = to_handle(pool);
  return VK_SUCCESS;
}

// Frees every set the pool handed out.  Unlike vkResetDescriptorPool this is
// what happens at destruction, so both share it.
void release_pool_sets(DescriptorPool* pool) {
  for (DescriptorSet* set : pool->allocated)
    delete set;
  pool->allocated.clear();
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorPool(
    VkDevice deviceHandle, VkDescriptorPool descriptorPoolHandle,
    const VkAllocationCallbacks* pAllocator) {
  (void)pAllocator;
  if (from_handle(deviceHandle) == nullptr)
    return;
  DescriptorPool* pool = from_handle(descriptorPoolHandle);
  if (pool == nullptr)
    return;
  release_pool_sets(pool);
  delete pool;
}

VKAPI_ATTR VkResult VKAPI_CALL ResetDescriptorPool(
    VkDevice deviceHandle, VkDescriptorPool descriptorPoolHandle,
    VkDescriptorPoolResetFlags flags) {
  (void)flags;
  if (from_handle(deviceHandle) == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  DescriptorPool* pool = from_handle(descriptorPoolHandle);
  if (pool == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  release_pool_sets(pool);
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateDescriptorSets(
    VkDevice deviceHandle, const VkDescriptorSetAllocateInfo* pAllocateInfo,
    VkDescriptorSet* pDescriptorSets) {
  if (pDescriptorSets == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr || pAllocateInfo == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  DescriptorPool* pool = from_handle(pAllocateInfo->descriptorPool);
  if (pool == nullptr) {
    KNIZ_ERR("descriptor pool is not a Kniz descriptor pool");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  if (pool->allocated.size() + pAllocateInfo->descriptorSetCount > pool->max_sets) {
    KNIZ_ERR("pool would hold %zu sets, more than the %u it was created with",
             pool->allocated.size() + pAllocateInfo->descriptorSetCount,
             pool->max_sets);
    return VK_ERROR_OUT_OF_POOL_MEMORY;
  }

  for (uint32_t i = 0; i < pAllocateInfo->descriptorSetCount; ++i) {
    DescriptorSetLayout* layout = from_handle(pAllocateInfo->pSetLayouts[i]);
    if (layout == nullptr) {
      KNIZ_ERR("set %u asks for a layout that is not a Kniz descriptor set layout", i);
      return VK_ERROR_INITIALIZATION_FAILED;
    }

    DescriptorSet* set = new (std::nothrow) DescriptorSet();
    if (set == nullptr)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

    set->device = device;
    set->pool = pool;
    set->layout = layout;
    set->bindings.resize(layout->binding_slots);

    pool->allocated.push_back(set);
    pDescriptorSets[i] = to_handle(set);
  }

  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FreeDescriptorSets(
    VkDevice deviceHandle, VkDescriptorPool descriptorPoolHandle,
    uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets) {
  Device* device = from_handle(deviceHandle);
  DescriptorPool* pool = from_handle(descriptorPoolHandle);
  if (device == nullptr || pool == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  for (uint32_t i = 0; i < descriptorSetCount; ++i) {
    DescriptorSet* set = from_handle(pDescriptorSets[i]);
    if (set == nullptr)
      continue;
    for (auto it = pool->allocated.begin(); it != pool->allocated.end(); ++it) {
      if (*it == set) {
        pool->allocated.erase(it);
        break;
      }
    }
    delete set;
  }

  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSets(
    VkDevice deviceHandle, uint32_t descriptorWriteCount,
    const VkWriteDescriptorSet* pDescriptorWrites,
    uint32_t descriptorCopyCount,
    const VkCopyDescriptorSet* pDescriptorCopies) {
  (void)pDescriptorCopies;
  Device* device = from_handle(deviceHandle);
  if (device == nullptr)
    return;

  if (descriptorCopyCount > 0)
    KNIZ_WARN("%u descriptor copy operation(s) ignored; vkCopyDescriptorSet is "
              "not implemented",
              descriptorCopyCount);

  for (uint32_t i = 0; i < descriptorWriteCount; ++i) {
    const VkWriteDescriptorSet& write = pDescriptorWrites[i];

    DescriptorSet* set = from_handle(write.dstSet);
    if (set == nullptr) {
      KNIZ_ERR("write %u targets something that is not a Kniz descriptor set", i);
      continue;
    }
    if (write.dstBinding >= set->bindings.size()) {
      KNIZ_ERR("write %u targets binding %u, which the layout does not declare",
               i, write.dstBinding);
      continue;
    }
    if (write.dstArrayElement != 0 || write.descriptorCount != 1) {
      KNIZ_ERR("write %u touches descriptor elements; this driver supports one "
               "descriptor per binding",
               i);
      continue;
    }
    if (!descriptor_type_supported(write.descriptorType)) {
      KNIZ_ERR("write %u uses %s, which is not implemented",
               i, descriptor_type_name(write.descriptorType));
      continue;
    }
    // The layout says what is at this binding; Vulkan requires the write to
    // agree, and a write that disagrees would put uniform data where the shader
    // reads a storage block.
    {
      const DescriptorSetLayoutBinding* declared = nullptr;
      for (const DescriptorSetLayoutBinding& candidate : set->layout->bindings)
        if (candidate.binding == write.dstBinding)
          declared = &candidate;
      if (declared == nullptr) {
        KNIZ_ERR("write %u targets binding %u, which the layout does not declare",
                 i, write.dstBinding);
        continue;
      }
      if (declared->type != write.descriptorType) {
        KNIZ_ERR("write %u uses %s for binding %u, but the layout declares %s",
                 i, descriptor_type_name(write.descriptorType),
                 write.dstBinding, descriptor_type_name(declared->type));
        continue;
      }
    }
    if (write.pBufferInfo == nullptr) {
      KNIZ_ERR("write %u carries no buffer info", i);
      continue;
    }

    Buffer* buffer = from_handle(write.pBufferInfo->buffer);
    if (buffer == nullptr) {
      KNIZ_ERR("write %u references something that is not a Kniz buffer", i);
      continue;
    }

    DescriptorBinding& binding = set->bindings[write.dstBinding];
    binding.bound = true;
    binding.buffer = buffer;
    binding.offset = write.pBufferInfo->offset;
    // VK_WHOLE_SIZE is resolved at bind time, when the buffer's size is known.
    binding.range = write.pBufferInfo->range;
  }
}

// ---------------------------------------------------------------------------
// Binding descriptors, and push constants, to OpenGL
// ---------------------------------------------------------------------------

bool bind_descriptor_sets(const Pipeline* pipeline,
                          const std::vector<DescriptorSet*>& sets,
                          const char* what) {
  if (pipeline == nullptr || pipeline->layout == nullptr)
    return true;

  bool ok = true;

  // The pipeline layout decided, at creation, which GL binding point each
  // (set, binding) got; the shader was generated from the same table.  So this
  // loop only has to read the table and bind what the sets hold.
  for (const GlBinding& mapping : pipeline->used_bindings) {
    if (mapping.set >= sets.size()) {
      KNIZ_ERR("%s needs set %u, which the pipeline layout declares, but the "
               "command buffer bound fewer sets",
               what, mapping.set);
      ok = false;
      continue;
    }

    DescriptorSet* set = sets[mapping.set];
    if (set == nullptr) {
      KNIZ_ERR("%s reads set %u binding %u, but no descriptor set is bound there",
               what, mapping.set, mapping.binding);
      ok = false;
      continue;
    }

    if (mapping.binding >= set->bindings.size()) {
      KNIZ_ERR("the bound set has no binding %u", mapping.binding);
      ok = false;
      continue;
    }

    const DescriptorBinding& binding = set->bindings[mapping.binding];
    if (!binding.bound || binding.buffer == nullptr) {
      KNIZ_ERR("set %u binding %u is never written; %s would read undefined "
               "data",
               mapping.set, mapping.binding, what);
      ok = false;
      continue;
    }

    Buffer* buffer = binding.buffer;
    uint64_t range = binding.range;
    if (range == VK_WHOLE_SIZE || range == 0 ||
        range > buffer->size - binding.offset)
      range = buffer->size - binding.offset;

    // GL requires an indexed buffer range to start at a multiple of the
    // alignment for its class.  Vulkan caps both of the equivalent limits at
    // 256, so on a driver that wants more this is the point where the two
    // cannot be reconciled -- which is why the driver says so once, at context
    // creation, rather than per failed bind.
    const bool uniform = descriptor_is_uniform(mapping.type);
    const int alignment = static_cast<int>(
        uniform ? buffer->device->instance->gl.ubo_offset_alignment
                : buffer->device->instance->gl.ssbo_offset_alignment);
    if (alignment > 0 && binding.offset % static_cast<uint64_t>(alignment) != 0) {
      KNIZ_ERR("buffer offset %llu is not a multiple of this driver's %s "
               "alignment (%d); the descriptor cannot be bound",
               static_cast<unsigned long long>(binding.offset),
               uniform ? "uniform buffer" : "shader storage buffer", alignment);
      ok = false;
      continue;
    }

    const gl::Enum target =
        uniform ? gl::kUniformBuffer : gl::kShaderStorageBuffer;
    KNIZ_TRACE("%s: set %u binding %u -> GL %s[%u] = buffer %u, offset %llu, "
               "range %llu",
               what, mapping.set, mapping.binding,
               uniform ? "UNIFORM_BUFFER" : "SHADER_STORAGE_BUFFER",
               mapping.gl_index, buffer->gl_name,
               static_cast<unsigned long long>(binding.offset),
               static_cast<unsigned long long>(range));

    gl::f.BindBufferRange(target, mapping.gl_index, buffer->gl_name,
                          static_cast<gl::IntPtr>(binding.offset),
                          gl::as_size_ptr(range));
  }

  if (gl::report_errors("after binding descriptor sets"))
    ok = false;
  return ok;
}

bool create_push_constant_buffer(Device* device) {
  if (device == nullptr)
    return false;
  if (device->push_constant_buffer != 0)
    return true;

  if (!device->instance->gl_ready) {
    KNIZ_ERR("no OpenGL context is available for the push constant buffer");
    return false;
  }
  if (!gl::make_current())
    return false;

  gl::f.GenBuffers(1, &device->push_constant_buffer);
  if (device->push_constant_buffer == 0) {
    KNIZ_ERR("glGenBuffers failed for the push constant buffer");
    return false;
  }

  // GL_DYNAMIC_DRAW because the contents are replaced every time a
  // vkCmdPushConstants replays, which is the definition of dynamic.
  gl::f.BindBuffer(gl::kUniformBuffer, device->push_constant_buffer);
  gl::f.BufferData(gl::kUniformBuffer,
                   static_cast<gl::SizePtr>(kMaxPushConstantsSize), nullptr,
                   gl::kDynamicDraw);
  gl::f.BindBuffer(gl::kUniformBuffer, 0);
  device->push_constant_capacity = kMaxPushConstantsSize;

  if (gl::report_errors("after creating the push constant buffer")) {
    destroy_push_constant_buffer(device);
    return false;
  }
  return true;
}

void destroy_push_constant_buffer(Device* device) {
  if (device == nullptr || device->push_constant_buffer == 0)
    return;
  if (device->instance->gl_ready && gl::make_current())
    gl::f.DeleteBuffers(1, &device->push_constant_buffer);
  device->push_constant_buffer = 0;
  device->push_constant_capacity = 0;
}

bool reset_push_constants(Device* device) {
  if (device == nullptr || device->push_constant_buffer == 0)
    return true;

  // kMaxPushConstantsSize zeroes is a small enough upload that doing it per
  // submission is cheaper than tracking whether the contents are known -- and
  // Vulkan leaves push constants undefined at the start of a command buffer, so
  // a defined starting point is a superset of what it promises.
  const uint8_t zeroes[kMaxPushConstantsSize] = {};
  gl::f.BindBuffer(gl::kUniformBuffer, device->push_constant_buffer);
  gl::f.BufferSubData(gl::kUniformBuffer, 0,
                      static_cast<gl::SizePtr>(kMaxPushConstantsSize), zeroes);
  gl::f.BindBuffer(gl::kUniformBuffer, 0);
  return !gl::report_errors("after resetting the push constant buffer");
}

bool apply_push_constants(Device* device, const PipelineLayout* layout,
                          uint32_t offset, const void* bytes, uint32_t size) {
  if (device == nullptr || layout == nullptr)
    return false;
  if (device->push_constant_buffer == 0) {
    KNIZ_ERR("push constants were written but this device has no push constant "
             "buffer; a pipeline layout that declares ranges creates one");
    return false;
  }
  if (size == 0)
    return true;

  if (offset + size > device->push_constant_capacity) {
    KNIZ_ERR("push constants cover bytes %u..%u, beyond the %u this device "
             "allocated",
             offset, offset + size, device->push_constant_capacity);
    return false;
  }

  gl::f.BindBuffer(gl::kUniformBuffer, device->push_constant_buffer);
  gl::f.BufferSubData(gl::kUniformBuffer, static_cast<gl::IntPtr>(offset),
                      static_cast<gl::SizePtr>(size), bytes);

  // Binding the whole buffer, not just the range the application wrote: the
  // shader may read any of the bytes its block declares, and Vulkan says the
  // ones this command did not cover keep their previous values, which are the
  // ones the device buffer already holds.
  gl::f.BindBufferRange(gl::kUniformBuffer, layout->push_constant_gl_index,
                        device->push_constant_buffer, 0,
                        static_cast<gl::SizePtr>(kMaxPushConstantsSize));
  gl::f.BindBuffer(gl::kUniformBuffer, 0);
  return !gl::report_errors("after uploading push constants");
}

}  // namespace kniz
