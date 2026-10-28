// Descriptor set layouts, pipeline layouts, pools, sets and updates.
//
// Five descriptor types are supported, in three GL namespaces:
//
//   * VK_DESCRIPTOR_TYPE_STORAGE_BUFFER (M1) and VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
//     (M2c) are a buffer, an offset and a range, indexed by two separate buffer
//     binding namespaces.
//   * VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
//     and VK_DESCRIPTOR_TYPE_SAMPLER (M2d) are texture units: GLSL has one
//     sampler type that carries both an image and its filter state, so a unit
//     holds a texture, a sampler object, or both.
//
// Every other type has a different GL binding point, a different layout rule or
// both, so they are refused at the earliest point the application reveals its
// intent.  That produces one clear error instead of a later mysterious one.

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
         type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
         type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
         type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
         type == VK_DESCRIPTOR_TYPE_SAMPLER;
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

// Whether this view can be sampled, checked where the application says it wants
// to sample so that the refusal names a reason instead of turning into black
// pixels later.
//
// The mip rule is the interesting one.  Kniz creates the full storage a
// mipmapped image asks for, but only level 0 is ever written -- no copy or clear
// addresses another level -- so levels 1 and up hold whatever the driver left
// there.  Sampling such an image would read that.  Refusing is the honest
// answer until the transfer path covers mip levels.
bool view_is_sampleable(const ImageView* view, uint32_t write) {
  const Image* image = view->image;

  if (!image->storage_allocated) {
    KNIZ_ERR("write %u samples an image that has no memory bound to it", write);
    return false;
  }
  if ((image->usage & VK_IMAGE_USAGE_SAMPLED_BIT) == 0) {
    KNIZ_ERR("write %u samples an image created without "
             "VK_IMAGE_USAGE_SAMPLED_BIT",
             write);
    return false;
  }
  if (view->view_type != VK_IMAGE_VIEW_TYPE_2D) {
    KNIZ_ERR("write %u samples a %s view; Kniz samples one 2D layer",
             write,
             view->view_type == VK_IMAGE_VIEW_TYPE_2D_ARRAY ? "2D array" : "3D");
    return false;
  }

  const uint32_t layers =
      view->range.layerCount == VK_REMAINING_ARRAY_LAYERS
          ? image->array_layers - view->range.baseArrayLayer
          : view->range.layerCount;
  if (view->range.baseArrayLayer != 0 || layers != 1) {
    KNIZ_ERR("write %u samples layer %u of %u; Kniz samples one layer",
             write, view->range.baseArrayLayer, layers);
    return false;
  }

  const uint32_t levels =
      view->range.levelCount == VK_REMAINING_MIP_LEVELS
          ? image->mip_levels - view->range.baseMipLevel
          : view->range.levelCount;
  if (view->range.baseMipLevel != 0 || levels != 1) {
    KNIZ_ERR("write %u samples mip level %u of %u; Kniz samples level 0",
             write, view->range.baseMipLevel, levels);
    return false;
  }
  if (image->mip_levels != 1) {
    KNIZ_ERR("write %u samples an image with %u mip levels, but only level 0 is "
             "ever written, so the rest is undefined; Kniz refuses to sample it",
             write, image->mip_levels);
    return false;
  }

  return true;
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
      KNIZ_ERR("binding %u uses %s; Kniz implements storage buffers, uniform "
               "buffers, combined image samplers, sampled images and samplers",
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
    uint32_t next_texture = 0;

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
        switch (descriptor_class(binding.type)) {
          case DescriptorClass::UniformBuffer:
            mapped.gl_index = next_uniform++;
            break;
          case DescriptorClass::StorageBuffer:
            mapped.gl_index = next_storage++;
            break;
          case DescriptorClass::TextureUnit:
            mapped.gl_index = next_texture++;
            break;
        }
        layout->bindings.push_back(mapped);
      }
    }
    layout->uniform_count = next_uniform;
    layout->texture_count = next_texture;
  }

  // A shader that does texelFetch on a separate image needs one texture unit
  // more than the layout's descriptors occupy, because the sampler GLSL insists
  // on is injected past the end of the mapping.  Checking it here is what keeps
  // the shader translation from addressing a unit the driver does not have.
  if (layout->texture_count + 1 >
      static_cast<uint32_t>(gl::info().max_texture_units)) {
    KNIZ_ERR("pipeline layout needs %u texture unit(s), one more than its %u "
             "image and sampler descriptors for the sampler a texelFetch needs, "
             "but this driver has %d",
             layout->texture_count + 1, layout->texture_count,
             gl::info().max_texture_units);
    delete layout;
    return VK_ERROR_FEATURE_NOT_PRESENT;
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
    if (descriptor_class(write.descriptorType) == DescriptorClass::TextureUnit) {
      // Image and sampler descriptors.  Vulkan fills pImageInfo for all three
      // types, and each type reads the fields it needs: a combined image sampler
      // reads both, a sampled image reads the view, a sampler reads the sampler.
      if (write.pImageInfo == nullptr) {
        KNIZ_ERR("write %u carries no image info", i);
        continue;
      }

      ImageView* view = nullptr;
      Sampler* sampler = nullptr;

      if (write.descriptorType != VK_DESCRIPTOR_TYPE_SAMPLER) {
        view = from_handle(write.pImageInfo->imageView);
        if (view == nullptr) {
          KNIZ_ERR("write %u references something that is not a Kniz image view",
                   i);
          continue;
        }
        if (!view_is_sampleable(view, i))
          continue;
      }
      if (write.descriptorType != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {
        sampler = from_handle(write.pImageInfo->sampler);
        if (sampler == nullptr) {
          KNIZ_ERR("write %u references something that is not a Kniz sampler", i);
          continue;
        }
      }

      DescriptorBinding& binding = set->bindings[write.dstBinding];
      binding.bound = true;
      binding.view = view;
      binding.sampler = sampler;
      continue;
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

namespace {

// Bind one texture-unit descriptor: the texture at `mapping.gl_index`, and the
// sampler object that goes with it.
//
// The sampler object always goes to the *same* unit as the texture, which is
// what makes the GLSL `sampler2D` the shader declares read the pair the
// application meant.  For a combined image sampler both come from one
// descriptor.  For a separate image the sampler comes from the descriptor the
// shader paired it with -- recorded at translation time in sampler_set and
// sampler_binding -- because the image's own binding does not say which sampler
// reads it.  A separate image with no pairing is read only by texelFetch, which
// ignores sampler state, so unit 0 (no sampler object) is the right binding.
bool bind_texture_unit(const GlBinding& mapping,
                       const std::vector<DescriptorSet*>& sets,
                       const char* what) {
  DescriptorSet* set = sets[mapping.set];
  const DescriptorBinding& binding = set->bindings[mapping.binding];

  if (!binding.bound || binding.view == nullptr) {
    KNIZ_ERR("set %u binding %u is never written; %s would sample an image that "
             "was never bound there",
             mapping.set, mapping.binding, what);
    return false;
  }

  const Image* image = binding.view->image;
  gl::f.ActiveTexture(gl::kTexture0 + mapping.gl_index);
  gl::f.BindTexture(gl::kTexture2D, image->gl_texture);

  gl::Uint sampler = 0;
  if (mapping.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
    if (binding.sampler == nullptr) {
      KNIZ_ERR("set %u binding %u is a combined image sampler with no sampler; "
               "%s would filter the image with whatever was left bound",
               mapping.set, mapping.binding, what);
      return false;
    }
    sampler = binding.sampler->gl_sampler;
  } else if (mapping.sampler_set != kNoBinding) {
    if (mapping.sampler_set >= sets.size() ||
        sets[mapping.sampler_set] == nullptr ||
        mapping.sampler_binding >= sets[mapping.sampler_set]->bindings.size()) {
      KNIZ_ERR("set %u binding %u is read through a sampler at set %u binding "
               "%u, but no such descriptor is bound",
               mapping.set, mapping.binding, mapping.sampler_set,
               mapping.sampler_binding);
      return false;
    }
    const DescriptorBinding& sampler_binding =
        sets[mapping.sampler_set]->bindings[mapping.sampler_binding];
    if (!sampler_binding.bound || sampler_binding.sampler == nullptr) {
      KNIZ_ERR("set %u binding %u is read through a sampler at set %u binding "
               "%u, which is never written",
               mapping.set, mapping.binding, mapping.sampler_set,
               mapping.sampler_binding);
      return false;
    }
    sampler = sampler_binding.sampler->gl_sampler;
  }

  KNIZ_TRACE("%s: set %u binding %u -> GL texture unit %u = texture %u, "
             "sampler %u",
             what, mapping.set, mapping.binding, mapping.gl_index,
             image->gl_texture, sampler);

  gl::f.BindSampler(mapping.gl_index, sampler);
  return true;
}

}  // namespace

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

    if (descriptor_class(mapping.type) == DescriptorClass::TextureUnit) {
      ok = bind_texture_unit(mapping, sets, what) && ok;
      continue;
    }

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
