// Descriptor set layouts, pipeline layouts, pools, sets and updates.
//
// M1 supports exactly one descriptor type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
// one descriptor per binding, no arrays and no dynamic offsets.  That is not a
// limitation of the design so much as of how much can be verified at once --
// shader storage buffers are the whole of what a compute dispatch needs, and
// every other type has a different GL binding point, a different layout rule or
// both.  Refusing the others here, at the earliest point the application
// reveals its intent, produces one clear error instead of a later mysterious
// one.

#include "kniz_common.h"
#include "kniz_objects.h"

#include <new>

namespace kniz {
namespace {

bool descriptor_type_supported(VkDescriptorType type) {
  return type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
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
      KNIZ_ERR("binding %u uses %s; M1 implements only "
               "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
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

  if (pCreateInfo->pushConstantRangeCount > 0) {
    // Recorded but not honoured: GL has no push constants.  A pipeline whose
    // shader actually reads them is refused at creation, so reaching this point
    // means the ranges are unused and silence is correct.
    KNIZ_DBG("pipeline layout declares %u push constant range(s); they are "
             "ignored because GL has no equivalent",
             pCreateInfo->pushConstantRangeCount);
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
    if (write.pBufferInfo == nullptr) {
      KNIZ_ERR("write %u declares a storage buffer but carries no buffer info", i);
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

}  // namespace kniz
