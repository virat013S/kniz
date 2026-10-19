// Device level Vulkan commands.
//
// The table below is the device half of the entry point surface.  It is one
// flat table, and both vkGetDeviceProcAddr and the instance level resolver read
// it: applications and tooling do ask for device commands through
// vkGetInstanceProcAddr, so the device commands have to be reachable from both
// without being listed twice.  See kniz_icd.cpp for how the instance table says
// so.
//
// The commands themselves live in namespace kniz with external linkage because
// this table references them by name.  Only the table is internal.

#include "kniz_common.h"
#include "kniz_device.h"
#include "kniz_entrypoints.h"
#include "kniz_objects.h"

#include <new>

namespace kniz {
namespace {

struct Entry {
  const char* name;
  PFN_vkVoidFunction fn;
};

template <typename Fn>
PFN_vkVoidFunction as_entry(Fn fn) {
  return reinterpret_cast<PFN_vkVoidFunction>(fn);
}

const Entry kEntries[] = {
    // Device lifetime and queues.
    {"vkCreateDevice", as_entry(CreateDevice)},
    {"vkDestroyDevice", as_entry(DestroyDevice)},
    {"vkGetDeviceQueue", as_entry(GetDeviceQueue)},
    {"vkDeviceWaitIdle", as_entry(DeviceWaitIdle)},
    {"vkQueueWaitIdle", as_entry(QueueWaitIdle)},

    // vkGetDeviceProcAddr itself.  The loader resolves this through
    // vk_icdGetInstanceProcAddr while assembling its dispatch tables, so
    // leaving it out of both tables makes the loader abandon vkCreateInstance
    // with VK_ERROR_INITIALIZATION_FAILED and nothing to explain why.
    {"vkGetDeviceProcAddr", as_entry(GetDeviceProcAddr)},

    // Buffers and memory.
    {"vkCreateBuffer", as_entry(CreateBuffer)},
    {"vkDestroyBuffer", as_entry(DestroyBuffer)},
    {"vkGetBufferMemoryRequirements", as_entry(GetBufferMemoryRequirements)},
    {"vkAllocateMemory", as_entry(AllocateMemory)},
    {"vkFreeMemory", as_entry(FreeMemory)},
    {"vkBindBufferMemory", as_entry(BindBufferMemory)},
    {"vkMapMemory", as_entry(MapMemory)},
    {"vkUnmapMemory", as_entry(UnmapMemory)},
    {"vkFlushMappedMemoryRanges", as_entry(FlushMappedMemoryRanges)},
    {"vkInvalidateMappedMemoryRanges", as_entry(InvalidateMappedMemoryRanges)},

    // Descriptors.
    {"vkCreateDescriptorSetLayout", as_entry(CreateDescriptorSetLayout)},
    {"vkDestroyDescriptorSetLayout", as_entry(DestroyDescriptorSetLayout)},
    {"vkCreatePipelineLayout", as_entry(CreatePipelineLayout)},
    {"vkDestroyPipelineLayout", as_entry(DestroyPipelineLayout)},
    {"vkCreateDescriptorPool", as_entry(CreateDescriptorPool)},
    {"vkDestroyDescriptorPool", as_entry(DestroyDescriptorPool)},
    {"vkResetDescriptorPool", as_entry(ResetDescriptorPool)},
    {"vkAllocateDescriptorSets", as_entry(AllocateDescriptorSets)},
    {"vkFreeDescriptorSets", as_entry(FreeDescriptorSets)},
    {"vkUpdateDescriptorSets", as_entry(UpdateDescriptorSets)},

    // Shader modules and pipelines.
    {"vkCreateShaderModule", as_entry(CreateShaderModule)},
    {"vkDestroyShaderModule", as_entry(DestroyShaderModule)},
    {"vkCreateComputePipelines", as_entry(CreateComputePipelines)},
    {"vkCreateGraphicsPipelines", as_entry(CreateGraphicsPipelines)},
    {"vkDestroyPipeline", as_entry(DestroyPipeline)},
    {"vkCreatePipelineCache", as_entry(CreatePipelineCache)},
    {"vkDestroyPipelineCache", as_entry(DestroyPipelineCache)},
    {"vkGetPipelineCacheData", as_entry(GetPipelineCacheData)},
    {"vkMergePipelineCaches", as_entry(MergePipelineCaches)},

    // Command buffers.
    {"vkCreateCommandPool", as_entry(CreateCommandPool)},
    {"vkDestroyCommandPool", as_entry(DestroyCommandPool)},
    {"vkResetCommandPool", as_entry(ResetCommandPool)},
    {"vkAllocateCommandBuffers", as_entry(AllocateCommandBuffers)},
    {"vkFreeCommandBuffers", as_entry(FreeCommandBuffers)},
    {"vkBeginCommandBuffer", as_entry(BeginCommandBuffer)},
    {"vkEndCommandBuffer", as_entry(EndCommandBuffer)},
    {"vkResetCommandBuffer", as_entry(ResetCommandBuffer)},
    {"vkCmdBindPipeline", as_entry(CmdBindPipeline)},
    {"vkCmdBindDescriptorSets", as_entry(CmdBindDescriptorSets)},
    {"vkCmdPushConstants", as_entry(CmdPushConstants)},
    {"vkCmdPipelineBarrier", as_entry(CmdPipelineBarrier)},
    {"vkCmdDispatch", as_entry(CmdDispatch)},
    {"vkCmdFillBuffer", as_entry(CmdFillBuffer)},
    {"vkCmdUpdateBuffer", as_entry(CmdUpdateBuffer)},
    {"vkCmdCopyBuffer", as_entry(CmdCopyBuffer)},

    // Images.
    {"vkCreateImage", as_entry(CreateImage)},
    {"vkDestroyImage", as_entry(DestroyImage)},
    {"vkGetImageMemoryRequirements", as_entry(GetImageMemoryRequirements)},
    {"vkBindImageMemory", as_entry(BindImageMemory)},
    {"vkGetImageSubresourceLayout", as_entry(GetImageSubresourceLayout)},
    {"vkCreateImageView", as_entry(CreateImageView)},
    {"vkDestroyImageView", as_entry(DestroyImageView)},
    {"vkCmdClearColorImage", as_entry(CmdClearColorImage)},
    {"vkCmdCopyBufferToImage", as_entry(CmdCopyBufferToImage)},
    {"vkCmdCopyImageToBuffer", as_entry(CmdCopyImageToBuffer)},

    // Render passes, framebuffers and drawing.
    {"vkCreateRenderPass", as_entry(CreateRenderPass)},
    {"vkDestroyRenderPass", as_entry(DestroyRenderPass)},
    {"vkCreateFramebuffer", as_entry(CreateFramebuffer)},
    {"vkDestroyFramebuffer", as_entry(DestroyFramebuffer)},
    {"vkCmdBeginRenderPass", as_entry(CmdBeginRenderPass)},
    {"vkCmdEndRenderPass", as_entry(CmdEndRenderPass)},
    {"vkCmdNextSubpass", as_entry(CmdNextSubpass)},
    {"vkCmdBindVertexBuffers", as_entry(CmdBindVertexBuffers)},
    {"vkCmdBindIndexBuffer", as_entry(CmdBindIndexBuffer)},
    {"vkCmdSetViewport", as_entry(CmdSetViewport)},
    {"vkCmdSetScissor", as_entry(CmdSetScissor)},
    {"vkCmdDraw", as_entry(CmdDraw)},
    {"vkCmdDrawIndexed", as_entry(CmdDrawIndexed)},
    {"vkCmdClearAttachments", as_entry(CmdClearAttachments)},

    // Submission and synchronisation.
    {"vkQueueSubmit", as_entry(QueueSubmit)},
    {"vkCreateFence", as_entry(CreateFence)},
    {"vkDestroyFence", as_entry(DestroyFence)},
    {"vkResetFences", as_entry(ResetFences)},
    {"vkGetFenceStatus", as_entry(GetFenceStatus)},
    {"vkWaitForFences", as_entry(WaitForFences)},
};

}  // namespace

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(
    VkPhysicalDevice physicalHandle, const VkDeviceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDevice* pDevice) {
  if (pDevice == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pDevice = VK_NULL_HANDLE;

  PhysicalDevice* physical = from_handle(physicalHandle);
  if (physical == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (pCreateInfo == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  if (pAllocator != nullptr)
    KNIZ_WARN("host allocation callbacks are not supported and will be ignored");

  // One queue family, one queue, so anything else is a request we cannot
  // satisfy.  Rejecting it here is far kinder than handing back a device whose
  // queues silently never execute.
  for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; ++i) {
    const VkDeviceQueueCreateInfo& q = pCreateInfo->pQueueCreateInfos[i];
    if (q.queueFamilyIndex != 0) {
      KNIZ_ERR("only queue family 0 exists; family %u requested",
               q.queueFamilyIndex);
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (q.queueCount > 1) {
      KNIZ_ERR("only one queue per family is available; %u requested",
               q.queueCount);
      return VK_ERROR_INITIALIZATION_FAILED;
    }
  }

  for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; ++i) {
    KNIZ_WARN("device extension '%s' requested but not implemented",
              pCreateInfo->ppEnabledExtensionNames[i]);
  }

  if (pCreateInfo->pEnabledFeatures != nullptr) {
    const auto* raw =
        reinterpret_cast<const VkBool32*>(pCreateInfo->pEnabledFeatures);
    constexpr size_t kCount =
        sizeof(VkPhysicalDeviceFeatures) / sizeof(VkBool32);
    size_t enabled = 0;
    for (size_t i = 0; i < kCount; ++i) {
      if (raw[i] == VK_TRUE)
        ++enabled;
    }
    if (enabled > 0) {
      // We advertise no optional features, so an application asking for one is
      // either confused or being generous.  Proceeding is safe: a correct
      // implementation checks vkGetPhysicalDeviceFeatures first.
      KNIZ_DBG("application requested %zu optional feature(s); none are "
               "advertised by this driver",
               enabled);
    }
  }

  Device* device = nullptr;
  try {
    device = new Device();
  } catch (const std::bad_alloc&) {
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  }

  // Every dispatchable object must have its reserved first word initialised
  // before the loader sees it.  See kniz_objects.h.
  set_loader_magic_value(device);
  set_loader_magic_value(&device->queue);

  device->instance = physical->instance;
  device->physical = physical;
  device->queue.device = device;
  device->queue.family_index = 0;
  device->queue.queue_index = 0;

  *pDevice = to_handle(device);
  KNIZ_INFO("device created (queue family 0)");
  KNIZ_TRACE("vkCreateDevice returned handle %p", static_cast<void*>(*pDevice));
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyDevice(
    VkDevice deviceHandle, const VkAllocationCallbacks* pAllocator) {
  (void)pAllocator;
  Device* device = from_handle(deviceHandle);
  if (device == nullptr)
    return;
  delete device;
  KNIZ_DBG("device destroyed");
}

// Reports a handle that failed validation, including the value found at the
// start of the object, so a protocol mismatch is distinguishable from a memory
// error.
void log_bad_device(const char* who, VkDevice handle) {
  uint64_t first_word = 0;
  if (handle != nullptr)
    std::memcpy(&first_word, handle, sizeof(first_word));
  KNIZ_ERR("%s: %p is not a Kniz device (first word 0x%016llx, expected "
           "0x%016llx)",
           who, static_cast<void*>(handle),
           static_cast<unsigned long long>(first_word),
           static_cast<unsigned long long>(kDeviceMagic));
}

VKAPI_ATTR void VKAPI_CALL GetDeviceQueue(VkDevice deviceHandle,
                                          uint32_t queueFamilyIndex,
                                          uint32_t queueIndex,
                                          VkQueue* pQueue) {
  if (pQueue == nullptr)
    return;
  *pQueue = VK_NULL_HANDLE;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr) {
    log_bad_device("vkGetDeviceQueue", deviceHandle);
    return;
  }

  if (queueFamilyIndex != 0 || queueIndex != 0) {
    KNIZ_ERR("vkGetDeviceQueue(%u, %u) is out of range; only (0, 0) exists",
             queueFamilyIndex, queueIndex);
    return;
  }

  // The queue is embedded in the device, so the handle is stable and repeated
  // calls return the same value, which is what the specification requires.
  *pQueue = to_handle(&device->queue);
}

// Work in Kniz is finished by the time vkQueueSubmit returns, so an idle wait
// has nothing of its own to wait for.  What it can still do is push any GL work
// the driver has buffered, which is worth doing at a point the application has
// explicitly asked to synchronise with.
static void drain_gl() {
  if (gl::f.Finish == nullptr)
    return;
  if (!gl::make_current())
    return;
  gl::f.Finish();
}

VKAPI_ATTR VkResult VKAPI_CALL DeviceWaitIdle(VkDevice deviceHandle) {
  if (from_handle(deviceHandle) == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  drain_gl();
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL QueueWaitIdle(VkQueue queueHandle) {
  Queue* queue = from_handle(queueHandle);
  if (queue == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  (void)queue;
  drain_gl();
  return VK_SUCCESS;
}

PFN_vkVoidFunction GetDeviceProcAddr(VkDevice deviceHandle, const char* pName) {
  // The handle is deliberately not validated.
  //
  // The loader calls this while assembling its own device dispatch table, and
  // the value it passes at that point is the application-visible handle -- whose
  // first word is a pointer to the loader's dispatch table, not our object --
  // rather than the VkDevice we returned from vkCreateDevice.  Validating it
  // rejects every lookup and leaves the loader with a device dispatch table full
  // of nulls, which surfaces much later as a jump through address zero.
  //
  // Skipping the check is also correct rather than merely convenient: the spec's
  // precondition for this entry point is that the device is valid, which the
  // loader guarantees by construction.  Every function we hand back validates
  // its own handle at the point it is actually called.
  (void)deviceHandle;

  if (pName == nullptr)
    return nullptr;

  KNIZ_TRACE("vkGetDeviceProcAddr(_, %s)", pName);

  for (const Entry& entry : kEntries) {
    if (std::strcmp(entry.name, pName) == 0)
      return entry.fn;
  }

  // Instance level commands remain valid off a device handle in practice, and
  // applications do ask for them this way.  Fall back rather than returning
  // nullptr.
  return resolve_instance_proc(pName);
}

PFN_vkVoidFunction resolve_device_proc(const char* name) {
  if (name == nullptr)
    return nullptr;
  for (const Entry& entry : kEntries) {
    if (std::strcmp(entry.name, name) == 0)
      return entry.fn;
  }
  return nullptr;
}

}  // namespace kniz
