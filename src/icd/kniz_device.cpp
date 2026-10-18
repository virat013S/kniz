// Device level Vulkan commands.
//
// At M0 this is a small set: create/destroy, dispatch lookup, and the queue
// handle.  Everything command-buffer related lands in M1.
//
// Note the deliberate split: the commands below live in namespace kniz with
// external linkage because kniz_icd.cpp references them by name when building
// the instance level entry point table.  Only the table itself is internal.

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

// Declared in kniz_device.h, defined below.  This is the only reason those
// definitions cannot be file-local.
const Entry kEntries[] = {
    {"vkCreateDevice", as_entry(CreateDevice)},
    {"vkDestroyDevice", as_entry(DestroyDevice)},
    {"vkGetDeviceQueue", as_entry(GetDeviceQueue)},
    {"vkDeviceWaitIdle", as_entry(DeviceWaitIdle)},
    {"vkQueueWaitIdle", as_entry(QueueWaitIdle)},
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

// M0 performs no GPU work: vkQueueSubmit does not exist yet and the GL context
// is only ever touched on the calling thread.  Both of these therefore have
// nothing to wait for.  They become real once M1 introduces submission and
// fences.
VKAPI_ATTR VkResult VKAPI_CALL DeviceWaitIdle(VkDevice deviceHandle) {
  if (from_handle(deviceHandle) == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL QueueWaitIdle(VkQueue queueHandle) {
  if (from_handle(queueHandle) == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
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
