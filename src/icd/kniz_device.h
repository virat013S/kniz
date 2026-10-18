#pragma once

// Device level entry points.
//
// These live in their own translation unit because it keeps the instance and
// device tables from tangling, and because M2 onwards will add a lot here.

#include <vulkan/vulkan.h>

namespace kniz {

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(
    VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDevice* pDevice);

VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice device,
                                         const VkAllocationCallbacks* pAllocator);

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device,
                                                           const char* pName);

VKAPI_ATTR void VKAPI_CALL GetDeviceQueue(VkDevice device,
                                          uint32_t queueFamilyIndex,
                                          uint32_t queueIndex, VkQueue* pQueue);

VKAPI_ATTR VkResult VKAPI_CALL DeviceWaitIdle(VkDevice device);

VKAPI_ATTR VkResult VKAPI_CALL QueueWaitIdle(VkQueue queue);

}  // namespace kniz
