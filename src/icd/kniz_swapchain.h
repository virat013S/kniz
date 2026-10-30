#pragma once

// Presentation: VK_KHR_surface, VK_KHR_xcb_surface and VK_KHR_swapchain.
//
// Everything before this file produces images.  This file is the first that has
// to put one in front of a person, and that is the one thing GL is worse at than
// Vulkan.
//
// A Vulkan swapchain is a queue of images the presentation engine will show, and
// vkQueuePresentKHR hands one over.  GL has no such object: a window's back
// buffer is owned by the window system, and a texture is never that buffer.  So
// the swapchain here is two things glued together -- a list of GL textures, and
// an EGL window surface -- and presentation is a draw from one into the other.
// The draw is the whole of the seam: no format need match, and no y flip is
// needed anywhere else in the driver (see docs/DESIGN.md §3.13).
//
// The window system is reached only through EGL.  A surface records the native
// window handle and nothing else, so there is no XCB, Xlib or Wayland code in
// the driver and no build dependency on any of them.

#include "kniz_objects.h"

namespace kniz {

// VK_KHR_xcb_surface's revision.  Spelled out because vulkan_xcb.h -- which
// defines it -- is only available when VK_USE_PLATFORM_XCB_KHR is defined, and
// Kniz does not include any platform header.  The number is part of the
// extension's published interface and has not changed since 2016.
inline constexpr uint32_t kXcbSurfaceSpecVersion = 6;

// VK_KHR_wayland_surface's current promoted revision (6) and structure type,
// spelled out here for the same reason the xcb one is: no platform header is
// included anywhere in this driver, the platform guards in <vulkan/vulkan.h>
// would have to be switched on, and a hand-written constant the extension spec
// fixes is smaller than that.
inline constexpr uint32_t kWaylandSurfaceSpecVersion = 6;

// The device extensions this file implements.  vkCreateDevice uses it to accept
// them quietly while still warning about everything else it cannot do.
bool swapchain_extension_supported(const char* name);

// The surface create info the platform extension defines, declared here for the
// same reason: no xcb header is included anywhere in the driver.  The layout is
// VK_KHR_xcb_surface's published ABI.
struct XcbSurfaceCreateInfo {
  VkStructureType sType;
  const void* pNext;
  VkFlags flags;
  void* connection;  // xcb_connection_t*
  uint32_t window;   // xcb_window_t, a 32-bit XID
};

// VK_KHR_wayland_surface's create info, same reason and same published ABI.
// Wayland hands over pointers rather than identifiers, and they belong to the
// application's own connection -- which is what makes a Wayland context a
// matter of the application's wl_display and not our own connection.
struct WaylandSurfaceCreateInfo {
  VkStructureType sType;
  const void* pNext;
  VkFlags flags;
  void* display;   // wl_display*, the application's connection
  void* surface;   // wl_surface*, the application's window shell
};

VKAPI_ATTR VkResult VKAPI_CALL CreateXcbSurfaceKHR(
    VkInstance instance, const XcbSurfaceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface);
VKAPI_ATTR VkResult VKAPI_CALL CreateWaylandSurfaceKHR(
    VkInstance instance, const WaylandSurfaceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface);
VKAPI_ATTR void VKAPI_CALL DestroySurfaceKHR(
    VkInstance instance, VkSurfaceKHR surface,
    const VkAllocationCallbacks* pAllocator);
VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceSupportKHR(
    VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex,
    VkSurfaceKHR surface, VkBool32* pSupported);
VKAPI_ATTR VkBool32 VKAPI_CALL GetPhysicalDeviceXcbPresentationSupportKHR(
    VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex,
    const void* connection, uint32_t visual);
VKAPI_ATTR VkBool32 VKAPI_CALL GetPhysicalDeviceXlibPresentationSupportKHR(
    VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex,
    const void* display, uint32_t screen);
VKAPI_ATTR VkBool32 VKAPI_CALL GetPhysicalDeviceWaylandPresentationSupportKHR(
    VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex,
    const void* display);
VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceCapabilitiesKHR(
    VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
    VkSurfaceCapabilitiesKHR* pSurfaceCapabilities);
VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceFormatsKHR(
    VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
    uint32_t* pSurfaceFormatCount, VkSurfaceFormatKHR* pSurfaceFormats);
VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfacePresentModesKHR(
    VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
    uint32_t* pPresentModeCount, VkPresentModeKHR* pPresentModes);

VKAPI_ATTR VkResult VKAPI_CALL CreateSwapchainKHR(
    VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain);
VKAPI_ATTR void VKAPI_CALL DestroySwapchainKHR(
    VkDevice device, VkSwapchainKHR swapchain,
    const VkAllocationCallbacks* pAllocator);
VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainImagesKHR(
    VkDevice device, VkSwapchainKHR swapchain, uint32_t* pSwapchainImageCount,
    VkImage* pSwapchainImages);
VKAPI_ATTR VkResult VKAPI_CALL AcquireNextImageKHR(
    VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout,
    VkSemaphore semaphore, VkFence fence, uint32_t* pImageIndex);
VKAPI_ATTR VkResult VKAPI_CALL QueuePresentKHR(
    VkQueue queue, const VkPresentInfoKHR* pPresentInfo);

}  // namespace kniz
