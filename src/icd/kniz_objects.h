#pragma once

// The Vulkan object model.
//
// Vulkan handles are opaque.  Kniz hands the application pointers to the
// structures below, wrapped in the real Vulkan handle types, and recovers them
// on the way back in.  Every object starts with a magic value so a stale handle
// -- or one belonging to another driver -- is rejected rather than
// dereferenced.
//
// On 64-bit platforms both dispatchable and non-dispatchable Vulkan handles are
// pointer sized, so pointers round-trip losslessly.  Kniz targets 64-bit only
// for now; a 32-bit build would need non-dispatchable handles to come from a
// table keyed by index instead.
//
// Deliberately absent: a dispatch table inside each object.  An ICD does not
// need one.  The loader assembles its own dispatch from the function pointers
// we hand back in vk_icdGetInstanceProcAddr, so there is nothing for us to
// store per object.

#include "kniz_common.h"
#include "kniz_gl.h"

#include <vulkan/vk_icd.h>
#include <vulkan/vulkan.h>

#include <cstdint>

namespace kniz {

inline constexpr uint64_t kInstanceMagic = 0x4B4E495A494E5354ull;  // KNIZINST
inline constexpr uint64_t kPhysicalMagic = 0x4B4E495A50485944ull;  // KNIZPHYD
inline constexpr uint64_t kDeviceMagic = 0x4B4E495A44455649ull;    // KNIZDEVI
inline constexpr uint64_t kQueueMagic = 0x4B4E495A51554555ull;     // KNIZQUEU

struct Instance;
struct PhysicalDevice;
struct Device;
struct Queue;

// ---------------------------------------------------------------------------
// Objects
//
// THE FIRST FIELD OF EVERY DISPATCHABLE OBJECT IS RESERVED.
//
// Vulkan puts a dispatch table pointer at offset 0 of every dispatchable
// handle.  For an ICD, the loader owns those bytes: after vkCreateDevice it
// executes
//
//     loader_init_dispatch(*pDevice, &dev->loader_dispatch);   // writes offset 0
//
// and it assumes the same layout when reading a handle back.  Meanwhile
// vulkan/vk_icd.h requires the ICD to initialise that first word with
// ICD_LOADER_MAGIC (via set_loader_magic_value), which the loader checks in
// debug builds.
//
// So the opening field must hold a VK_LOADER_DATA initialised by
// set_loader_magic_value(), must never be read by us, and nothing of ours may
// live there.  Our own bookkeeping -- including the magic used to validate a
// handle -- starts immediately after it.  Getting this wrong produces a driver
// whose handles appear valid until the loader overwrites them, at which point
// every device entry point silently rejects its own device.
// ---------------------------------------------------------------------------

struct Instance {
  VK_LOADER_DATA loader_data;  // reserved by the loader; must be first

  uint64_t magic = kInstanceMagic;

  // What the application asked for, and what we will actually honour.  Kniz
  // implements Vulkan 1.0 semantics until the corresponding work lands, and
  // reports that honestly rather than advertising a version whose entry points
  // would return VK_ERROR_FEATURE_NOT_PRESENT.
  uint32_t api_version = VK_API_VERSION_1_0;
  uint32_t effective_api_version = VK_API_VERSION_1_0;

  bool gl_ready = false;
  gl::ContextInfo gl{};

  // Exactly one physical device per instance.  It lives inline so that
  // VkPhysicalDevice stays stable across vkEnumeratePhysicalDevices calls, as
  // the specification requires.
  PhysicalDevice* physical = nullptr;
};

struct PhysicalDevice {
  VK_LOADER_DATA loader_data;  // reserved by the loader; must be first

  uint64_t magic = kPhysicalMagic;
  Instance* instance = nullptr;

  // Index within the instance, i.e. the value returned by
  // vkEnumeratePhysicalDevices.  vkCreateDevice must be handed the same index
  // back, so it is stored rather than recomputed.
  uint32_t index = 0;

  // Device identity, currently derived from the GL adapter we picked up.
  uint32_t vendor_id = 0;
  uint32_t device_id = 0;
};

struct Queue {
  // VkQueue is a non-dispatchable handle, but on 64-bit the loader still stores
  // a dispatch pointer at offset 0 of it (see loader_set_dispatch(*pQueue, ...)
  // in the loader's vkGetDeviceQueue trampoline), so the same reservation
  // applies.
  VK_LOADER_DATA loader_data;  // reserved by the loader; must be first

  uint64_t magic = kQueueMagic;
  Device* device = nullptr;
  uint32_t family_index = 0;
  uint32_t queue_index = 0;
};

struct Device {
  VK_LOADER_DATA loader_data;  // reserved by the loader; must be first

  uint64_t magic = kDeviceMagic;
  Instance* instance = nullptr;
  PhysicalDevice* physical = nullptr;

  // One queue family, one queue.
  Queue queue{};
};

// ---------------------------------------------------------------------------
// Handle conversions
// ---------------------------------------------------------------------------

namespace detail {

template <typename T>
inline T* to_object(uint64_t magic, void* handle) {
  if (handle == nullptr)
    return nullptr;
  auto* object = static_cast<T*>(handle);
  if (object->magic != magic)
    return nullptr;
  return object;
}

}  // namespace detail

// Each accessor returns nullptr for a null handle, an uninitialised handle, or
// a handle that does not belong to Kniz.
inline Instance* from_handle(VkInstance handle) {
  return detail::to_object<Instance>(kInstanceMagic, handle);
}

inline PhysicalDevice* from_handle(VkPhysicalDevice handle) {
  return detail::to_object<PhysicalDevice>(kPhysicalMagic, handle);
}

inline Device* from_handle(VkDevice handle) {
  return detail::to_object<Device>(kDeviceMagic, handle);
}

inline Queue* from_handle(VkQueue handle) {
  return detail::to_object<Queue>(kQueueMagic, handle);
}

// The inverse conversions are equally direct.
inline VkInstance to_handle(Instance* object) {
  return reinterpret_cast<VkInstance>(object);
}

inline VkPhysicalDevice to_handle(PhysicalDevice* object) {
  return reinterpret_cast<VkPhysicalDevice>(object);
}

inline VkDevice to_handle(Device* object) {
  return reinterpret_cast<VkDevice>(object);
}

inline VkQueue to_handle(Queue* object) {
  return reinterpret_cast<VkQueue>(object);
}

}  // namespace kniz
