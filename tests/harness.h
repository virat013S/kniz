#pragma once

// Shared scaffolding for the test programs.
//
// Every test drives Kniz through the real Vulkan loader rather than calling the
// ICD directly -- see any test for why -- which means each one would otherwise
// repeat the same forty lines of loader opening, entry point resolution and
// check bookkeeping.
//
// It lives in one place because the duplicate had already drifted: one test
// honoured KNIZ_TEST_LOADER and the other had quietly stopped, so the same
// environment variable meant two different things depending on which test ran.
// A third test is about to be added, and it would have inherited whichever
// copy it was written next to.

#include <vulkan/vulkan.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace kniz_test {

// Number of failed checks.  Each test prints it and exits non-zero when it is
// not zero, so a failed check can never be merely printed and ignored.
inline int& failures() {
  static int count = 0;
  return count;
}

inline void check(bool ok, const char* what) {
  std::printf("  %-58s %s\n", what, ok ? "ok" : "FAILED");
  if (!ok)
    ++failures();
}

inline void require(const char* name, PFN_vkVoidFunction fn) {
  if (fn == nullptr) {
    std::printf("  missing entry point: %s\n", name);
    ++failures();
  }
}

// Copies one function pointer into another typed slot of the same size.  Going
// through memcpy keeps the void* -> function pointer conversion out of the
// source, where it would be a conditionally supported cast.
template <typename Fn, typename From>
Fn to_fn(From symbol) {
  static_assert(sizeof(Fn) == sizeof(From), "function pointer size mismatch");
  Fn fn = nullptr;
  std::memcpy(&fn, &symbol, sizeof(fn));
  return fn;
}

// Opens the loader by dlopen rather than by linking it, so the tests build
// without a Vulkan loader development package.  KNIZ_TEST_LOADER overrides the
// name: a distribution that installs it somewhere unhelpful can still run the
// suite, and a missing loader can be made an explicit failure instead of a
// silent skip.
inline void* open_loader() {
  const char* name = std::getenv("KNIZ_TEST_LOADER");
  if (name == nullptr || *name == '\0')
    name = "libvulkan.so.1";

  void* loader = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (loader == nullptr)
    std::fprintf(stderr, "cannot open '%s': %s\n", name, dlerror());
  return loader;
}

// Entry point resolution, done the way an application must do it: global
// commands with a null instance, instance level commands with a live instance,
// device level commands through vkGetDeviceProcAddr.
inline PFN_vkVoidFunction (*get_instance_proc_addr)(VkInstance, const char*) = nullptr;
inline PFN_vkGetDeviceProcAddr get_device_proc_addr = nullptr;

inline bool resolve_get_instance_proc_addr(void* loader) {
  get_instance_proc_addr = to_fn<PFN_vkVoidFunction (*)(VkInstance, const char*)>(
      dlsym(loader, "vkGetInstanceProcAddr"));
  if (get_instance_proc_addr == nullptr) {
    std::fprintf(stderr, "loader has no vkGetInstanceProcAddr\n");
    return false;
  }
  return true;
}

template <typename Fn>
Fn instance_proc(VkInstance instance, const char* name) {
  return to_fn<Fn>(get_instance_proc_addr(instance, name));
}

template <typename Fn>
Fn device_proc(VkDevice device, const char* name) {
  return to_fn<Fn>(get_device_proc_addr(device, name));
}

// Every test must be pointed at the ICD under test.  Without that the loader
// selects the machine's own driver and the suite passes while proving nothing
// about Kniz.
//
// Both loader variables are consulted, and that is not belt and braces: when
// VK_DRIVER_FILES is set the loader ignores VK_ICD_FILENAMES completely.  A
// machine that pins one vendor's driver with VK_DRIVER_FILES therefore redirects
// every test to the system driver no matter what VK_ICD_FILENAMES says, and the
// failure looks like a driver bug rather than a stale environment.  The
// authoritative check is the vendor ID in bring_up; this one exists to explain
// the situation when that check fails.
inline bool icd_selected() {
  // Deliberately a substring test rather than a comparison: the manifest can live
  // anywhere, and the authoritative check is the vendor ID that bring_up makes.
  const auto names_kniz = [](const char* value) -> bool {
    return value != nullptr && std::strstr(value, "kniz") != nullptr;
  };
  const char* driver_files = std::getenv("VK_DRIVER_FILES");
  const char* icd_filenames = std::getenv("VK_ICD_FILENAMES");
  if (names_kniz(driver_files) || names_kniz(icd_filenames))
    return true;

  std::fprintf(stderr, "this test is not pointed at Kniz\n");
  if (driver_files != nullptr)
    std::fprintf(stderr, "     VK_DRIVER_FILES = %s\n", driver_files);
  if (icd_filenames != nullptr)
    std::fprintf(stderr, "     VK_ICD_FILENAMES = %s\n", icd_filenames);
  std::fprintf(stderr,
               "     set both to the manifest in build/src so that whichever the "
               "loader prefers, it finds Kniz\n");
  return false;
}

// Kniz's PCI vendor ID.  Tests identify the driver by this rather than by
// position, because the loader may expose other ICDs and a test that ran against
// the system's own driver would prove nothing.
constexpr uint32_t kKnizVendorId = 0x4B4E;

// An instance, the one physical device Kniz exposes, and a device with its single
// queue.  Every test needs this much, and repeating it per test is how the copies
// drift.
struct Runtime {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;

  VkPhysicalDeviceMemoryProperties memory{};

  // The memory type with VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, found by asking
  // the driver rather than by assuming an index.
  uint32_t host_memory_type = 0xFFFFFFFFu;

  PFN_vkDestroyInstance destroy_instance = nullptr;
  PFN_vkDestroyDevice destroy_device = nullptr;
  PFN_vkDeviceWaitIdle device_wait_idle = nullptr;
};

// Extensions to enable at creation time.  Empty by default, which is what every
// test before the present one wants: none of them needs an extension to do its
// work, and asking for one that the driver does not advertise would make the
// loader refuse the instance.
struct Extensions {
  std::vector<const char*> instance;
  std::vector<const char*> device;
};

// Creates the instance, selects the Kniz device, and creates the device and
// queue.  Returns false after reporting why; on failure the caller should call
// tear_down, which copes with a partly initialised Runtime.
bool bring_up(Runtime* runtime, const char* app_name,
              const Extensions& extensions = Extensions{});

// Releases everything bring_up created that is still alive.  Waits for the device
// first, because teardown in these tests happens right after a submission.
void tear_down(Runtime* runtime);

inline bool bring_up(Runtime* runtime, const char* app_name,
                     const Extensions& extensions) {
  auto create_instance =
      instance_proc<PFN_vkCreateInstance>(nullptr, "vkCreateInstance");
  if (create_instance == nullptr) {
    std::fprintf(stderr, "no vkCreateInstance\n");
    return false;
  }

  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = app_name;
  app.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo instance_info{};
  instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_info.pApplicationInfo = &app;
  instance_info.enabledExtensionCount =
      static_cast<uint32_t>(extensions.instance.size());
  instance_info.ppEnabledExtensionNames =
      extensions.instance.empty() ? nullptr : extensions.instance.data();

  if (create_instance(&instance_info, nullptr, &runtime->instance) != VK_SUCCESS) {
    std::fprintf(stderr, "vkCreateInstance failed\n");
    return false;
  }

  runtime->destroy_instance =
      instance_proc<PFN_vkDestroyInstance>(runtime->instance, "vkDestroyInstance");
  auto enumerate_physical = instance_proc<PFN_vkEnumeratePhysicalDevices>(
      runtime->instance, "vkEnumeratePhysicalDevices");
  auto get_properties = instance_proc<PFN_vkGetPhysicalDeviceProperties>(
      runtime->instance, "vkGetPhysicalDeviceProperties");
  auto get_memory_properties = instance_proc<PFN_vkGetPhysicalDeviceMemoryProperties>(
      runtime->instance, "vkGetPhysicalDeviceMemoryProperties");
  auto create_device =
      instance_proc<PFN_vkCreateDevice>(runtime->instance, "vkCreateDevice");

  // vkGetDeviceProcAddr is an instance level command, and has to be resolved
  // before any device exists.
  get_device_proc_addr =
      instance_proc<PFN_vkGetDeviceProcAddr>(runtime->instance, "vkGetDeviceProcAddr");
  if (get_device_proc_addr == nullptr) {
    std::fprintf(stderr, "no vkGetDeviceProcAddr\n");
    return false;
  }

  uint32_t device_count = 0;
  enumerate_physical(runtime->instance, &device_count, nullptr);
  if (device_count == 0) {
    std::fprintf(stderr, "the loader reports no physical devices\n");
    return false;
  }
  std::vector<VkPhysicalDevice> devices(device_count);
  enumerate_physical(runtime->instance, &device_count, devices.data());

  for (VkPhysicalDevice candidate : devices) {
    VkPhysicalDeviceProperties properties{};
    get_properties(candidate, &properties);
    if (properties.vendorID == kKnizVendorId) {
      runtime->physical = candidate;
      std::printf("     device     : %s\n", properties.deviceName);
      break;
    }
  }
  check(runtime->physical != VK_NULL_HANDLE, "a Kniz physical device is present");
  if (runtime->physical == VK_NULL_HANDLE)
    return false;

  get_memory_properties(runtime->physical, &runtime->memory);
  for (uint32_t i = 0; i < runtime->memory.memoryTypeCount; ++i) {
    if ((runtime->memory.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
      runtime->host_memory_type = i;
      break;
    }
  }
  check(runtime->host_memory_type != 0xFFFFFFFFu,
        "the driver reports a host visible memory type");

  float priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = 0;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;

  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.enabledExtensionCount =
      static_cast<uint32_t>(extensions.device.size());
  device_info.ppEnabledExtensionNames =
      extensions.device.empty() ? nullptr : extensions.device.data();

  if (create_device(runtime->physical, &device_info, nullptr, &runtime->device) !=
      VK_SUCCESS) {
    std::fprintf(stderr, "vkCreateDevice failed\n");
    return false;
  }

  runtime->destroy_device = device_proc<PFN_vkDestroyDevice>(runtime->device, "vkDestroyDevice");
  runtime->device_wait_idle =
      device_proc<PFN_vkDeviceWaitIdle>(runtime->device, "vkDeviceWaitIdle");
  auto get_queue = device_proc<PFN_vkGetDeviceQueue>(runtime->device, "vkGetDeviceQueue");
  get_queue(runtime->device, 0, 0, &runtime->queue);

  return true;
}

inline void tear_down(Runtime* runtime) {
  if (runtime->device != VK_NULL_HANDLE) {
    if (runtime->device_wait_idle != nullptr)
      runtime->device_wait_idle(runtime->device);
    if (runtime->destroy_device != nullptr)
      runtime->destroy_device(runtime->device, nullptr);
    runtime->device = VK_NULL_HANDLE;
  }
  if (runtime->instance != VK_NULL_HANDLE) {
    if (runtime->destroy_instance != nullptr)
      runtime->destroy_instance(runtime->instance, nullptr);
    runtime->instance = VK_NULL_HANDLE;
  }
}

}  // namespace kniz_test
