// M0 regression test: instance -> physical device -> device -> teardown, driven
// through the real Vulkan loader.
//
// Deliberately exercised through the loader rather than by calling the ICD
// directly.  The failure this test exists to prevent -- an ICD whose
// dispatchable objects do not reserve their first word for the loader's
// dispatch pointer -- is invisible when the ICD is driven directly and appears
// only once the loader is in the path.  It shows up as a device handle that
// vkCreateDevice hands back successfully and that every subsequent call
// rejects.
//
// The loader is opened with dlopen instead of being linked, so the test builds
// without a Vulkan loader development package.  Which ICD is used is chosen by
// VK_ICD_FILENAMES, which CTest sets.
//
// Entry points are resolved the way a real application must: globals with
// vkGetInstanceProcAddr(NULL, ...), instance level commands with a live
// instance, and device level commands through vkGetDeviceProcAddr.

#include <vulkan/vulkan.h>

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
  std::printf("  %-58s %s\n", what, ok ? "ok" : "FAILED");
  if (!ok)
    ++g_failures;
}

// Copies one function pointer into another typed slot of the same size.  Using
// memcpy keeps the void* -> function pointer conversion out of the source.
template <typename Fn, typename From>
Fn to_fn(From symbol) {
  static_assert(sizeof(Fn) == sizeof(From), "function pointer size mismatch");
  Fn fn = nullptr;
  std::memcpy(&fn, &symbol, sizeof(fn));
  return fn;
}

void require(const char* name, PFN_vkVoidFunction fn) {
  if (fn == nullptr) {
    std::printf("  missing entry point: %s\n", name);
    ++g_failures;
  }
}

void* load_loader() {
  const char* name = std::getenv("KNIZ_TEST_LOADER");
  if (name == nullptr || *name == '\0')
    name = "libvulkan.so.1";

  void* lib = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr)
    std::fprintf(stderr, "cannot open '%s': %s\n", name, dlerror());
  return lib;
}

PFN_vkVoidFunction (*g_get_instance_proc_addr)(VkInstance, const char*) = nullptr;
PFN_vkGetDeviceProcAddr g_get_device_proc_addr = nullptr;

template <typename Fn>
Fn instance_proc(VkInstance instance, const char* name) {
  return to_fn<Fn>(g_get_instance_proc_addr(instance, name));
}

template <typename Fn>
Fn device_proc(VkDevice device, const char* name) {
  return to_fn<Fn>(g_get_device_proc_addr(device, name));
}

}  // namespace

int main() {
  std::printf("kniz lifecycle test\n");

  if (std::getenv("VK_ICD_FILENAMES") == nullptr) {
    std::fprintf(stderr,
                 "VK_ICD_FILENAMES is not set; the test would run against "
                 "whatever ICD is registered on this system\n");
    return 2;
  }

  void* loader = load_loader();
  if (loader == nullptr)
    return 2;

  g_get_instance_proc_addr =
      to_fn<PFN_vkVoidFunction (*)(VkInstance, const char*)>(
          dlsym(loader, "vkGetInstanceProcAddr"));
  if (g_get_instance_proc_addr == nullptr) {
    std::fprintf(stderr, "loader has no vkGetInstanceProcAddr\n");
    return 2;
  }

  // ---- global commands (valid with a NULL instance) ------------------------
  auto create_instance =
      instance_proc<PFN_vkCreateInstance>(nullptr, "vkCreateInstance");
  require("vkCreateInstance", reinterpret_cast<PFN_vkVoidFunction>(create_instance));
  if (g_failures > 0)
    return 2;

  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "kniz-lifecycle";
  app.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pApplicationInfo = &app;

  VkInstance instance = VK_NULL_HANDLE;
  check(create_instance(&ci, nullptr, &instance) == VK_SUCCESS,
        "vkCreateInstance");
  if (instance == VK_NULL_HANDLE)
    return 1;

  // ---- instance level commands --------------------------------------------
  auto destroy_instance =
      instance_proc<PFN_vkDestroyInstance>(instance, "vkDestroyInstance");
  auto enumerate_physical =
      instance_proc<PFN_vkEnumeratePhysicalDevices>(instance,
                                                    "vkEnumeratePhysicalDevices");
  auto get_properties = instance_proc<PFN_vkGetPhysicalDeviceProperties>(
      instance, "vkGetPhysicalDeviceProperties");
  auto get_queue_families = instance_proc<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
      instance, "vkGetPhysicalDeviceQueueFamilyProperties");
  auto create_device = instance_proc<PFN_vkCreateDevice>(instance, "vkCreateDevice");

  require("vkDestroyInstance", reinterpret_cast<PFN_vkVoidFunction>(destroy_instance));
  require("vkEnumeratePhysicalDevices",
          reinterpret_cast<PFN_vkVoidFunction>(enumerate_physical));
  require("vkGetPhysicalDeviceProperties",
          reinterpret_cast<PFN_vkVoidFunction>(get_properties));
  require("vkGetPhysicalDeviceQueueFamilyProperties",
          reinterpret_cast<PFN_vkVoidFunction>(get_queue_families));
  require("vkCreateDevice", reinterpret_cast<PFN_vkVoidFunction>(create_device));

  g_get_device_proc_addr =
      instance_proc<PFN_vkGetDeviceProcAddr>(instance, "vkGetDeviceProcAddr");
  require("vkGetDeviceProcAddr",
          reinterpret_cast<PFN_vkVoidFunction>(g_get_device_proc_addr));

  if (g_failures > 0) {
    destroy_instance(instance, nullptr);
    return 1;
  }

  uint32_t count = 0;
  VkResult res = enumerate_physical(instance, &count, nullptr);
  check(res == VK_SUCCESS && count == 1,
        "vkEnumeratePhysicalDevices reports one device");
  if (count == 0) {
    destroy_instance(instance, nullptr);
    return 1;
  }

  VkPhysicalDevice physical = VK_NULL_HANDLE;
  res = enumerate_physical(instance, &count, &physical);
  check(res == VK_SUCCESS && physical != VK_NULL_HANDLE, "physical device handle");

  VkPhysicalDeviceProperties props;
  std::memset(&props, 0, sizeof(props));
  get_properties(physical, &props);

  // Provenance check.  Without it the test would still pass against the
  // system's Intel driver if VK_ICD_FILENAMES were ignored.
  check(props.vendorID == 0x4B4E, "physical device is provided by Kniz");
  check(std::strstr(props.deviceName, "Kniz") != nullptr,
        "deviceName mentions Kniz");
  check(props.limits.maxBoundDescriptorSets > 0, "limits were filled in");
  std::printf("     deviceName : %s\n", props.deviceName);
  std::printf("     apiVersion : %u.%u\n", VK_VERSION_MAJOR(props.apiVersion),
              VK_VERSION_MINOR(props.apiVersion));

  uint32_t family_count = 0;
  get_queue_families(physical, &family_count, nullptr);
  check(family_count == 1, "one queue family");

  float priority = 1.0f;
  VkDeviceQueueCreateInfo queue{};
  queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue.queueFamilyIndex = 0;
  queue.queueCount = 1;
  queue.pQueuePriorities = &priority;

  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &queue;

  VkDevice device = VK_NULL_HANDLE;
  check(create_device(physical, &dci, nullptr, &device) == VK_SUCCESS,
        "vkCreateDevice");
  if (device == VK_NULL_HANDLE) {
    destroy_instance(instance, nullptr);
    return 1;
  }

  // ---- device level commands ----------------------------------------------
  auto destroy_device = device_proc<PFN_vkDestroyDevice>(device, "vkDestroyDevice");
  auto get_device_queue = device_proc<PFN_vkGetDeviceQueue>(device, "vkGetDeviceQueue");
  auto device_wait_idle = device_proc<PFN_vkDeviceWaitIdle>(device, "vkDeviceWaitIdle");

  require("vkDestroyDevice", reinterpret_cast<PFN_vkVoidFunction>(destroy_device));
  require("vkGetDeviceQueue", reinterpret_cast<PFN_vkVoidFunction>(get_device_queue));
  require("vkDeviceWaitIdle", reinterpret_cast<PFN_vkVoidFunction>(device_wait_idle));
  if (g_failures > 0) {
    destroy_device(device, nullptr);
    destroy_instance(instance, nullptr);
    return 1;
  }

  // The regression this test exists for: the loader overwrites the first word of
  // the device object with a pointer to its dispatch table, so a device handle
  // validated by reading offset 0 fails here even though vkCreateDevice
  // succeeded.
  VkQueue vk_queue = VK_NULL_HANDLE;
  get_device_queue(device, 0, 0, &vk_queue);
  check(vk_queue != VK_NULL_HANDLE,
        "vkGetDeviceQueue: device handle survived dispatch setup");

  check(device_wait_idle(device) == VK_SUCCESS,
        "vkDeviceWaitIdle: device handle accepted");

  destroy_device(device, nullptr);
  check(true, "vkDestroyDevice");

  destroy_instance(instance, nullptr);
  check(true, "vkDestroyInstance");

  if (g_failures == 0) {
    std::printf("PASSED\n");
    return 0;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return 1;
}
