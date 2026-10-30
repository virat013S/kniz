// kniz-info -- reports what the Kniz ICD looks like from an application's point
// of view.
//
// This is the tool to run when you want to know whether Kniz is working.  It
// opens the real Vulkan loader, creates an instance and a device through it, and
// prints what came back, together with an honest statement of how much of Vulkan
// is actually implemented.
//
// It is deliberately self-contained: the loader is opened with dlopen, so
// building and running this needs the Vulkan headers and nothing else -- no
// loader development package, no vulkan-tools.
//
// The capability table at the end is hard-coded and must be updated as
// milestones land.  It is the point of the tool, so keep it truthful.

#include <vulkan/vulkan.h>

#include <dlfcn.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

#if defined(__GNUC__)
#define KNIZ_PRINTF(fmt_index, first_arg) \
  __attribute__((format(printf, fmt_index, first_arg)))
#else
#define KNIZ_PRINTF(fmt_index, first_arg)
#endif

// Wide enough for every label below, so values always line up.
constexpr int kLabelWidth = 22;

int g_problems = 0;

// Copies one function pointer into another typed slot of the same size.  Using
// memcpy keeps the void* -> function pointer conversion out of the source.
template <typename Fn, typename From>
Fn to_fn(From symbol) {
  static_assert(sizeof(Fn) == sizeof(From), "function pointer size mismatch");
  Fn fn = nullptr;
  std::memcpy(&fn, &symbol, sizeof(fn));
  return fn;
}

PFN_vkVoidFunction (*g_gipa)(VkInstance, const char*) = nullptr;
PFN_vkGetDeviceProcAddr g_gdpa = nullptr;

template <typename Fn>
Fn instance_proc(VkInstance instance, const char* name) {
  return to_fn<Fn>(g_gipa(instance, name));
}

template <typename Fn>
Fn device_proc(VkDevice device, const char* name) {
  return to_fn<Fn>(g_gdpa(device, name));
}

void field(const char* label, const char* fmt, ...) KNIZ_PRINTF(2, 3);

void field(const char* label, const char* fmt, ...) {
  std::printf("  %-*s ", kLabelWidth, label);
  va_list args;
  va_start(args, fmt);
  std::vprintf(fmt, args);
  va_end(args);
  std::printf("\n");
}

void note(const char* fmt, ...) KNIZ_PRINTF(1, 2);

void note(const char* fmt, ...) {
  std::printf("  %-*s ", kLabelWidth, "");
  va_list args;
  va_start(args, fmt);
  std::vprintf(fmt, args);
  va_end(args);
  std::printf("\n");
}

void section(const char* title) { std::printf("\n%s\n", title); }

void capability(const char* what, bool done, const char* milestone) {
  std::printf("  %-40s %-4s %s\n", what, done ? "yes" : "no", milestone);
}

// Append a flag name to a human readable list, separating with spaces.
void add_flag(char* buffer, size_t size, const char* flag) {
  size_t used = std::strlen(buffer);
  if (used != 0 && used + 1 < size) {
    buffer[used] = ' ';
    buffer[used + 1] = '\0';
  }
  used = std::strlen(buffer);
  if (used + 1 < size)
    std::strncpy(buffer + used, flag, size - used - 1);
  buffer[size - 1] = '\0';
}

const char* device_type_name(VkPhysicalDeviceType type) {
  switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return "integrated GPU";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return "discrete GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return "virtual GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return "CPU";
    case VK_PHYSICAL_DEVICE_TYPE_OTHER:
    default:
      return "other (Kniz does not claim to be real hardware)";
  }
}

void print_queue_families(const VkQueueFamilyProperties* families,
                          uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    const VkQueueFamilyProperties& f = families[i];
    char flags[128] = "";
    if (f.queueFlags & VK_QUEUE_GRAPHICS_BIT)
      add_flag(flags, sizeof(flags), "graphics");
    if (f.queueFlags & VK_QUEUE_COMPUTE_BIT)
      add_flag(flags, sizeof(flags), "compute");
    if (f.queueFlags & VK_QUEUE_TRANSFER_BIT)
      add_flag(flags, sizeof(flags), "transfer");
    if (f.queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)
      add_flag(flags, sizeof(flags), "sparse");

    std::printf("  %-*s %u: %u queue(s), %s\n", kLabelWidth,
                i == 0 ? "families" : "", i, f.queueCount,
                flags[0] != '\0' ? flags : "(none)");
  }
}

void print_memory(const VkPhysicalDeviceMemoryProperties& m) {
  for (uint32_t i = 0; i < m.memoryHeapCount; ++i) {
    std::printf("  %-*s %u: %.0f MiB%s\n", kLabelWidth,
                i == 0 ? "heaps" : "", i,
                static_cast<double>(m.memoryHeaps[i].size) / (1024.0 * 1024.0),
                (m.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    ? ", device local"
                    : "");
  }
  for (uint32_t i = 0; i < m.memoryTypeCount; ++i) {
    const VkMemoryType& t = m.memoryTypes[i];
    char flags[192] = "";
    if (t.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
      add_flag(flags, sizeof(flags), "device-local");
    if (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
      add_flag(flags, sizeof(flags), "host-visible");
    if (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
      add_flag(flags, sizeof(flags), "host-coherent");
    if (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
      add_flag(flags, sizeof(flags), "host-cached");

    std::printf("  %-*s %u: heap %u, %s\n", kLabelWidth, i == 0 ? "types" : "",
                i, t.heapIndex, flags[0] != '\0' ? flags : "(no flags)");
  }
}

void print_version(const char* label, uint32_t version) {
  std::printf("  %-*s %u.%u.%u\n", kLabelWidth, label, VK_VERSION_MAJOR(version),
              VK_VERSION_MINOR(version), VK_VERSION_PATCH(version));
}

}  // namespace

int main(int argc, char** argv) {
  // Kniz logs through the standard error stream.  Line buffering this stream
  // keeps the driver's messages in the order they happened relative to the
  // report, instead of letting them jump ahead when the output is piped.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  const char* loader_name = "libvulkan.so.1";
  if (argc > 1 && argv[1][0] != '\0')
    loader_name = argv[1];

  std::printf("Kniz -- a Vulkan ICD implemented on top of OpenGL\n");
  std::printf("==================================================\n");

  // Let the driver's own log lines through, so the report includes the OpenGL
  // context it brought up.  Do not override an explicit choice.
  setenv("KNIZ_LOG", "2", 0);

  // Which loader variable is actually in force.
  //
  // The loader prefers VK_DRIVER_FILES when it is set and then ignores
  // VK_ICD_FILENAMES completely, so a report that named the other one would send
  // the reader to fix a variable nothing is reading.
  const char* icd_variable = "VK_DRIVER_FILES";
  const char* icd = std::getenv(icd_variable);
  if (icd == nullptr) {
    icd_variable = "VK_ICD_FILENAMES";
    icd = std::getenv(icd_variable);
  }
  if (icd == nullptr) {
    std::fprintf(stderr,
                 "\nNeither VK_DRIVER_FILES nor VK_ICD_FILENAMES is set, so this "
                 "would exercise whatever\nVulkan driver is installed rather than "
                 "Kniz.  Use the 'kniz' launcher, or set\nthem yourself:\n"
                 "  VK_ICD_FILENAMES=/abs/path/to/kniz_icd.json \\\n"
                 "  VK_DRIVER_FILES=/abs/path/to/kniz_icd.json kniz-info\n");
    return 2;
  }

  section("Driver");
  field("manifest", "%s", icd);
  field("selected by", "%s", icd_variable);
  field("loader", "%s", loader_name);

  void* loader = dlopen(loader_name, RTLD_NOW | RTLD_LOCAL);
  if (loader == nullptr) {
    std::fprintf(stderr, "\ncannot open '%s': %s\n", loader_name, dlerror());
    return 2;
  }

  g_gipa = to_fn<PFN_vkVoidFunction (*)(VkInstance, const char*)>(
      dlsym(loader, "vkGetInstanceProcAddr"));
  if (g_gipa == nullptr) {
    std::fprintf(stderr, "\nloader has no vkGetInstanceProcAddr\n");
    return 2;
  }

  // ---- global commands (the only ones valid with a NULL instance) ----------
  auto create_instance = instance_proc<PFN_vkCreateInstance>(nullptr, "vkCreateInstance");
  if (create_instance == nullptr) {
    std::fprintf(stderr, "\nloader did not return vkCreateInstance\n");
    return 2;
  }

  auto enumerate_loader_version =
      instance_proc<PFN_vkEnumerateInstanceVersion>(nullptr, "vkEnumerateInstanceVersion");
  if (enumerate_loader_version != nullptr) {
    uint32_t version = 0;
    if (enumerate_loader_version(&version) == VK_SUCCESS)
      print_version("loader API", version);
  }

  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "kniz-info";
  app.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pApplicationInfo = &app;

  VkInstance instance = VK_NULL_HANDLE;
  VkResult res = create_instance(&ci, nullptr, &instance);
  if (res != VK_SUCCESS) {
    std::fprintf(stderr,
                 "\nvkCreateInstance failed (VkResult %d).\n"
                 "The OpenGL context could not be created -- run with KNIZ_LOG=3 "
                 "to see why.\n",
                 static_cast<int>(res));
    return 1;
  }

  // ---- instance level commands need a live instance ------------------------
  auto destroy_instance =
      instance_proc<PFN_vkDestroyInstance>(instance, "vkDestroyInstance");
  auto enumerate_physical = instance_proc<PFN_vkEnumeratePhysicalDevices>(
      instance, "vkEnumeratePhysicalDevices");
  auto get_properties = instance_proc<PFN_vkGetPhysicalDeviceProperties>(
      instance, "vkGetPhysicalDeviceProperties");
  auto get_queue_families = instance_proc<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
      instance, "vkGetPhysicalDeviceQueueFamilyProperties");
  auto get_memory = instance_proc<PFN_vkGetPhysicalDeviceMemoryProperties>(
      instance, "vkGetPhysicalDeviceMemoryProperties");
  auto enumerate_device_extensions = instance_proc<PFN_vkEnumerateDeviceExtensionProperties>(
      instance, "vkEnumerateDeviceExtensionProperties");
  auto create_device = instance_proc<PFN_vkCreateDevice>(instance, "vkCreateDevice");
  g_gdpa = instance_proc<PFN_vkGetDeviceProcAddr>(instance, "vkGetDeviceProcAddr");

  section("Instance");
  field("vkCreateInstance", "%s", "ok");

  if (destroy_instance == nullptr || enumerate_physical == nullptr ||
      get_properties == nullptr || get_queue_families == nullptr ||
      get_memory == nullptr || create_device == nullptr || g_gdpa == nullptr) {
    note("missing entry points; this ICD is not usable");
    destroy_instance(instance, nullptr);
    return 1;
  }

  uint32_t count = 0;
  res = enumerate_physical(instance, &count, nullptr);
  if (res != VK_SUCCESS || count == 0) {
    field("physical devices", "%s", "FAILED");
    note("vkEnumeratePhysicalDevices reported no devices");
    destroy_instance(instance, nullptr);
    return 1;
  }
  field("physical devices", "%u", count);

  VkPhysicalDevice physical = VK_NULL_HANDLE;
  enumerate_physical(instance, &count, &physical);

  VkPhysicalDeviceProperties props;
  std::memset(&props, 0, sizeof(props));
  get_properties(physical, &props);

  section("Device");
  field("name", "%s", props.deviceName);
  field("type", "%s", device_type_name(props.deviceType));
  field("vendorID", "0x%04x%s", props.vendorID,
        props.vendorID == 0x4B4E ? "  (Kniz)" : "");
  field("deviceID", "0x%04x", props.deviceID);
  print_version("apiVersion", props.apiVersion);
  print_version("driverVersion", props.driverVersion);

  if (props.vendorID != 0x4B4E) {
    note("NOT Kniz: %s pointed at something else", icd_variable);
    ++g_problems;
  }

  if (enumerate_device_extensions != nullptr) {
    uint32_t ext_count = 0;
    if (enumerate_device_extensions(physical, nullptr, &ext_count, nullptr) ==
        VK_SUCCESS) {
      field("device extensions", "%u", ext_count);
      if (ext_count == 0)
        note("(swapchain and surface support arrive at M3)");
    }
  }

  section("Queues and memory");
  uint32_t family_count = 0;
  get_queue_families(physical, &family_count, nullptr);
  if (family_count > 0 && family_count <= 64) {
    VkQueueFamilyProperties families[64];
    get_queue_families(physical, &family_count, families);
    print_queue_families(families, family_count);
  } else {
    field("families", "%u", family_count);
  }

  VkPhysicalDeviceMemoryProperties memory;
  std::memset(&memory, 0, sizeof(memory));
  get_memory(physical, &memory);
  print_memory(memory);

  section("Limits (Vulkan's required floor)");
  field("maxImage2D", "%u", props.limits.maxImageDimension2D);
  field("maxUBO range", "%u", props.limits.maxUniformBufferRange);
  field("maxSSBO range", "%u", props.limits.maxStorageBufferRange);
  field("maxDescSets", "%u", props.limits.maxBoundDescriptorSets);
  field("maxWorkGroup", "%u x %u x %u", props.limits.maxComputeWorkGroupSize[0],
        props.limits.maxComputeWorkGroupSize[1],
        props.limits.maxComputeWorkGroupSize[2]);
  field("maxViewports", "%u", props.limits.maxViewports);

  // ---- device creation: the real test of the handle plumbing --------------
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
  res = create_device(physical, &dci, nullptr, &device);

  section("Device creation");
  field("vkCreateDevice", "%s", res == VK_SUCCESS ? "ok" : "FAILED");
  if (res != VK_SUCCESS) {
    ++g_problems;
  } else {
    auto destroy_device = device_proc<PFN_vkDestroyDevice>(device, "vkDestroyDevice");
    auto get_device_queue = device_proc<PFN_vkGetDeviceQueue>(device, "vkGetDeviceQueue");
    auto wait_idle = device_proc<PFN_vkDeviceWaitIdle>(device, "vkDeviceWaitIdle");

    VkQueue vk_queue = VK_NULL_HANDLE;
    if (get_device_queue != nullptr)
      get_device_queue(device, 0, 0, &vk_queue);
    field("vkGetDeviceQueue", "%s", vk_queue != VK_NULL_HANDLE ? "ok" : "FAILED");
    if (vk_queue == VK_NULL_HANDLE)
      ++g_problems;

    const bool idle_ok = wait_idle != nullptr && wait_idle(device) == VK_SUCCESS;
    field("vkDeviceWaitIdle", "%s", idle_ok ? "ok" : "FAILED");
    if (!idle_ok)
      ++g_problems;

    if (destroy_device != nullptr)
      destroy_device(device, nullptr);
  }

  // ---------------------------------------------------------------------------
  // Honest capability report.  Update as milestones land.
  //
  // These are claims about the driver, and the tool that reports on the driver
  // is the last place that should be optimistic: everything marked done here is
  // exercised by a test in tests/ that runs on every build.
  // ---------------------------------------------------------------------------
  section("What is implemented");
  capability("loader handshake, instance/device", true, "M0");
  capability("properties, limits, memory model", true, "M0");
  capability("buffers and memory objects", true, "M1");
  capability("SPIR-V -> GLSL translation", true, "M1");
  capability("descriptor sets (storage buffers)", true, "M1");
  capability("compute dispatch", true, "M1");
  capability("images, image views, memory binding", true, "M2a");
  capability("images: clear, upload, readback", true, "M2a");
  capability("render passes, framebuffers, draws", true, "M2b");
  capability("graphics pipelines and rasterisation", true, "M2b");
  capability("uniform buffers", true, "M2c");
  capability("push constants", true, "M2c (lowered onto a uniform buffer)");
  capability("samplers", true, "M2d (GL sampler objects)");
  capability("textures in shaders", true, "M2d (combined or separate)");
  capability("swapchain / window presentation", true,
             "M3/M3c (X11 and Wayland windows, FIFO)");
  capability("acquire / present / image cycling", true, "M3");
  capability("depth attachment, depth test/write/compare", true,
             "M3b (stencil and depth bounds refused)");
  capability("binary semaphores", true,
             "M3b (no state: submission and present are synchronous)");
  capability("linear tiling (layout is observable)", false, "never planned");
  capability("images: storage, input attachments", false, "not implemented");
  capability("mapping an image's allocation", false, "impossible, see DESIGN.md");
  capability("stencil, depth bounds, blending, multisampling", false,
             "refused on purpose, see DESIGN.md");
  capability("dynamic uniform buffers", false, "M2c refused on purpose");
  capability("push constants: non-std140 packing", false,
             "OpenGL cannot express it, see DESIGN.md");
  capability("comparison samplers, anisotropy", false,
             "M2d refused on purpose");
  capability("sampling mip levels above 0", false,
             "only level 0 is ever written, see DESIGN.md");
  capability("Wayland surfaces", true,
             "M3c (wl_egl_window; the context is on the application's "
             "wl_display)");
  capability("sRGB and BGRA swapchain formats", false,
             "M3 refused on purpose");
  capability("MAILBOX and IMMEDIATE present modes", false,
             "presentation is synchronous, see DESIGN.md");
  capability("fences, queries, timestamps", false, "M4");

  std::printf(
      "\n  Images exist and their contents move correctly -- a buffer to image and\n"
      "  back again returns the bytes it was given, a clear lands as the exact\n"
      "  texels its float value denotes, and a sub-rectangle copy respects both its\n"
      "  offset and a padded buffer row length.  Lines and triangles are drawn into\n"
      "  an image through a render pass, a graphics pipeline and one or two vertex\n"
      "  buffers, and the result is checked against a CPU rasterisation of the same\n"
      "  geometry rather than merely eyeballed.  Uniform buffers are read by both\n"
      "  compute and graphics shaders, including two blocks that share a binding\n"
      "  number in different sets, and push constants arrive in the shader that\n"
      "  declared them.  Textures are sampled in both compute and graphics shaders,\n"
      "  through combined image samplers and through a separate image and sampler,\n"
      "  with the filter and address modes coming from the sampler object rather\n"
      "  than from the image.  Swapchains are created on X11 windows and, through\n"
      "  the wayland test, on Wayland ones.  Each is drawn into via an ordinary\n"
      "  render pass and presented: the frame that is about to\n"
      "  be shown is read back and compared against the image, so the present path\n"
      "  is checked rather than looked at.  A D16 depth attachment can be cleared,\n"
      "  tested with the pipeline's compare operation, written and loaded across a\n"
      "  pass; vkcube draws and spins through the whole stack.\n"
      "\n  Not implemented yet, and refused with a specific error rather than\n"
      "  approximated: the stencil test, the depth bounds test, depth clamping and\n"
      "  depth bias, blending, multisampling, instancing, geometry and tessellation\n"
      "  stages, more than one subpass, input attachments, dynamic uniform buffers,\n"
      "  comparison samplers, anisotropic filtering, mip levels above 0, timeline\n"
      "  semaphores, and every swapchain format but\n"
      "  R8G8B8A8_UNORM.\n"
      "  Programs that create a device, ask questions about it, run a compute shader\n"
      "  over buffers, move image contents, draw depth-tested geometry textured from\n"
      "  uniform and sampler data, or draw into a window will work.\n"
      "  See docs/DESIGN.md for what OpenGL cannot express at all.\n"
      "\n  The 'kniz:' lines above are the driver's own log, including the\n"
      "  OpenGL context it created to back this device.\n");

  destroy_instance(instance, nullptr);

  std::printf("\n%s\n", g_problems == 0 ? "OK" : "COMPLETED WITH PROBLEMS");
  return g_problems == 0 ? 0 : 1;
}
