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

// Every test must be pointed at the ICD under test.  With VK_ICD_FILENAMES
// unset the loader would select the machine's own driver, and the suite would
// pass while proving nothing at all about Kniz.
inline bool icd_selected() {
  if (std::getenv("VK_ICD_FILENAMES") != nullptr)
    return true;
  std::fprintf(stderr,
               "VK_ICD_FILENAMES is not set; the test would run against "
               "whatever ICD is registered on this system\n");
  return false;
}

}  // namespace kniz_test
