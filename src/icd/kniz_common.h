#pragma once

// Common definitions, logging and small utilities.
//
// Visibility is handled in two layers, and both matter:
//
//   1. the target is compiled with -fvisibility=hidden, so every internal
//      symbol is invisible by default and intra-library references bind at link
//      time instead of going through the dynamic symbol table;
//   2. the linker version script (src/kniz.map) then confines the dynamic
//      symbol table to the vk* namespace.
//
// Layer 1 is not optional.  Without it, taking the address of an internal
// function from a translation unit in the same library produces a *dynamic*
// reference to a symbol the version script has just hidden -- which links
// cleanly, because undefined symbols in a shared object are permitted by
// default, and then fails at load time.  -Wl,-z,defs turns that class of
// mistake into a link error.

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

// The only symbols that may leave the shared object.
#if defined(_WIN32)
#define KNIZ_EXPORT __declspec(dllexport)
#else
#define KNIZ_EXPORT __attribute__((visibility("default")))
#endif

// VK_MAKE_API_VERSION only exists in reasonably recent headers.  Distributions
// ship a wide range, and the extra machinery buys us nothing, so degrade to the
// older macro when it is missing.
#ifndef VK_MAKE_API_VERSION
#define VK_MAKE_API_VERSION(variant, major, minor, patch) \
  VK_MAKE_VERSION(major, minor, patch)
#endif

// Version components are supplied by the build system so the driver version
// reported to applications tracks the project version automatically.
#ifndef KNIZ_VERSION_MAJOR
#define KNIZ_VERSION_MAJOR 0
#endif
#ifndef KNIZ_VERSION_MINOR
#define KNIZ_VERSION_MINOR 0
#endif
#ifndef KNIZ_VERSION_PATCH
#define KNIZ_VERSION_PATCH 0
#endif

namespace kniz {

// Vendor identity.  "KN" -- deliberate and unambiguous: we are not pretending
// to be Intel, AMD, NVIDIA or anyone else's PCI vendor.
inline constexpr uint32_t kVendorId = 0x4B4E;
inline constexpr uint32_t kDriverVersion =
    VK_MAKE_API_VERSION(0, KNIZ_VERSION_MAJOR, KNIZ_VERSION_MINOR,
                        KNIZ_VERSION_PATCH);

enum class LogLevel : int {
  Error = 0,
  Warn = 1,
  Info = 2,
  Debug = 3,
  Trace = 4,
};

// Threshold comes from the KNIZ_LOG environment variable (0..4), default Info.
int log_level();

#if defined(__GNUC__) || defined(__clang__)
#define KNIZ_PRINTF(a, b) __attribute__((format(printf, a, b)))
#else
#define KNIZ_PRINTF(a, b)
#endif

void log_write(LogLevel level, const char* fmt, ...) KNIZ_PRINTF(2, 3);

#define KNIZ_LOG(level, ...)                                                  \
  do {                                                                        \
    if (static_cast<int>(::kniz::LogLevel::level) <= ::kniz::log_level())     \
      ::kniz::log_write(::kniz::LogLevel::level, __VA_ARGS__);                \
  } while (0)

#define KNIZ_ERR(...) KNIZ_LOG(Error, __VA_ARGS__)
#define KNIZ_WARN(...) KNIZ_LOG(Warn, __VA_ARGS__)
#define KNIZ_INFO(...) KNIZ_LOG(Info, __VA_ARGS__)
#define KNIZ_DBG(...) KNIZ_LOG(Debug, __VA_ARGS__)
#define KNIZ_TRACE(...) KNIZ_LOG(Trace, __VA_ARGS__)

// Copy `src` into the fixed-size Vulkan character array `dst`, always
// null-terminating.  Vulkan's own guidance is that strings are truncated rather
// than rejected, so this is intentionally lossy.
template <size_t N>
inline void copy_string(char (&dst)[N], const char* src) {
  if (src == nullptr)
    src = "";
  const size_t len = std::strlen(src);
  const size_t n = len < N - 1 ? len : N - 1;
  std::memcpy(dst, src, n);
  dst[n] = '\0';
}

// Implements the standard two-call enumeration pattern shared by every
// vkEnumerate* function: query the count with a null output pointer, then fill
// or report VK_INCOMPLETE.
inline VkResult enumerate_count(uint32_t available, uint32_t* pCount,
                                void* pOut) {
  if (pOut == nullptr) {
    *pCount = available;
    return VK_SUCCESS;
  }
  if (*pCount < available) {
    *pCount = available;
    return VK_INCOMPLETE;
  }
  *pCount = available;
  return VK_SUCCESS;
}

}  // namespace kniz
