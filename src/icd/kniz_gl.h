#pragma once

// The OpenGL backend.
//
// Kniz owns its GL context rather than borrowing one from the application:
// the application is talking Vulkan and should not have to care that OpenGL
// exists.  M0 creates a headless (window system independent) context purely to
// prove the backend is reachable and to report what we actually got.
//
// Later milestones replace/extend this with:
//   - one context per Vulkan calling thread (GL state is per-thread and
//     "current", while Vulkan is thread agnostic), and
//   - a window-system surface for the swapchain (see docs/DESIGN.md, WSI).

#include <cstdint>
#include <string>

namespace kniz::gl {

struct ContextInfo {
  bool valid = false;
  int major = 0;
  int minor = 0;
  bool core = false;
  std::string vendor;
  std::string renderer;
  std::string version;
  std::string shading_language;
};

// Create a headless GL context and make it current on the calling thread.
// Idempotent: repeated calls are cheap no-ops.  Returns false on failure, in
// which case the reason has already been logged.
bool create_headless();

// Release the context.  Safe to call when nothing was created.
void destroy();

// Describes the context returned by the most recent create_headless().
const ContextInfo& info();

// ---------------------------------------------------------------------------
// Entry point table
//
// Deliberately not GLAD: a generated loader would add a Python/codegen step to
// every build.  We resolve what we need through eglGetProcAddress at runtime,
// which also means we do not link libGL and therefore cannot conflict with the
// application's own GL stack.
//
// Only the functions the current milestone uses are declared.  Add to this
// struct as the backend grows.
// ---------------------------------------------------------------------------

struct Functions {
  const unsigned char* (*GetString)(unsigned int name) = nullptr;
  void (*GetIntegerv)(unsigned int pname, int* data) = nullptr;
  unsigned int (*GetError)() = nullptr;
};

extern Functions f;

}  // namespace kniz::gl
