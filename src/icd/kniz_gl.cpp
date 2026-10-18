#include "kniz_gl.h"

#include "kniz_common.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <dlfcn.h>

#include <cstdio>

namespace kniz::gl {

Functions f;

namespace {

EGLDisplay g_display = EGL_NO_DISPLAY;
EGLConfig g_config = nullptr;
EGLSurface g_surface = EGL_NO_SURFACE;
EGLContext g_context = EGL_NO_CONTEXT;
bool g_surfaceless = false;

ContextInfo g_info;
bool g_attempted = false;
bool g_ok = false;

// GL tokens spelled out locally.  Including <GL/gl.h> would risk picking up a
// GL or GLES header that disagrees with the EGL we resolved against; we only
// need a handful of constants.
constexpr unsigned int kGlVendor = 0x1F00;
constexpr unsigned int kGlRenderer = 0x1F01;
constexpr unsigned int kGlVersion = 0x1F02;
constexpr unsigned int kGlShadingLanguageVersion = 0x8B8C;
constexpr unsigned int kGlNoError = 0;

// Resolve one entry point.  eglGetProcAddress is the sanctioned route, but a
// number of stacks return NULL for entry points that are exported directly by
// the GL library, so dlsym is the fallback.  Both results are copied through
// memcpy rather than reinterpret_cast from void* to avoid a
// conditionally-supported conversion.
bool load_proc(void* slot, size_t slot_size, const char* name) {
  auto fn = eglGetProcAddress(name);
  if (fn != nullptr) {
    if (slot_size != sizeof(fn)) {
      KNIZ_ERR("size mismatch resolving %s", name);
      return false;
    }
    std::memcpy(slot, &fn, slot_size);
    return true;
  }

  void* p = dlsym(RTLD_DEFAULT, name);
  if (p == nullptr) {
    KNIZ_ERR("EGL could not resolve %s", name);
    return false;
  }
  if (slot_size != sizeof(p)) {
    KNIZ_ERR("size mismatch resolving %s", name);
    return false;
  }
  std::memcpy(slot, &p, slot_size);
  return true;
}

void release();
bool bring_up(EGLDisplay display, bool wants_surface, const char* label);
bool finish_setup();

// Probe the GL version from the version string, which is specified to begin
// with "<major>.<minor>" regardless of profile or vendor.
void parse_version(const char* version, ContextInfo& info) {
  unsigned major = 0;
  unsigned minor = 0;
  if (version != nullptr && std::sscanf(version, "%u.%u", &major, &minor) == 2) {
    info.major = static_cast<int>(major);
    info.minor = static_cast<int>(minor);
  }
}

bool query_context_info() {
  auto str = [](unsigned int name) -> const char* {
    const unsigned char* s = f.GetString ? f.GetString(name) : nullptr;
    return reinterpret_cast<const char*>(s);
  };

  const char* version = str(kGlVersion);
  if (version == nullptr) {
    // glGetString only returns NULL before a context is current or after a lost
    // context; neither should be reachable here.
    KNIZ_ERR("glGetString(GL_VERSION) returned NULL");
    return false;
  }

  if (const char* vendor = str(kGlVendor))
    g_info.vendor = vendor;
  if (const char* renderer = str(kGlRenderer))
    g_info.renderer = renderer;
  g_info.version = version;
  if (const char* glsl = str(kGlShadingLanguageVersion))
    g_info.shading_language = glsl;

  parse_version(version, g_info);
  return true;
}

}  // namespace

const ContextInfo& info() { return g_info; }

bool create_headless() {
  if (g_attempted)
    return g_ok;
  g_attempted = true;

  // Two ways to get a context with no window, tried in order.
  //
  //   1. The default display with a pbuffer.  This is what an ordinary desktop
  //      session gives us, and it is the better option when it works because it
  //      goes through the vendor's normal platform path.
  //   2. The surfaceless platform.  This is the one that works in containers,
  //      CI runners and any other place with no X11 or Wayland display, and it
  //      is why the backend does not simply require EGL_DEFAULT_DISPLAY to
  //      succeed.
  //
  // Both are headless: no window is created and nothing is ever presented.  M3
  // adds the third path, a real surface for the swapchain.
  if (bring_up(eglGetDisplay(EGL_DEFAULT_DISPLAY), true, "default display")) {
    if (finish_setup())
      return true;
    KNIZ_ERR("OpenGL context came up but could not be used");
    release();
    return false;
  }

#if defined(EGL_VERSION_1_5) && defined(EGL_PLATFORM_SURFACELESS_MESA)
  KNIZ_DBG("retrying with the surfaceless EGL platform");
  if (bring_up(eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA,
                                     EGL_DEFAULT_DISPLAY, nullptr),
               false, "surfaceless platform")) {
    if (finish_setup())
      return true;
    KNIZ_ERR("OpenGL context came up but could not be used");
    release();
    return false;
  }
#else
  KNIZ_WARN("surfaceless EGL unavailable at build time; a display is required");
#endif

  KNIZ_ERR("no usable OpenGL context: neither the default display nor the "
           "surfaceless platform produced one");
  release();
  return false;
}

namespace {

// Brings up a context on one display, leaving the globals set on success.
bool bring_up(EGLDisplay display, bool wants_surface, const char* label) {
  if (display == EGL_NO_DISPLAY) {
    KNIZ_DBG("%s: no display", label);
    return false;
  }

  EGLint egl_major = 0;
  EGLint egl_minor = 0;
  if (eglInitialize(display, &egl_major, &egl_minor) != EGL_TRUE) {
    KNIZ_DBG("%s: eglInitialize failed (0x%04x)", label, eglGetError());
    return false;
  }

  if (eglBindAPI(EGL_OPENGL_API) != EGL_TRUE) {
    KNIZ_DBG("%s: this EGL has no desktop OpenGL support", label);
    eglTerminate(display);
    return false;
  }

  // EGL_SURFACE_TYPE must always be supplied, even for a surfaceless display.
  //
  // Leaving it out does not mean "any surface type": Mesa's surfaceless
  // platform then matches no configs at all, and reports it as EGL_SUCCESS with
  // a count of zero, so there is nothing in the error code to go on.  Requesting
  // pbuffer-capable configs works on both display kinds; only actual surface
  // creation differs between them.
  EGLint config_attribs[16];
  int n = 0;
  config_attribs[n++] = EGL_SURFACE_TYPE;
  config_attribs[n++] = EGL_PBUFFER_BIT;
  config_attribs[n++] = EGL_RENDERABLE_TYPE;
  config_attribs[n++] = EGL_OPENGL_BIT;
  config_attribs[n++] = EGL_RED_SIZE;
  config_attribs[n++] = 8;
  config_attribs[n++] = EGL_GREEN_SIZE;
  config_attribs[n++] = 8;
  config_attribs[n++] = EGL_BLUE_SIZE;
  config_attribs[n++] = 8;
  config_attribs[n++] = EGL_DEPTH_SIZE;
  config_attribs[n++] = 24;
  config_attribs[n++] = EGL_NONE;

  EGLConfig config = nullptr;
  EGLint num_configs = 0;
  if (eglChooseConfig(display, config_attribs, &config, 1, &num_configs) !=
          EGL_TRUE ||
      num_configs < 1) {
    KNIZ_DBG("%s: no matching EGL config (0x%04x)", label, eglGetError());
    eglTerminate(display);
    return false;
  }

  EGLSurface surface = EGL_NO_SURFACE;
  if (wants_surface) {
    const EGLint pbuffer_attribs[] = {
        EGL_WIDTH,  16,
        EGL_HEIGHT, 16,
        EGL_NONE,
    };
    surface = eglCreatePbufferSurface(display, config, pbuffer_attribs);
    if (surface == EGL_NO_SURFACE) {
      KNIZ_DBG("%s: eglCreatePbufferSurface failed (0x%04x)", label,
               eglGetError());
      eglTerminate(display);
      return false;
    }
  }

  // Ask for the newest context we know how to use first, then walk down.  The
  // bottom entries are compatibility-profile fallbacks for stacks that refuse
  // to hand out a core context.
  struct Candidate {
    int major;
    int minor;
    bool core;
  };
  constexpr Candidate kCandidates[] = {
      {4, 6, true},  {4, 5, true},  {4, 3, true},  {3, 3, true},
      {4, 6, false}, {3, 3, false}, {2, 1, false},
  };

  for (const Candidate& c : kCandidates) {
    const EGLint ctx_attribs[] = {
        EGL_CONTEXT_MAJOR_VERSION_KHR,
        c.major,
        EGL_CONTEXT_MINOR_VERSION_KHR,
        c.minor,
        EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR,
        c.core ? EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR
               : EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT_KHR,
        EGL_NONE,
    };

    EGLContext context =
        eglCreateContext(display, config, EGL_NO_CONTEXT, ctx_attribs);
    if (context == EGL_NO_CONTEXT)
      continue;

    if (eglMakeCurrent(display, surface, surface, context) == EGL_TRUE) {
      g_display = display;
      g_config = config;
      g_surface = surface;
      g_context = context;
      g_surfaceless = !wants_surface;
      g_info.core = c.core;

      KNIZ_DBG("OpenGL %d.%d %s context via %s (EGL %d.%d)", c.major, c.minor,
               c.core ? "core" : "compatibility", label, egl_major, egl_minor);
      return true;
    }

    eglDestroyContext(display, context);
  }

  KNIZ_DBG("%s: no context version could be made current", label);
  if (surface != EGL_NO_SURFACE)
    eglDestroySurface(display, surface);
  eglTerminate(display);
  return false;
}

void release() {
  if (g_display != EGL_NO_DISPLAY) {
    eglMakeCurrent(g_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (g_context != EGL_NO_CONTEXT)
      eglDestroyContext(g_display, g_context);
    if (g_surface != EGL_NO_SURFACE)
      eglDestroySurface(g_display, g_surface);
    eglTerminate(g_display);
  }

  g_display = EGL_NO_DISPLAY;
  g_config = nullptr;
  g_surface = EGL_NO_SURFACE;
  g_context = EGL_NO_CONTEXT;
  g_surfaceless = false;
}

}  // namespace

void destroy() {
  release();
  g_info = ContextInfo{};
  g_ok = false;
  g_attempted = false;
  f = Functions{};
}

namespace {

// Fills in the entry point table and the context description once a context is
// current.  Kept separate from bring_up so both display paths share it.
bool finish_setup() {
  if (!load_proc(&f.GetString, sizeof(f.GetString), "glGetString") ||
      !load_proc(&f.GetIntegerv, sizeof(f.GetIntegerv), "glGetIntegerv") ||
      !load_proc(&f.GetError, sizeof(f.GetError), "glGetError"))
    return false;

  if (!query_context_info())
    return false;

  if (f.GetError && f.GetError() != kGlNoError)
    KNIZ_WARN("GL error set while querying context info");

  g_info.valid = true;
  g_ok = true;

  KNIZ_INFO("OpenGL %d.%d %s context%s: %s / %s", g_info.major, g_info.minor,
            g_info.core ? "core" : "compatibility",
            g_surfaceless ? " (surfaceless)" : "", g_info.renderer.c_str(),
            g_info.version.c_str());
  return true;
}

}  // namespace

}  // namespace kniz::gl
