#include "kniz_gl.h"

#include "kniz_common.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cstdint>
#include <vector>

namespace kniz::gl {

Functions f;

namespace {

EGLDisplay g_display = EGL_NO_DISPLAY;
EGLConfig g_config = nullptr;
EGLSurface g_surface = EGL_NO_SURFACE;
EGLContext g_context = EGL_NO_CONTEXT;
bool g_surfaceless = false;
bool g_window_capable = false;
WindowPlatform g_platform = WindowPlatform::None;
// The wl_display the context's EGL display was built on, when it was built on
// the Wayland platform.  A Wayland EGL display draws to windows that belong to
// the connection it was built on, so this pointer is what decides whether a
// surface's display can be served by the context as it stands.
const void* g_wayland_display = nullptr;

// Live X11 window surfaces.  Wayland surfaces are counted by
// g_wayland_pairings.size(); X11 windows need a tally of their own (an XID has
// no window to pair with: XIDs are server-wide, so the EGL surface alone is the
// whole of the state).  Rebuilding the context on another platform strands
// every live EGLSurface of the old one, so a rebuild is allowed only while both
// counts are zero.
int g_x11_windows = 0;

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
constexpr unsigned int kGlShaderStorageBufferOffsetAlignment = 0x90DF;
constexpr unsigned int kGlUniformBufferOffsetAlignment = 0x8A34;

// Resolve one entry point.  eglGetProcAddress is the sanctioned route, but a
// number of stacks return NULL for entry points that are exported directly by
// the GL library, so dlsym is the fallback.  Both results are copied through
// memcpy rather than reinterpret_cast from void* to avoid a
// conditionally-supported conversion.
bool resolve_proc(void* slot, size_t slot_size, const char* name, bool quiet) {
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
    if (!quiet)
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

bool load_proc(void* slot, size_t slot_size, const char* name) {
  return resolve_proc(slot, slot_size, name, false);
}

// Silent variant.  Some entry points are legitimately absent on contexts that
// are otherwise perfectly usable -- compute on a GL 3.3 context, for instance --
// and reporting each one as an error would turn a supported configuration into
// something that looks broken.
bool try_load_proc(void* slot, size_t slot_size, const char* name) {
  return resolve_proc(slot, slot_size, name, true);
}

// One row of the entry point table, so the loader loop below stays a loop.
struct ProcSlot {
  void* slot;
  size_t size;
  const char* name;
};

void release();
bool bring_up(EGLDisplay display, bool wants_surface, const char* label);

// Whether an EGL client extension is present, asked before any display exists.
bool has_client_extension(const char* name) {
  const char* extensions = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
  return extensions != nullptr && std::strstr(extensions, name) != nullptr;
}

// The window system a display belongs to, as far as its extension string says.
// Used only for the display obtained from EGL_DEFAULT_DISPLAY, where the
// platform was not chosen by us; a display we asked for by name is known.
WindowPlatform platform_of_display(EGLDisplay display) {
  const char* extensions = eglQueryString(display, EGL_EXTENSIONS);
  if (extensions == nullptr)
    return WindowPlatform::Other;
  if (std::strstr(extensions, "EGL_KHR_platform_x11") != nullptr ||
      std::strstr(extensions, "EGL_EXT_platform_x11") != nullptr)
    return WindowPlatform::X11;
  if (std::strstr(extensions, "EGL_KHR_platform_wayland") != nullptr ||
      std::strstr(extensions, "EGL_EXT_platform_wayland") != nullptr)
    return WindowPlatform::Wayland;
  return WindowPlatform::Other;
}

bool finish_setup(WindowPlatform platform);

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

// ---------------------------------------------------------------------------
// libwayland-egl
//
// A Wayland window surface is a wl_egl_window wrapped around the application's
// wl_surface; EGL's Wayland platform draws to that window, never to the
// wl_surface itself.  The three functions that manage the wrapper live in
// libwayland-egl, and Kniz opens it with dlopen rather than link against it:
// the driver's build has no Wayland requirement, and a machine that never
// touches Wayland never loads the library.  This mirrors how the loader itself
// and every test binary treat their platform libraries.
//
// The three functions have been stable in name and signature for as long as
// the library has existed, which is as much of an ABI contract as dlopen needs.
// ---------------------------------------------------------------------------

struct WaylandEglProcs {
  void* (*create)(void* surface, int width, int height);
  void (*resize)(void* window, int width, int height, int dx, int dy);
  void (*destroy)(void* window);
};

WaylandEglProcs g_wayland_egl{};
bool g_wayland_egl_loaded = false;

bool load_wayland_egl() {
  if (g_wayland_egl_loaded)
    return g_wayland_egl.create != nullptr;
  g_wayland_egl_loaded = true;

  void* library = dlopen("libwayland-egl.so.1", RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    KNIZ_ERR("libwayland-egl.so.1 could not be loaded (%s); Wayland surfaces "
             "need it to turn a wl_surface into something EGL can draw to",
             dlerror());
    return false;
  }
  // Resolved from the handle directly: try_load_proc's routes are
  // eglGetProcAddress and dlsym(RTLD_DEFAULT), and neither of those can reach
  // a library opened with RTLD_LOCAL.
  const auto resolve = [library](void* slot, size_t slot_size,
                                 const char* name) {
    void* symbol = dlsym(library, name);
    if (symbol == nullptr) {
      KNIZ_ERR("libwayland-egl has no %s", name);
      return false;
    }
    if (slot_size != sizeof(symbol)) {
      KNIZ_ERR("size mismatch resolving %s", name);
      return false;
    }
    std::memcpy(slot, &symbol, slot_size);
    return true;
  };
  bool ok = true;
  ok &= resolve(&g_wayland_egl.create, sizeof(g_wayland_egl.create),
                "wl_egl_window_create");
  ok &= resolve(&g_wayland_egl.resize, sizeof(g_wayland_egl.resize),
                "wl_egl_window_resize");
  ok &= resolve(&g_wayland_egl.destroy, sizeof(g_wayland_egl.destroy),
                "wl_egl_window_destroy");
  return ok;
}

// The EGL surface and its wl_egl_window travel together: the window is created
// before the EGL surface and has to be destroyed after it, and every operation
// that either changes the window's size or tears it down needs to find the
// window again from the surface alone.  Surfaces number in the ones, so a
// linear scan is the whole of the cost.
struct WindowPairing {
  EGLSurface egl_surface;
  void* wayland_window;
};

std::vector<WindowPairing> g_wayland_pairings;

WindowPairing* find_pairing(WindowSurface surface) {
  const EGLSurface egl_surface = reinterpret_cast<EGLSurface>(surface);
  for (WindowPairing& pairing : g_wayland_pairings) {
    if (pairing.egl_surface == egl_surface)
      return &pairing;
  }
  return nullptr;
}

}  // namespace

namespace {

// Entry points M1 needs to move bytes: buffers, shaders and programs.
constexpr ProcSlot kBufferProcs[] = {
    {&f.GenBuffers, sizeof(f.GenBuffers), "glGenBuffers"},
    {&f.DeleteBuffers, sizeof(f.DeleteBuffers), "glDeleteBuffers"},
    {&f.BindBuffer, sizeof(f.BindBuffer), "glBindBuffer"},
    {&f.BufferData, sizeof(f.BufferData), "glBufferData"},
    {&f.BufferSubData, sizeof(f.BufferSubData), "glBufferSubData"},
    {&f.GetBufferSubData, sizeof(f.GetBufferSubData), "glGetBufferSubData"},
    {&f.BindBufferBase, sizeof(f.BindBufferBase), "glBindBufferBase"},
    {&f.BindBufferRange, sizeof(f.BindBufferRange), "glBindBufferRange"},
    {&f.CopyBufferSubData, sizeof(f.CopyBufferSubData), "glCopyBufferSubData"},
    {&f.CreateShader, sizeof(f.CreateShader), "glCreateShader"},
    {&f.ShaderSource, sizeof(f.ShaderSource), "glShaderSource"},
    {&f.CompileShader, sizeof(f.CompileShader), "glCompileShader"},
    {&f.GetShaderiv, sizeof(f.GetShaderiv), "glGetShaderiv"},
    {&f.GetShaderInfoLog, sizeof(f.GetShaderInfoLog), "glGetShaderInfoLog"},
    {&f.DeleteShader, sizeof(f.DeleteShader), "glDeleteShader"},
    {&f.CreateProgram, sizeof(f.CreateProgram), "glCreateProgram"},
    {&f.AttachShader, sizeof(f.AttachShader), "glAttachShader"},
    {&f.LinkProgram, sizeof(f.LinkProgram), "glLinkProgram"},
    {&f.GetProgramiv, sizeof(f.GetProgramiv), "glGetProgramiv"},
    {&f.GetProgramInfoLog, sizeof(f.GetProgramInfoLog), "glGetProgramInfoLog"},
    {&f.DeleteProgram, sizeof(f.DeleteProgram), "glDeleteProgram"},
    {&f.UseProgram, sizeof(f.UseProgram), "glUseProgram"},
    {&f.Finish, sizeof(f.Finish), "glFinish"},
    {&f.Flush, sizeof(f.Flush), "glFlush"},
};

// Compute dispatch.  Absent on anything before GL 4.3, which is a supported
// configuration rather than a broken one.
constexpr ProcSlot kComputeProcs[] = {
    {&f.DispatchCompute, sizeof(f.DispatchCompute), "glDispatchCompute"},
    {&f.MemoryBarrier, sizeof(f.MemoryBarrier), "glMemoryBarrier"},
};

// Textures, for the image half of the backend.  glTexStorage2D is 4.2,
// glClearTexImage is 4.4 and glGetTexImage returned to the core profile in 4.5,
// so 4.5 is the floor for the set as a whole.
constexpr ProcSlot kTextureProcs[] = {
    {&f.GenTextures, sizeof(f.GenTextures), "glGenTextures"},
    {&f.DeleteTextures, sizeof(f.DeleteTextures), "glDeleteTextures"},
    {&f.BindTexture, sizeof(f.BindTexture), "glBindTexture"},
    {&f.ActiveTexture, sizeof(f.ActiveTexture), "glActiveTexture"},
    {&f.TexParameteri, sizeof(f.TexParameteri), "glTexParameteri"},
    {&f.TexStorage2D, sizeof(f.TexStorage2D), "glTexStorage2D"},
    {&f.TexSubImage2D, sizeof(f.TexSubImage2D), "glTexSubImage2D"},
    {&f.GetTexImage, sizeof(f.GetTexImage), "glGetTexImage"},
    {&f.ClearTexImage, sizeof(f.ClearTexImage), "glClearTexImage"},
    {&f.PixelStorei, sizeof(f.PixelStorei), "glPixelStorei"},
};

// Sampler objects: filter, wrap, LOD and border state with no image in it, which
// is exactly what a Vulkan sampler is.  Core in 3.3.
constexpr ProcSlot kSamplerProcs[] = {
    {&f.GenSamplers, sizeof(f.GenSamplers), "glGenSamplers"},
    {&f.DeleteSamplers, sizeof(f.DeleteSamplers), "glDeleteSamplers"},
    {&f.BindSampler, sizeof(f.BindSampler), "glBindSampler"},
    {&f.SamplerParameteri, sizeof(f.SamplerParameteri), "glSamplerParameteri"},
    {&f.SamplerParameterf, sizeof(f.SamplerParameterf), "glSamplerParameterf"},
    {&f.SamplerParameterfv, sizeof(f.SamplerParameterfv), "glSamplerParameterfv"},
};

// Framebuffers, vertex arrays, draws and raster state.  Core in 3.0, so on any
// context that got this far the only way these are missing is a broken stack --
// which is exactly what the capability flag is for.
constexpr ProcSlot kRenderProcs[] = {
    {&f.GenFramebuffers, sizeof(f.GenFramebuffers), "glGenFramebuffers"},
    {&f.DeleteFramebuffers, sizeof(f.DeleteFramebuffers), "glDeleteFramebuffers"},
    {&f.BindFramebuffer, sizeof(f.BindFramebuffer), "glBindFramebuffer"},
    {&f.FramebufferTexture2D, sizeof(f.FramebufferTexture2D),
     "glFramebufferTexture2D"},
    {&f.CheckFramebufferStatus, sizeof(f.CheckFramebufferStatus),
     "glCheckFramebufferStatus"},
    {&f.DrawBuffers, sizeof(f.DrawBuffers), "glDrawBuffers"},
    {&f.ClearBufferfv, sizeof(f.ClearBufferfv), "glClearBufferfv"},
    {&f.ClearBufferfi, sizeof(f.ClearBufferfi), "glClearBufferfi"},
    {&f.GenVertexArrays, sizeof(f.GenVertexArrays), "glGenVertexArrays"},
    {&f.DeleteVertexArrays, sizeof(f.DeleteVertexArrays), "glDeleteVertexArrays"},
    {&f.BindVertexArray, sizeof(f.BindVertexArray), "glBindVertexArray"},
    {&f.EnableVertexAttribArray, sizeof(f.EnableVertexAttribArray),
     "glEnableVertexAttribArray"},
    {&f.DisableVertexAttribArray, sizeof(f.DisableVertexAttribArray),
     "glDisableVertexAttribArray"},
    {&f.VertexAttribPointer, sizeof(f.VertexAttribPointer),
     "glVertexAttribPointer"},
    {&f.DrawArrays, sizeof(f.DrawArrays), "glDrawArrays"},
    {&f.DrawElements, sizeof(f.DrawElements), "glDrawElements"},
    {&f.DrawElementsBaseVertex, sizeof(f.DrawElementsBaseVertex),
     "glDrawElementsBaseVertex"},
    {&f.Viewport, sizeof(f.Viewport), "glViewport"},
    {&f.Scissor, sizeof(f.Scissor), "glScissor"},
    {&f.Enable, sizeof(f.Enable), "glEnable"},
    {&f.Disable, sizeof(f.Disable), "glDisable"},
    {&f.CullFace, sizeof(f.CullFace), "glCullFace"},
    {&f.FrontFace, sizeof(f.FrontFace), "glFrontFace"},
    {&f.ColorMask, sizeof(f.ColorMask), "glColorMask"},
    {&f.DepthMask, sizeof(f.DepthMask), "glDepthMask"},
    {&f.DepthFunc, sizeof(f.DepthFunc), "glDepthFunc"},
    {&f.DepthRange, sizeof(f.DepthRange), "glDepthRange"},
    {&f.ReadPixels, sizeof(f.ReadPixels), "glReadPixels"},
};

template <size_t N>
bool load_all(const ProcSlot (&procs)[N]) {
  bool all = true;
  for (const ProcSlot& p : procs)
    all &= try_load_proc(p.slot, p.size, p.name);
  return all;
}

}  // namespace

const ContextInfo& info() { return g_info; }

bool report_errors(const char* where) {
  if (f.GetError == nullptr)
    return false;
  bool found = false;
  for (int i = 0; i < 8; ++i) {
    const Enum error = f.GetError();
    if (error == kNoError)
      break;
    KNIZ_ERR("OpenGL error 0x%04x (%s)", error, where);
    found = true;
  }
  return found;
}

bool make_current() {
  if (g_context == EGL_NO_CONTEXT || g_display == EGL_NO_DISPLAY)
    return false;
  return eglMakeCurrent(g_display, g_surface, g_surface, g_context) == EGL_TRUE;
}

WindowPlatform window_platform() { return g_platform; }

// ---------------------------------------------------------------------------
// M3: window surfaces
// ---------------------------------------------------------------------------

WindowSurface create_window_surface(uint64_t native_window) {
  if (g_display == EGL_NO_DISPLAY || g_config == nullptr ||
      g_context == EGL_NO_CONTEXT) {
    KNIZ_ERR("cannot create a window surface: there is no OpenGL context");
    return nullptr;
  }
  if (g_platform != WindowPlatform::X11) {
    // The handle is an X11 XID and this display is something else.  Handing it
    // to eglCreateWindowSurface would have EGL interpret it as a pointer to a
    // Wayland surface or worse, so it is refused here with the reason.
    KNIZ_ERR("cannot create an X11 window surface: the OpenGL context is on the "
             "%s platform, and a GL context cannot be moved to another window "
             "system once its objects exist",
             g_platform == WindowPlatform::Wayland   ? "Wayland"
             : g_platform == WindowPlatform::None    ? "no"
                                                     : "an unidentified");
    return nullptr;
  }
  if (!g_window_capable) {
    KNIZ_ERR("cannot create a window surface: this OpenGL context has no "
             "display or config that supports windows%s",
             g_surfaceless ? " (it is a surfaceless context)" : "");
    return nullptr;
  }

  // The native handle is an X11 XID, which is what EGL wants for an X11
  // window.  The connection is EGL's own: XIDs are server-wide, so a window
  // created on the application's connection is still a window EGL can draw to.
  const EGLNativeWindowType window =
      static_cast<EGLNativeWindowType>(native_window);
  const EGLint attribs[] = {EGL_NONE};
  EGLSurface surface = eglCreateWindowSurface(g_display, g_config, window, attribs);
  if (surface == EGL_NO_SURFACE) {
    KNIZ_ERR("eglCreateWindowSurface failed (0x%04x)", eglGetError());
    return nullptr;
  }
  ++g_x11_windows;

  // The swap interval is the whole of what FIFO means here: the presentation
  // queue is the compositor's, and an interval of 1 is what makes eglSwapBuffers
  // wait for it.  EGL's default is already 1, but the one present mode this
  // driver offers should not depend on a default being remembered, and the
  // interval can only be set while the surface is current.
  WindowSurface handle = reinterpret_cast<WindowSurface>(surface);
  if (bind_window_surface(handle)) {
    eglSwapInterval(g_display, 1);
    bind_headless_surface();
  }

  KNIZ_DBG("window surface created for native window 0x%llx",
           static_cast<unsigned long long>(native_window));
  return handle;
}

void destroy_window_surface(WindowSurface surface) {
  if (surface == nullptr || g_display == EGL_NO_DISPLAY)
    return;
  EGLSurface egl_surface = reinterpret_cast<EGLSurface>(surface);
  // A surface that is still current on some thread cannot be destroyed, and the
  // thread that created it may not be this one.  Making it not-current on the
  // calling thread is all that can be done from here.  The headless drawable is
  // then made current again: the context has to keep somewhere to draw, and every
  // GL call after this one assumes it has.
  eglMakeCurrent(g_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  eglDestroySurface(g_display, egl_surface);
  // If this surface went through a wl_egl_window, destroy that too, after the
  // EGL surface: Mesa's Wayland platform keeps a reference to the window for
  // as long as the EGL surface lives.
  if (WindowPairing* pairing = find_pairing(surface)) {
    if (pairing->wayland_window != nullptr && g_wayland_egl.destroy != nullptr)
      g_wayland_egl.destroy(pairing->wayland_window);
    for (size_t i = 0; i < g_wayland_pairings.size(); ++i) {
      if (&g_wayland_pairings[i] == pairing) {
        g_wayland_pairings.erase(g_wayland_pairings.begin() +
                                 static_cast<std::ptrdiff_t>(i));
        break;
      }
    }
    KNIZ_DBG("wl_egl_window destroyed with its EGL surface");
  } else {
    // An X11 window surface: no pairing, but the drawable's tally still owes
    // one.
    --g_x11_windows;
  }
  bind_headless_surface();
}

bool bind_window_surface(WindowSurface surface) {
  if (surface == nullptr || g_display == EGL_NO_DISPLAY ||
      g_context == EGL_NO_CONTEXT)
    return false;
  EGLSurface egl_surface = reinterpret_cast<EGLSurface>(surface);
  if (eglMakeCurrent(g_display, egl_surface, egl_surface, g_context) != EGL_TRUE) {
    KNIZ_ERR("could not make the window surface current (0x%04x)", eglGetError());
    return false;
  }
  return true;
}

bool bind_headless_surface() {
  if (g_display == EGL_NO_DISPLAY || g_context == EGL_NO_CONTEXT)
    return false;
  return eglMakeCurrent(g_display, g_surface, g_surface, g_context) == EGL_TRUE;
}

bool window_surface_size(WindowSurface surface, int* width, int* height) {
  if (surface == nullptr || g_display == EGL_NO_DISPLAY || width == nullptr ||
      height == nullptr)
    return false;
  EGLint w = 0;
  EGLint h = 0;
  EGLSurface egl_surface = reinterpret_cast<EGLSurface>(surface);
  if (eglQuerySurface(g_display, egl_surface, EGL_WIDTH, &w) != EGL_TRUE ||
      eglQuerySurface(g_display, egl_surface, EGL_HEIGHT, &h) != EGL_TRUE) {
    KNIZ_ERR("could not query the window surface size (0x%04x)", eglGetError());
    return false;
  }
  *width = w;
  *height = h;
  return true;
}

bool swap_window_surface(WindowSurface surface) {
  if (surface == nullptr || g_display == EGL_NO_DISPLAY)
    return false;
  EGLSurface egl_surface = reinterpret_cast<EGLSurface>(surface);
  if (eglSwapBuffers(g_display, egl_surface) != EGL_TRUE) {
    KNIZ_ERR("eglSwapBuffers failed (0x%04x)", eglGetError());
    return false;
  }
  return true;
}

WindowSurface create_window_surface_wayland(const void* wl_surface,
                                            int width, int height) {
  if (wl_surface == nullptr || width <= 0 || height <= 0) {
    KNIZ_ERR("a Wayland window surface needs a wl_surface and a positive size");
    return nullptr;
  }
  if (g_display == EGL_NO_DISPLAY || g_config == nullptr ||
      g_context == EGL_NO_CONTEXT) {
    KNIZ_ERR("cannot create a window surface: there is no OpenGL context");
    return nullptr;
  }
  if (g_platform != WindowPlatform::Wayland) {
    // The reversed mirror of the X11 refusal: a wl_surface belongs to a
    // wl_display connection, and this context's EGL display draws windows only
    // on the connection it was built on.
    KNIZ_ERR("cannot create a Wayland window surface: the OpenGL context is "
             "not on the Wayland platform%s",
             g_platform == WindowPlatform::X11 ? " (it is on X11)" : "");
    return nullptr;
  }
  if (!g_window_capable) {
    KNIZ_ERR("cannot create a window surface: this OpenGL context has no "
             "config that supports windows");
    return nullptr;
  }
  if (!load_wayland_egl())
    return nullptr;

  // Wrap the application's wl_surface in a wl_egl_window of the swapchain's
  // extent.  The wrap is what EGL wants; the wl_surface itself is not a
  // drawable.
  void* wayland_window =
      g_wayland_egl.create(const_cast<void*>(wl_surface), width, height);
  if (wayland_window == nullptr) {
    KNIZ_ERR("wl_egl_window_create failed");
    return nullptr;
  }

  // EGLNativeWindowType is an unsigned long in the X11 flavoured headers this
  // build resolves against; a pointer passes through it unchanged on every
  // platform this driver targets, and the Wayland EGL display is defined to
  // read it as a wl_egl_window*.
  const EGLint attribs[] = {EGL_NONE};
  EGLSurface egl_surface = eglCreateWindowSurface(
      g_display, g_config,
      reinterpret_cast<EGLNativeWindowType>(wayland_window), attribs);
  if (egl_surface == EGL_NO_SURFACE) {
    KNIZ_ERR("eglCreateWindowSurface failed for the wl_egl_window (0x%04x)",
             eglGetError());
    g_wayland_egl.destroy(wayland_window);
    return nullptr;
  }
  g_wayland_pairings.push_back({egl_surface, wayland_window});

  const WindowSurface handle = reinterpret_cast<WindowSurface>(egl_surface);
  if (bind_window_surface(handle)) {
    // Interval 0, deliberately: swap interval 1 waits for the compositor's
    // frame callback, which a surface that is not yet mapped may never issue,
    // and FIFO presentation here means "the compositor decides the pace",
    // which on Wayland the compositor does regardless of the interval.  The
    // interval is what makes X11 eglSwapBuffers wait; dropping it does not
    // make Wayland presents tear, since on Wayland the buffer the compositor
    // is showing is never the one the application is drawing into.
    eglSwapInterval(g_display, 0);
    bind_headless_surface();
  }

  KNIZ_DBG("window surface created for wl_surface %p at %dx%d", wl_surface,
           width, height);
  return handle;
}

void window_surface_resize_wayland(WindowSurface surface, int width,
                                   int height) {
  if (surface == nullptr || !load_wayland_egl())
    return;
  if (WindowPairing* pairing = find_pairing(surface)) {
    g_wayland_egl.resize(pairing->wayland_window, width, height, 0, 0);
  }
}

bool create_headless() {
  if (g_attempted)
    return g_ok;
  g_attempted = true;

  // Two ways to get a context with no window, tried in order.
  //
  //   1. The X11 platform, when DISPLAY is set.  This is what an ordinary
  //      desktop session gives us, and it is the only platform presentation
  //      works on so far.
  //   2. The default display with a pbuffer.  Whatever platform the
  //      implementation prefers, used for drawing when X11 was not available.
  //   3. The surfaceless platform.  This is the one that works in containers,
  //      CI runners and any other place with no display at all, and it is why
  //      the backend does not simply require a display to succeed.
  //
  // The context is headless in all three cases: it draws to a pbuffer or to
  // nothing, and a window is only ever made current on it at present time
  // (§3.13).
  //
  // The platform is chosen explicitly rather than left to Mesa's preference,
  // which is derived from the environment: with both WAYLAND_DISPLAY and
  // DISPLAY set -- the normal shape of a Wayland desktop running XWayland -- it
  // picks Wayland, and then an X11 window handle cannot be presented to at all.
  // Which platform the headless context starts on is a starting choice, not a
  // commitment: X11 is asked for first because it is reachable on both an X11
  // desktop and a Wayland one (through XWayland), and Wayland surfaces move
  // the context over when the application asks for them (§3.15).  Platforms
  // whose extension is missing are reported rather than worked around.
#if defined(EGL_PLATFORM_X11_KHR)
  if (std::getenv("DISPLAY") != nullptr &&
      has_client_extension("EGL_EXT_platform_x11")) {
    EGLDisplay display = eglGetPlatformDisplay(
        EGL_PLATFORM_X11_KHR, EGL_DEFAULT_DISPLAY, nullptr);
    if (bring_up(display, true, "X11 platform")) {
      if (finish_setup(WindowPlatform::X11))
        return true;
      KNIZ_ERR("OpenGL context came up but could not be used");
      release();
      return false;
    }
  }
#endif

  if (bring_up(eglGetDisplay(EGL_DEFAULT_DISPLAY), true, "default display")) {
    // EGL_DEFAULT_DISPLAY leaves the choice to the implementation, so the
    // platform is identified afterwards rather than assumed.
    if (finish_setup(WindowPlatform::Other))
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
    if (finish_setup(WindowPlatform::None))
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
  //
  // The surface type asks for window surfaces as well when there is a display
  // that might have them, because a swapchain needs the same config the context
  // was created with -- EGL will not make a context current with a drawable from
  // a different config.  A surfaceless display matches no such config at all, so
  // that attempt is allowed to fail and the pbuffer-only request follows it.
  EGLint config_attribs[16];
  int n = 0;
  config_attribs[n++] = EGL_SURFACE_TYPE;
  config_attribs[n++] =
      wants_surface ? (EGL_PBUFFER_BIT | EGL_WINDOW_BIT) : EGL_PBUFFER_BIT;
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
  bool chose =
      eglChooseConfig(display, config_attribs, &config, 1, &num_configs) ==
          EGL_TRUE &&
      num_configs >= 1;
  if (!chose && wants_surface) {
    KNIZ_DBG("%s: no config supports window surfaces, asking for a pbuffer only",
             label);
    config_attribs[1] = EGL_PBUFFER_BIT;
    chose = eglChooseConfig(display, config_attribs, &config, 1, &num_configs) ==
                EGL_TRUE &&
            num_configs >= 1;
  }
  if (!chose) {
    KNIZ_DBG("%s: no matching EGL config (0x%04x)", label, eglGetError());
    eglTerminate(display);
    return false;
  }

  // Whether this config can back a window.  Recorded rather than assumed: a
  // pbuffer-only config works for everything except presentation, and knowing
  // which one we got is what lets a swapchain be refused with a reason instead
  // of failing at eglCreateWindowSurface later.
  EGLint surface_type = 0;
  const bool window_capable =
      eglGetConfigAttrib(display, config, EGL_SURFACE_TYPE, &surface_type) ==
          EGL_TRUE &&
      (surface_type & EGL_WINDOW_BIT) != 0;

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
      g_window_capable = window_capable;
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
  // The wl_egl_windows that belong to Wayland window surfaces are destroyed
  // here as well: release() means the whole context is going away, and every
  // EGL surface in the pairings dies with its display.  Doing this before
  // eglTerminate keeps the order -- EGL surface first, wl_egl_window after --
  // that also holds when surfaces are destroyed one at a time.
  for (const WindowPairing& pairing : g_wayland_pairings) {
    if (pairing.wayland_window != nullptr && g_wayland_egl.destroy != nullptr)
      g_wayland_egl.destroy(pairing.wayland_window);
  }
  g_wayland_pairings.clear();

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
  g_window_capable = false;
  g_platform = WindowPlatform::None;
  g_wayland_display = nullptr;
}

}  // namespace

void destroy() {
  release();
  g_info = ContextInfo{};
  g_ok = false;
  g_attempted = false;
  f = Functions{};
}

// ---------------------------------------------------------------------------
// Bringing the context up on an application's Wayland connection
// ---------------------------------------------------------------------------

bool wayland_context_matches(const void* wl_display) {
  return g_display != EGL_NO_DISPLAY && g_platform == WindowPlatform::Wayland &&
         g_wayland_display == wl_display;
}

bool create_on_wayland(const void* wl_display) {
  if (wl_display == nullptr) {
    KNIZ_ERR("a Wayland surface needs both a wl_display and a wl_surface");
    return false;
  }
#if defined(EGL_PLATFORM_WAYLAND_KHR)
  if (g_display != EGL_NO_DISPLAY) {
    KNIZ_ERR("the OpenGL context is already up; a Wayland surface the context "
             "was not built on needs migrate_to_wayland, and that is only "
             "possible before anything has been drawn");
    return false;
  }
  if (!has_client_extension("EGL_EXT_platform_wayland")) {
    KNIZ_ERR("this EGL has no Wayland platform support (EGL_EXT_platform_"
             "wayland absent), so a wl_display cannot become an EGL display");
    return false;
  }
  if (!load_wayland_egl())
    return false;

  // The application's wl_display is passed straight through: libEGL expects
  // the client's own display pointer for the Wayland platform.
  const EGLDisplay display = eglGetPlatformDisplay(
      EGL_PLATFORM_WAYLAND_KHR, const_cast<void*>(wl_display), nullptr);
  if (bring_up(display, true, "Wayland platform")) {
    if (finish_setup(WindowPlatform::Wayland)) {
      g_wayland_display = wl_display;
      KNIZ_DBG("OpenGL context is up on the application's wl_display (Wayland "
               "platform)");
      return true;
    }
    KNIZ_ERR("OpenGL context came up but could not be used (Wayland platform)");
    release();
    return false;
  }
  return false;
#else
  KNIZ_ERR("built against headers without EGL_PLATFORM_WAYLAND_KHR; Wayland "
           "surfaces are not available in this build");
  return false;
#endif
}

bool migrate_to_wayland(const void* wl_display) {
  // Deliberately the same entry point nothing else may call: migration is the
  // one sanctioned way to move the context between window systems, and it is
  // only reachable while no live window surface depends on the current
  // platform, which the caller checks with can_migrate_to_wayland.
  release();
  if (create_on_wayland(wl_display)) {
    KNIZ_DBG("the OpenGL context was rebuilt on the application's wl_display");
    return true;
  }
  return false;
}

bool can_migrate_to_wayland(const void* wl_display) {
  // With a live window surface of some other platform the move strands its
  // drawable, so it is not a mere rebuild of an idle context any more.  The
  // tally counts X11 drawables; Wayland pairings only exist for the display
  // the context currently serves, so their presence alone bars any move.  The
  // wl_display argument is part of the question for symmetry with the other
  // Wayland entry points, though its answer is already decided here.
  (void)wl_display;
  return g_x11_windows == 0 && g_wayland_pairings.empty();
}


namespace {

// Fills in the entry point table and the context description once a context is
// current.  Kept separate from bring_up so both display paths share it.
bool finish_setup(WindowPlatform platform) {
  g_platform = platform == WindowPlatform::Other
                   ? platform_of_display(g_display)
                   : platform;

  if (!load_proc(&f.GetString, sizeof(f.GetString), "glGetString") ||
      !load_proc(&f.GetIntegerv, sizeof(f.GetIntegerv), "glGetIntegerv") ||
      !load_proc(&f.GetError, sizeof(f.GetError), "glGetError"))
    return false;

  if (!query_context_info())
    return false;

  if (f.GetError && f.GetError() != kNoError)
    KNIZ_WARN("GL error set while querying context info");

  // Compute shaders and shader storage blocks became core in 4.3.  The version
  // string is the authority here, not the presence of the entry points: a
  // driver will happily hand out glDispatchCompute on a context that is not
  // allowed to call it.
  const bool version_supports_compute =
      g_info.major > 4 || (g_info.major == 4 && g_info.minor >= 3);

  const bool bytes_ok = load_all(kBufferProcs);
  const bool compute_ok = load_all(kComputeProcs);
  const bool textures_ok = load_all(kTextureProcs);
  const bool samplers_ok = load_all(kSamplerProcs);
  const bool rendering_ok = load_all(kRenderProcs);
  g_info.compute_capable = version_supports_compute && bytes_ok && compute_ok;

  // glTexStorage2D is 4.2, glClearTexImage 4.4 and glGetTexImage only returned
  // to the core profile in 4.5, so 4.5 is the floor for the texture set.
  const bool version_supports_textures =
      g_info.major > 4 || (g_info.major == 4 && g_info.minor >= 5);
  g_info.texture_capable = version_supports_textures && textures_ok;

  // Framebuffer objects and vertex array objects are core in 3.0, but the draw
  // path also uses glDrawElementsBaseVertex, which arrived in 3.2, because
  // vkCmdDrawIndexed's base vertex parameter has no earlier equivalent.  The
  // floor is 3.2 for that reason rather than for the framebuffer objects.
  const bool version_supports_rendering =
      g_info.major > 3 || (g_info.major == 3 && g_info.minor >= 2);
  g_info.render_capable = version_supports_rendering && rendering_ok;

  // Uniform buffer objects became core in 3.1.  They need no entry point that
  // the buffer set does not already resolve: glBindBufferRange arrived with the
  // rest of them, because storage buffer objects were modelled on uniform ones.
  const bool version_supports_uniforms =
      g_info.major > 3 || (g_info.major == 3 && g_info.minor >= 1);
  g_info.uniform_capable = version_supports_uniforms && bytes_ok;

  // Sampler objects became core in 3.3.  They are what lets one image be read
  // through two different filter and wrap settings, so without them a Vulkan
  // sampler has nowhere to live and sampling is refused rather than made to
  // depend on the image's own state.
  const bool version_supports_samplers =
      g_info.major > 3 || (g_info.major == 3 && g_info.minor >= 3);
  g_info.sampler_capable = version_supports_samplers && samplers_ok;

  if (g_info.sampler_capable) {
    gl::Int units = 0;
    f.GetIntegerv(static_cast<Enum>(kMaxCombinedTextureImageUnits), &units);
    g_info.max_texture_units = units;
  }

  if (!bytes_ok) {
    KNIZ_ERR("this OpenGL context is missing buffer or program entry points; "
             "Kniz cannot work with it");
    return false;
  }

  if (!g_info.texture_capable) {
    KNIZ_INFO("images are unavailable: textures need OpenGL 4.5 and this is a "
              "%d.%d context%s; buffers and compute still work",
              g_info.major, g_info.minor,
              textures_ok ? "" : " without the required entry points");
  }

  if (!g_info.compute_capable) {
    KNIZ_INFO("compute is unavailable: OpenGL %d.%d context, and compute "
              "shaders need 4.3; buffers will still work",
              g_info.major, g_info.minor);
  }

  if (g_info.compute_capable) {
    gl::Int alignment = 0;
    f.GetIntegerv(static_cast<Enum>(kGlShaderStorageBufferOffsetAlignment),
                  &alignment);
    g_info.ssbo_offset_alignment = alignment;

    // Vulkan caps minStorageBufferOffsetAlignment at 256 and so does OpenGL, so
    // a driver that wants more than 256 leaves some legal Vulkan offsets
    // unrepresentable.  Say so once, here, rather than once per failed bind.
    if (alignment > 256) {
      KNIZ_WARN(
          "this driver requires shader storage buffer offsets to be a multiple "
          "of %d, but Vulkan caps minStorageBufferOffsetAlignment at 256; "
          "offsets in between cannot be expressed and will be refused",
          alignment);
    }
  }

  if (g_info.uniform_capable) {
    gl::Int alignment = 0;
    f.GetIntegerv(static_cast<Enum>(kGlUniformBufferOffsetAlignment), &alignment);
    g_info.ubo_offset_alignment = alignment;

    // The same reasoning as above, with the same cap: Kniz reports 256 for
    // minUniformBufferOffsetAlignment, so a driver wanting more is the one case
    // where a legal Vulkan offset has no GL equivalent.
    if (alignment > 256) {
      KNIZ_WARN(
          "this driver requires uniform buffer offsets to be a multiple of %d, "
          "but Vulkan caps minUniformBufferOffsetAlignment at 256; offsets in "
          "between cannot be expressed and will be refused",
          alignment);
    }
  }

  g_info.valid = true;
  g_ok = true;

  KNIZ_INFO("OpenGL %d.%d %s context%s: %s / %s", g_info.major, g_info.minor,
            g_info.core ? "core" : "compatibility",
            g_surfaceless ? " (surfaceless)" : "", g_info.renderer.c_str(),
            g_info.version.c_str());
  if (g_info.compute_capable)
    KNIZ_INFO("compute shaders available; shader storage buffer offset "
              "alignment %d",
              g_info.ssbo_offset_alignment);
  if (g_info.uniform_capable)
    KNIZ_INFO("uniform buffers available; uniform buffer offset alignment %d",
              g_info.ubo_offset_alignment);
  else
    KNIZ_INFO("uniform buffers are unavailable: they need OpenGL 3.1 and this "
              "is a %d.%d context",
              g_info.major, g_info.minor);
  if (g_info.texture_capable)
    KNIZ_INFO("images available: textures, with glTexStorage2D, glClearTexImage "
              "and glGetTexImage");
  if (g_info.sampler_capable)
    KNIZ_INFO("samplers available: sampler objects, %d texture unit(s)",
              g_info.max_texture_units);

  else
    KNIZ_INFO("samplers are unavailable: sampler objects need OpenGL 3.3 and "
              "this is a %d.%d context",
              g_info.major, g_info.minor);

  // A window surface needs both a config that supports windows and a display
  // whose platform the surface handle belongs to.  Which platforms count is
  // what the swapchain layer answers for: X11 windows on the X11 platform,
  // Wayland windows on the Wayland platform (§3.15).
  g_info.window_capable = g_window_capable && g_info.render_capable &&
                          (g_platform == WindowPlatform::X11 ||
                           g_platform == WindowPlatform::Wayland);
  if (g_info.window_capable)
    KNIZ_INFO("presentation available: %s windows, through EGL on the %s "
              "platform",
              g_platform == WindowPlatform::X11 ? "X11" : "Wayland",
              g_platform == WindowPlatform::X11 ? "X11" : "Wayland");
  else
    KNIZ_INFO("presentation is unavailable: %s",
              g_platform == WindowPlatform::Wayland
                  ? "the context is on Wayland, and this EGL cannot make "
                    "window surfaces there"
                  : g_platform == WindowPlatform::None
                        ? "there is no window system (a surfaceless context)"
                        : "this EGL display cannot create a window surface");
  if (g_info.render_capable)
    KNIZ_INFO("rendering available: framebuffer objects, vertex array objects, "
              "draws");
  return true;
}

}  // namespace

}  // namespace kniz::gl
