// M3c test: presentation on Wayland, through the real loader and a real
// compositor.
//
// This is the present test (M3) over again with Wayland where X11 stood.  That
// is deliberate: the present path's pixel-level guarantees -- the y flip, the
// sampler, the clean GL state before every present draw -- live in one piece of
// code that both platforms share, so re-running the same battery on the Wayland
// side of the surface seam is the honest way to check that the seam does not
// change what lands on screen.
//
// Three things differ, and each is part of what is being tested:
//
//   * The surface comes from the application's own wl_display and wl_surface,
//     and the driver's GL context must be rebuilt on that display before the
//     first draw.  The build does not do it at vkCreateWaylandSurfaceKHR --
//     the loader stops there and defers the ICD's first contact until the
//     first WSI query, which the test reaches after vkCreateDevice precisely
//     because real applications do (see docs/DESIGN.md §3.15).  That is also
//     why the test creates the surface between vkCreateInstance and
//     vkCreateDevice rather than going through the shared harness, which does
//     both in one call: the app's wl_surface must exist for the first WSI
//     query to find.
//
//   * The swapchain's wl_egl_window is born with a size, so a smaller second
//     swapchain on the same surface exercises wl_egl_window_resize rather than
//     the X11 path's fixed window size.  The presented frame then has the
//     swapchain's extent, which is what the Wayland backend defines as the
//     drawable's size.
//
//   * The window is mapped.  The X11 test's window is deliberately unmapped so
//     that no window manager resizes it mid-test; Wayland's compositor deals
//     with surfaces one level further away (it scales the buffer to whatever
//     space it assigned, and the presented frame is captured before
//     eglSwapBuffers hands the buffer over), so mapping is safe here and it is
//     what makes the buffer release flow complete.
//
// The checks are the present test's, in the same order: extensions advertised,
// surface queries honest, the refusal battery, a swapchain whose images hold
// the triangle the vertex data describes, the presented frame equal to the
// image pixel for pixel, round-robin acquisition, an out-of-range present
// refused, and a smaller swapchain scaled to its drawable.
//
// On a machine with no Wayland compositor (a CI runner, for one) there is
// nothing to connect to, and the test says so and succeeds -- the Wayland
// suite, like the X11 one, must skip rather than fail where it cannot run.

#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <unistd.h>

#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "harness.h"

using namespace kniz_test;

namespace {

// The window, and the swapchain that matches it.  Same sizes and colours as the
// X11 present test, on purpose: one set of expectations, two platforms.
constexpr uint32_t kWidth = 320;
constexpr uint32_t kHeight = 240;

constexpr uint32_t kSmallWidth = 160;
constexpr uint32_t kSmallHeight = 120;

constexpr uint32_t kBytesPerPixel = 4;

constexpr uint8_t kClearBytes[4] = {17, 34, 51, 255};
constexpr uint8_t kSmallClearBytes[4] = {200, 100, 50, 255};

VkClearValue clear_value(const uint8_t bytes[4]) {
  VkClearValue value{};
  value.color.float32[0] = bytes[0] / 255.0f;
  value.color.float32[1] = bytes[1] / 255.0f;
  value.color.float32[2] = bytes[2] / 255.0f;
  value.color.float32[3] = bytes[3] / 255.0f;
  return value;
}

// The same triangle the M2b and M3 tests draw: three vertices carrying red,
// green and blue, with the apex opposite the flat edge so that a y flip cannot
// survive comparison.
struct Vertex {
  float position[2];
  float color[4];
};

constexpr Vertex kVertices[3] = {
    {{-1.0f, -1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{1.0f, -1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
    {{0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},
};

constexpr uint32_t kVertexStride = sizeof(Vertex);

void barycentric(double px, double py, double* w0, double* w1, double* w2) {
  const double x0 = (kVertices[0].position[0] + 1.0) * 0.5 * kWidth;
  const double y0 = (kVertices[0].position[1] + 1.0) * 0.5 * kHeight;
  const double x1 = (kVertices[1].position[0] + 1.0) * 0.5 * kWidth;
  const double y1 = (kVertices[1].position[1] + 1.0) * 0.5 * kHeight;
  const double x2 = (kVertices[2].position[0] + 1.0) * 0.5 * kWidth;
  const double y2 = (kVertices[2].position[1] + 1.0) * 0.5 * kHeight;

  const double denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
  *w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denominator;
  *w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denominator;
  *w2 = 1.0 - *w0 - *w1;
}

bool inside_triangle(uint32_t x, uint32_t y) {
  double w0 = 0, w1 = 0, w2 = 0;
  barycentric(x + 0.5, y + 0.5, &w0, &w1, &w2);
  constexpr double kSlack = 1e-9;
  return w0 >= -kSlack && w1 >= -kSlack && w2 >= -kSlack;
}

void expected_triangle_color(uint32_t x, uint32_t y, uint8_t out[4]) {
  if (!inside_triangle(x, y)) {
    std::memcpy(out, kClearBytes, 4);
    return;
  }
  double w0 = 0, w1 = 0, w2 = 0;
  barycentric(x + 0.5, y + 0.5, &w0, &w1, &w2);
  for (int channel = 0; channel < 4; ++channel) {
    const double value = w0 * kVertices[0].color[channel] +
                         w1 * kVertices[1].color[channel] +
                         w2 * kVertices[2].color[channel];
    out[channel] = static_cast<uint8_t>(std::lround(value * 255.0));
  }
}

constexpr int kTolerance = 3;

struct Image {
  const uint8_t* data = nullptr;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t pixel_bytes = 4;
  const char* label = "";
};

int compare_with_triangle(const Image& image, std::string* first_difference) {
  int mismatches = 0;
  for (uint32_t y = 0; y < image.height; ++y) {
    for (uint32_t x = 0; x < image.width; ++x) {
      uint8_t expected[4] = {0, 0, 0, 0};
      expected_triangle_color(x, y, expected);
      const uint8_t* actual =
          image.data + (static_cast<size_t>(y) * image.width + x) * image.pixel_bytes;
      bool differs = false;
      for (uint32_t channel = 0; channel < 3; ++channel) {
        if (std::abs(static_cast<int>(actual[channel]) -
                     static_cast<int>(expected[channel])) > kTolerance) {
          differs = true;
        }
      }
      if (!differs)
        continue;
      if (mismatches == 0 && first_difference != nullptr) {
        char text[256];
        std::snprintf(text, sizeof text,
                      "%s: (%u, %u) is (%u, %u, %u), expected (%u, %u, %u)",
                      image.label, x, y, actual[0], actual[1], actual[2],
                      expected[0], expected[1], expected[2]);
        *first_difference = text;
      }
      ++mismatches;
    }
  }
  return mismatches;
}

struct Ppm {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<uint8_t> rgb;
};

bool read_ppm(const char* path, Ppm* out) {
  std::FILE* file = std::fopen(path, "rb");
  if (file == nullptr)
    return false;

  char magic[3] = {};
  int width = 0;
  int height = 0;
  int maximum = 0;
  const bool header =
      std::fscanf(file, "%2s %d %d %d", magic, &width, &height, &maximum) == 4;
  const int separator = std::fgetc(file);
  const bool ok = header && std::strcmp(magic, "P6") == 0 && width > 0 &&
                  height > 0 && maximum == 255 && separator == '\n';
  if (!ok) {
    std::fclose(file);
    return false;
  }

  out->width = static_cast<uint32_t>(width);
  out->height = static_cast<uint32_t>(height);
  out->rgb.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
  const size_t read = std::fread(out->rgb.data(), 1, out->rgb.size(), file);
  std::fclose(file);
  return read == out->rgb.size();
}

// ---------------------------------------------------------------------------
// A Wayland client without Wayland headers.
//
// libwayland-client is opened with dlopen, like libX11 in the present test, so
// the suite builds with nothing but Vulkan headers.  The core protocol's
// proxies, interfaces and opcodes come out of the library's own data symbols
// and exported functions.  xdg-shell is not part of the library, so the three
// interfaces the test needs are declared by hand: name, version and the wire
// signatures of exactly the requests and events that get used, with `types`
// arrays left null (libwayland accepts null type pointers; they only affect
// validation strictness, not the wire bytes).
//
// The window is mapped with xdg-shell rather than left role-less on purpose: a
// commit on a role-less surface is legal, but what the compositor does with it
// afterwards varies, and Mesa's Wayland platform waits for buffer release to
// recycle its pool.  A mapped surface makes the whole flow boring in every
// compositor, which is what a gate wants.
// ---------------------------------------------------------------------------

// ABI shapes of libwayland's interface and message records, stable for as long
// as the library has exported them.  Declared locally rather than by including
// <wayland-util.h>, so libwayland stays a dlopened dependency.
struct FakeMessage {
  const char* name;
  const char* signature;
  const void** types;  // wl_interface* per argument, null everywhere here
};

struct FakeInterface {
  const char* name;
  int version;
  int method_count;
  const FakeMessage* methods;
  int event_count;
  const FakeMessage* events;
};

// xdg_wm_base: destroy(0), create_positioner(1), get_xdg_surface(2), pong(3);
// event ping(0).  The request order is the order in the protocol XML, and
// xdg_wm_base is one of those interfaces whose second request is easy to
// assume wrong: create_positioner really is opcode 1, get_xdg_surface is 2.
const FakeMessage kXdgWmBaseMethods[] = {
    {"destroy", "", nullptr},
    {"create_positioner", "n", nullptr},
    {"get_xdg_surface", "no", nullptr},
    {"pong", "u", nullptr},
};
const FakeMessage kXdgWmBaseEvents[] = {
    {"ping", "u", nullptr},
};
const FakeInterface kXdgWmBaseInterface = {"xdg_wm_base", 1, 4, kXdgWmBaseMethods, 1, kXdgWmBaseEvents};

// xdg_surface: destroy(0), get_toplevel(1), get_popup(2), set_window_geometry(3),
// ack_configure(4); events configure(0), close(1).
const FakeMessage kXdgSurfaceMethods[] = {
    {"destroy", "", nullptr},
    {"get_toplevel", "n", nullptr},
    {"get_popup", "noo", nullptr},
    {"set_window_geometry", "iiii", nullptr},
    {"ack_configure", "u", nullptr},
};
const FakeMessage kXdgSurfaceEvents[] = {
    {"configure", "u", nullptr},
    {"close", "", nullptr},
};
const FakeInterface kXdgSurfaceInterface = {"xdg_surface", 1, 5, kXdgSurfaceMethods, 2, kXdgSurfaceEvents};

// xdg_toplevel: the requests are unused; the events are what the compositor
// sends, and as of xdg-shell v7 there are four of them, so all four are
// declared: configure(0), close(1), configure_bounds(2), wm_capabilities(3).
// Declaring only the first two would make the compositor's initial burst on a
// v7 surface a proxy error.
const FakeMessage kXdgToplevelEvents[] = {
    {"configure", "iia", nullptr},
    {"close", "", nullptr},
    {"configure_bounds", "ii", nullptr},
    {"wm_capabilities", "a", nullptr},
};
const FakeInterface kXdgToplevelInterface = {"xdg_toplevel", 1, 0, nullptr, 4, kXdgToplevelEvents};

struct Wayland {
  void* library = nullptr;
  struct wl_client_state {
    void* display = nullptr;    // wl_display*
    void* surface = nullptr;    // wl_surface*
    void* xdg_surface = nullptr;
    void* xdg_toplevel = nullptr;
  } client;

  // libwayland-client entry points.  Every one that data flows through.
  void* (*display_connect)(const char*) = nullptr;
  void (*display_disconnect)(void*) = nullptr;
  int (*display_roundtrip)(void*) = nullptr;
  uint32_t (*display_get_error)(void*) = nullptr;
  void* (*proxy_marshal_flags)(void*, uint32_t, const void*, uint32_t, uint32_t,
                               ...) = nullptr;
  void (*proxy_marshal)(void*, uint32_t, ...) = nullptr;
  int (*proxy_add_listener)(void*, void (**)(void), void*) = nullptr;
  void (*proxy_destroy)(void*) = nullptr;

  // The core interface records, dlsym'd out of the library.
  const void* display_interface = nullptr;
  const void* registry_interface = nullptr;
  const void* compositor_interface = nullptr;
  const void* surface_interface = nullptr;

  // The xdg-shell handshake state.
  void* wm_base = nullptr;
  uint32_t compositor_name = 0;
  uint32_t compositor_version = 0;
  uint32_t wm_base_name = 0;
  uint32_t wm_base_version = 0;
  bool configured = false;
};

// The registry listener, in the shape wl_proxy_add_listener expects (an array
// of function pointers, handed over as void (**)(void)).
void registry_global(void* data, void*, uint32_t name, const char* interface,
                     uint32_t version) {
  auto* wayland = static_cast<Wayland*>(data);
  if (std::strcmp(interface, "wl_compositor") == 0) {
    wayland->compositor_name = name;
    wayland->compositor_version = version;
  }
  if (std::strcmp(interface, "xdg_wm_base") == 0) {
    wayland->wm_base_name = name;
    wayland->wm_base_version = version;
  }
}

void registry_global_remove(void*, void*, uint32_t) {}

// The ping handler needs the Wayland struct, and wl_proxy_add_listener hands
// each listener its own data pointer -- so the pong keeps the connection
// struct in a file-static slot set when the listener is attached.
Wayland* wm_base_ping_state = nullptr;

void wm_base_ping(void*, void* wm_base, uint32_t serial) {
  auto* wayland = static_cast<Wayland*>(wm_base_ping_state);
  if (wayland->proxy_marshal != nullptr && wm_base != nullptr)
    wayland->proxy_marshal(wm_base, 2 /*pong*/, serial);
}

void surface_configure(void* data, void* xdg_surface, uint32_t serial) {
  auto* wayland = static_cast<Wayland*>(data);
  wayland->configured = true;
  if (wayland->proxy_marshal != nullptr && xdg_surface != nullptr) {
    // ack_configure is opcode 4; opcode 3 is set_window_geometry, and asking
    // for an ack with a uint lands on the compositor as a bogus, zero-sized
    // geometry request (a day-long puzzle: the compositor reports error 5,
    // invalid_size, naming a set_window_geometry nobody sent on purpose).
    wayland->proxy_marshal(xdg_surface, 4 /*ack_configure*/, serial);
    // The second commit is what turns the role into a mapped window.
    wayland->proxy_marshal(static_cast<void*>(wayland->client.surface), 6);
  }
}

void xdg_close(void*, void*, uint32_t) {}

bool open_wayland(Wayland* wayland) {
  wayland->library = dlopen("libwayland-client.so.0", RTLD_NOW | RTLD_LOCAL);
  if (wayland->library == nullptr)
    return false;

  const auto load = [wayland](const char* name) { return dlsym(wayland->library, name); };
  wayland->display_connect = to_fn<void* (*)(const char*)>(load("wl_display_connect"));
  wayland->display_disconnect =
      to_fn<void (*)(void*)>(load("wl_display_disconnect"));
  wayland->display_roundtrip =
      to_fn<int (*)(void*)>(load("wl_display_roundtrip"));
  wayland->display_get_error =
      to_fn<uint32_t (*)(void*)>(load("wl_display_get_error"));
  wayland->proxy_marshal_flags = to_fn<void* (*)(void*, uint32_t, const void*,
                                                 uint32_t, uint32_t, ...)>(load("wl_proxy_marshal_flags"));
  wayland->proxy_marshal =
      load("wl_proxy_marshal") != nullptr
          ? to_fn<void (*)(void*, uint32_t, ...)>(load("wl_proxy_marshal"))
          : nullptr;
  wayland->proxy_add_listener =
      to_fn<int (*)(void*, void (**)(void), void*)>(load("wl_proxy_add_listener"));
  wayland->proxy_destroy = to_fn<void (*)(void*)>(load("wl_proxy_destroy"));

  wayland->display_interface = load("wl_display_interface");
  wayland->registry_interface = load("wl_registry_interface");
  wayland->compositor_interface = load("wl_compositor_interface");
  wayland->surface_interface = load("wl_surface_interface");

  if (wayland->display_connect == nullptr || wayland->display_disconnect == nullptr ||
      wayland->display_roundtrip == nullptr || wayland->display_get_error == nullptr ||
      wayland->proxy_marshal_flags == nullptr || wayland->proxy_marshal == nullptr ||
      wayland->proxy_add_listener == nullptr || wayland->proxy_destroy == nullptr ||
      wayland->display_interface == nullptr || wayland->registry_interface == nullptr ||
      wayland->compositor_interface == nullptr ||
      wayland->surface_interface == nullptr) {
    // wl_proxy_marshal_flags arrived in libwayland-client 1.22; older stacks
    // can create no method-validated new_id without the protocol headers, and
    // the test would rather say so than hand-roll the wire.
    return false;
  }

  void* display = wayland->display_connect(nullptr);
  if (display == nullptr) {
    // No compositor on this machine: the normal case in a container (and on a
    // CI runner), the same shape of skip as the X11 test's missing display.
    return false;
  }
  wayland->client.display = display;

  // The registry: wl_display.get_registry (opcode 1) makes one.  Every new_id
  // argument consumes one vararg slot even though the proxy is created from
  // the interface parameter -- a null placeholder, matching what the protocol
  // codegen emits.  wl_registry.bind's wire order is the historical quirk:
  // usun, so the new_id placeholder is the LAST argument, not the first.
  void* registry = wayland->proxy_marshal_flags(
      display, 1, wayland->registry_interface, 1, 0, /*new_id*/ nullptr);
  if (registry == nullptr)
    return false;

  void (*registry_listener[2])(void) = {};
  registry_listener[0] = reinterpret_cast<void (*)(void)>(registry_global);
  registry_listener[1] = reinterpret_cast<void (*)(void)>(registry_global_remove);
  wayland->proxy_add_listener(registry, registry_listener, wayland);
  if (wayland->display_roundtrip(display) < 0)
    return false;

  if (wayland->compositor_version == 0 || wayland->wm_base_version == 0)
    return false;  // a compositor without xdg-shell: no window to present

  const auto error = [wayland, display]() {
    return wayland->display_get_error(display) != 0;
  };

  // wl_registry.bind (opcode 0), wire order usun: name, interface string,
  // version, then the new_id placeholder last.
  void* compositor = wayland->proxy_marshal_flags(
      registry, 0, wayland->compositor_interface, wayland->compositor_version, 0,
      wayland->compositor_name, "wl_compositor", wayland->compositor_version,
      /*new_id last*/ nullptr);
  void* wm_base = wayland->proxy_marshal_flags(
      registry, 0, &kXdgWmBaseInterface, wayland->wm_base_version, 0,
      wayland->wm_base_name, "xdg_wm_base", wayland->wm_base_version,
      /*new_id last*/ nullptr);
  if (compositor == nullptr || wm_base == nullptr || error())
    return false;
  void* surface = wayland->proxy_marshal_flags(
      compositor, 0 /*create_surface*/, wayland->surface_interface, 1, 0,
      /*new_id*/ nullptr);
  if (surface == nullptr || error())
    return false;
  wayland->client.surface = surface;

  // The xdg handshake: wm_base.get_xdg_surface (opcode 2), then
  // xdg_surface.get_toplevel (opcode 1), listeners, and a first commit
  // (empty) that asks for a configure.
  void* xdg_surface = wayland->proxy_marshal_flags(
      wm_base, 2 /*get_xdg_surface*/, &kXdgSurfaceInterface, 1, 0,
      /*new_id*/ nullptr, surface);
  if (xdg_surface == nullptr || error())
    return false;
  wayland->client.xdg_surface = xdg_surface;
  void* xdg_toplevel = wayland->proxy_marshal_flags(
      xdg_surface, 1 /*get_toplevel*/, &kXdgToplevelInterface, 1, 0,
      /*new_id*/ nullptr);
  if (xdg_toplevel == nullptr || error())
    return false;
  wayland->client.xdg_toplevel = xdg_toplevel;
  wayland->wm_base = wm_base;

  wm_base_ping_state = wayland;
  void (*wm_listener[1])(void) = {};
  wm_listener[0] = reinterpret_cast<void (*)(void)>(wm_base_ping);
  wayland->proxy_add_listener(wm_base, wm_listener, wayland);

  void (*xdg_listener[2])(void) = {};
  xdg_listener[0] = reinterpret_cast<void (*)(void)>(surface_configure);
  xdg_listener[1] = reinterpret_cast<void (*)(void)>(xdg_close);
  wayland->proxy_add_listener(xdg_surface, xdg_listener, wayland);

  wayland->proxy_marshal(surface, 6 /*wl_surface.commit*/);
  wayland->display_roundtrip(display);
  if (!wayland->configured || error())
    return false;  // the compositor never confirmed the role
  wayland->display_roundtrip(display);  // the map commit's result

  // The compositor may resize the toplevel when it places it (Hyprland tiles).
  // The test keeps its buffer at the size the swapchains will use and ignores
  // the size part of the configure: the compositor scales the buffer to the
  // space it assigned, and nothing the test compares is affected.
  return true;
}

void close_wayland(Wayland* wayland) {
  if (wayland->client.surface != nullptr && wayland->proxy_destroy != nullptr) {
    wayland->proxy_destroy(wayland->client.xdg_toplevel);
    wayland->proxy_destroy(wayland->client.xdg_surface);
    wayland->proxy_destroy(wayland->client.surface);
    wayland->proxy_destroy(wayland->wm_base);
    wayland->client.xdg_toplevel = nullptr;
    wayland->client.xdg_surface = nullptr;
    wayland->client.surface = nullptr;
    wayland->wm_base = nullptr;
  }
  if (wayland->client.display != nullptr) {
    wayland->display_disconnect(wayland->client.display);
    wayland->client.display = nullptr;
  }
  if (wayland->library != nullptr) {
    dlclose(wayland->library);
    wayland->library = nullptr;
  }
}

// What a deadlock would beat: a watchdog.  The Wayland side of the test has
// more synchronous waits (a compositor handshake, buffer release) than the X11
// one, and any of them going wrong would otherwise hang ctest for its whole
// timeout.  Fifteen seconds is far past anything a live test does and far short
// of a timeout that would stall a suite.
void watchdog_handler(int) {
  std::fprintf(stderr, "wayland test: the watchdog fired; aborting\n");
  _exit(1);
}

// VK_KHR_wayland_surface's create info, same reason as the xcb one in
// present.cpp: no platform header is included, the ABI is the extension's.
struct WaylandSurfaceCreateInfo {
  VkStructureType sType;
  const void* pNext;
  VkFlags flags;
  void* display;
  void* surface;
};

using CreateWaylandSurfaceFn = VkResult (*)(VkInstance,
                                            const WaylandSurfaceCreateInfo*,
                                            const VkAllocationCallbacks*,
                                            VkSurfaceKHR*);

VkResult create_wayland_surface(VkInstance instance, void* display,
                                void* surface, VkSurfaceKHR* out) {
  auto create =
      instance_proc<CreateWaylandSurfaceFn>(instance, "vkCreateWaylandSurfaceKHR");
  if (create == nullptr)
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  WaylandSurfaceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
  info.display = display;
  info.surface = surface;
  return create(instance, &info, nullptr, out);
}

std::vector<uint32_t> read_spirv(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::fprintf(stderr, "cannot open shader '%s'\n", path);
    return {};
  }
  const std::streamsize size = file.tellg();
  file.seekg(0);
  std::vector<uint32_t> words(static_cast<size_t>(size) / sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(words.data()), size);
  return words;
}

}  // namespace

// The Wayland test's own bring-up: instance, surface, device in that order.
// It is the shared harness's bring_up with the surface creation in the middle,
// because that is the order vkCreateWaylandSurfaceKHR's migration demands --
// and the order every real Wayland Vulkan application happens to use.
int main(int argc, char** argv) {
  std::signal(SIGALRM, watchdog_handler);
  alarm(15);

  std::printf("kniz wayland test (M3c)\n");

  const char* shader_dir = argc > 1 ? argv[1] : ".";
  const std::string triangle_vert = std::string(shader_dir) + "/triangle.vert.spv";
  const std::string triangle_frag = std::string(shader_dir) + "/triangle.frag.spv";

  if (!icd_selected())
    return 2;

  void* loader = open_loader();
  if (loader == nullptr)
    return 2;
  if (!resolve_get_instance_proc_addr(loader))
    return 2;

  Wayland wayland;
  if (!open_wayland(&wayland)) {
    // The surfaces of failure are all ambient: no library, no compositor, no
    // xdg-shell support.  None of them is the driver's doing, and the test is
    // a guest of whatever compositor the machine happens to run.
    const char* reason = "skipped: no Wayland display to present to";
    std::printf("  %-58s %s\n", reason, "ok");
    return 0;
  }

  auto create_instance =
      instance_proc<PFN_vkCreateInstance>(nullptr, "vkCreateInstance");

  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "kniz-wayland";
  app.apiVersion = VK_API_VERSION_1_0;

  const char* instance_extensions[] = {"VK_KHR_surface", "VK_KHR_wayland_surface"};
  VkInstanceCreateInfo instance_info{};
  instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_info.pApplicationInfo = &app;
  instance_info.enabledExtensionCount = 2;
  instance_info.ppEnabledExtensionNames = instance_extensions;

  Runtime runtime;
  if (create_instance(&instance_info, nullptr, &runtime.instance) != VK_SUCCESS) {
    std::fprintf(stderr, "vkCreateInstance failed\n");
    close_wayland(&wayland);
    return 1;
  }
  runtime.destroy_instance = instance_proc<PFN_vkDestroyInstance>(
      runtime.instance, "vkDestroyInstance");

  // The surface.  Its ICD half is created lazily, at the test's first WSI
  // query, which is where the GL context actually gets moved onto the
  // application's wl_display; the driver's own log says so there.  Creating it
  // before the device keeps the app's wl_surface in place for that query to
  // find.
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  if (create_wayland_surface(runtime.instance, wayland.client.display,
                             wayland.client.surface, &surface) != VK_SUCCESS ||
      surface == VK_NULL_HANDLE) {
    std::fprintf(stderr, "vkCreateWaylandSurfaceKHR failed\n");
    tear_down(&runtime);
    close_wayland(&wayland);
    return 1;
  }
  auto destroy_surface = instance_proc<PFN_vkDestroySurfaceKHR>(
      runtime.instance, "vkDestroySurfaceKHR");

  // Everything below is bring_up's second half, unchanged: the Kniz physical
  // device is picked by its inventoried vendor id, the host visible memory type
  // is found by asking, and the device carries only VK_KHR_swapchain.
  auto enumerate_physical = instance_proc<PFN_vkEnumeratePhysicalDevices>(
      runtime.instance, "vkEnumeratePhysicalDevices");
  auto get_properties = instance_proc<PFN_vkGetPhysicalDeviceProperties>(
      runtime.instance, "vkGetPhysicalDeviceProperties");
  auto get_memory_properties =
      instance_proc<PFN_vkGetPhysicalDeviceMemoryProperties>(
          runtime.instance, "vkGetPhysicalDeviceMemoryProperties");
  auto create_device_fn =
      instance_proc<PFN_vkCreateDevice>(runtime.instance, "vkCreateDevice");

  get_device_proc_addr = instance_proc<PFN_vkGetDeviceProcAddr>(
      runtime.instance, "vkGetDeviceProcAddr");

  uint32_t device_count = 0;
  enumerate_physical(runtime.instance, &device_count, nullptr);
  std::vector<VkPhysicalDevice> devices(device_count);
  enumerate_physical(runtime.instance, &device_count, devices.data());
  for (VkPhysicalDevice candidate : devices) {
    VkPhysicalDeviceProperties properties{};
    get_properties(candidate, &properties);
    if (properties.vendorID == kKnizVendorId) {
      runtime.physical = candidate;
      std::printf("     device     : %s\n", properties.deviceName);
      break;
    }
  }
  check(runtime.physical != VK_NULL_HANDLE, "a Kniz physical device is present");
  if (runtime.physical == VK_NULL_HANDLE || failures() > 0) {
    destroy_surface(runtime.instance, surface, nullptr);
    tear_down(&runtime);
    close_wayland(&wayland);
    return 1;
  }

  get_memory_properties(runtime.physical, &runtime.memory);
  for (uint32_t i = 0; i < runtime.memory.memoryTypeCount; ++i) {
    if ((runtime.memory.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
      runtime.host_memory_type = i;
      break;
    }
  }

  float priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = 0;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;

  const char* device_extensions[] = {"VK_KHR_swapchain"};
  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.enabledExtensionCount = 1;
  device_info.ppEnabledExtensionNames = device_extensions;

  if (create_device_fn(runtime.physical, &device_info, nullptr,
                       &runtime.device) != VK_SUCCESS) {
    std::fprintf(stderr, "vkCreateDevice failed\n");
    destroy_surface(runtime.instance, surface, nullptr);
    tear_down(&runtime);
    close_wayland(&wayland);
    return 1;
  }
  runtime.destroy_device =
      device_proc<PFN_vkDestroyDevice>(runtime.device, "vkDestroyDevice");
  runtime.device_wait_idle =
      device_proc<PFN_vkDeviceWaitIdle>(runtime.device, "vkDeviceWaitIdle");
  auto get_queue =
      device_proc<PFN_vkGetDeviceQueue>(runtime.device, "vkGetDeviceQueue");
  get_queue(runtime.device, 0, 0, &runtime.queue);

  VkInstance instance = runtime.instance;
  VkDevice device = runtime.device;
  VkQueue queue = runtime.queue;

  // ---- entry points -------------------------------------------------------
  auto enumerate_instance_extensions =
      instance_proc<PFN_vkEnumerateInstanceExtensionProperties>(
          nullptr, "vkEnumerateInstanceExtensionProperties");
  auto enumerate_device_extensions =
      instance_proc<PFN_vkEnumerateDeviceExtensionProperties>(
          instance, "vkEnumerateDeviceExtensionProperties");
  auto get_surface_support = instance_proc<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
      instance, "vkGetPhysicalDeviceSurfaceSupportKHR");
  auto get_surface_capabilities =
      instance_proc<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(
          instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
  auto get_surface_formats = instance_proc<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(
      instance, "vkGetPhysicalDeviceSurfaceFormatsKHR");
  auto get_surface_present_modes =
      instance_proc<PFN_vkGetPhysicalDeviceSurfacePresentModesKHR>(
          instance, "vkGetPhysicalDeviceSurfacePresentModesKHR");

  auto create_swapchain =
      device_proc<PFN_vkCreateSwapchainKHR>(device, "vkCreateSwapchainKHR");
  auto destroy_swapchain =
      device_proc<PFN_vkDestroySwapchainKHR>(device, "vkDestroySwapchainKHR");
  auto get_swapchain_images = device_proc<PFN_vkGetSwapchainImagesKHR>(
      device, "vkGetSwapchainImagesKHR");
  auto acquire_next_image = device_proc<PFN_vkAcquireNextImageKHR>(
      device, "vkAcquireNextImageKHR");
  auto queue_present =
      device_proc<PFN_vkQueuePresentKHR>(device, "vkQueuePresentKHR");

  auto create_image_view =
      device_proc<PFN_vkCreateImageView>(device, "vkCreateImageView");
  auto destroy_image_view =
      device_proc<PFN_vkDestroyImageView>(device, "vkDestroyImageView");
  auto create_render_pass =
      device_proc<PFN_vkCreateRenderPass>(device, "vkCreateRenderPass");
  auto destroy_render_pass =
      device_proc<PFN_vkDestroyRenderPass>(device, "vkDestroyRenderPass");
  auto create_framebuffer =
      device_proc<PFN_vkCreateFramebuffer>(device, "vkCreateFramebuffer");
  auto destroy_framebuffer =
      device_proc<PFN_vkDestroyFramebuffer>(device, "vkDestroyFramebuffer");
  auto create_shader_module =
      device_proc<PFN_vkCreateShaderModule>(device, "vkCreateShaderModule");
  auto destroy_shader_module =
      device_proc<PFN_vkDestroyShaderModule>(device, "vkDestroyShaderModule");
  auto create_pipeline_layout = device_proc<PFN_vkCreatePipelineLayout>(
      device, "vkCreatePipelineLayout");
  auto destroy_pipeline_layout = device_proc<PFN_vkDestroyPipelineLayout>(
      device, "vkDestroyPipelineLayout");
  auto create_graphics_pipelines =
      device_proc<PFN_vkCreateGraphicsPipelines>(device, "vkCreateGraphicsPipelines");
  auto destroy_pipeline =
      device_proc<PFN_vkDestroyPipeline>(device, "vkDestroyPipeline");

  auto create_buffer = device_proc<PFN_vkCreateBuffer>(device, "vkCreateBuffer");
  auto destroy_buffer = device_proc<PFN_vkDestroyBuffer>(device, "vkDestroyBuffer");
  auto get_buffer_requirements = device_proc<PFN_vkGetBufferMemoryRequirements>(
      device, "vkGetBufferMemoryRequirements");
  auto bind_buffer_memory =
      device_proc<PFN_vkBindBufferMemory>(device, "vkBindBufferMemory");
  auto allocate_memory = device_proc<PFN_vkAllocateMemory>(device, "vkAllocateMemory");
  auto free_memory = device_proc<PFN_vkFreeMemory>(device, "vkFreeMemory");
  auto map_memory = device_proc<PFN_vkMapMemory>(device, "vkMapMemory");

  auto create_command_pool =
      device_proc<PFN_vkCreateCommandPool>(device, "vkCreateCommandPool");
  auto destroy_command_pool =
      device_proc<PFN_vkDestroyCommandPool>(device, "vkDestroyCommandPool");
  auto allocate_command_buffers =
      device_proc<PFN_vkAllocateCommandBuffers>(device, "vkAllocateCommandBuffers");
  auto begin_command_buffer =
      device_proc<PFN_vkBeginCommandBuffer>(device, "vkBeginCommandBuffer");
  auto end_command_buffer =
      device_proc<PFN_vkEndCommandBuffer>(device, "vkEndCommandBuffer");
  auto reset_command_buffer =
      device_proc<PFN_vkResetCommandBuffer>(device, "vkResetCommandBuffer");
  auto cmd_begin_render_pass =
      device_proc<PFN_vkCmdBeginRenderPass>(device, "vkCmdBeginRenderPass");
  auto cmd_end_render_pass =
      device_proc<PFN_vkCmdEndRenderPass>(device, "vkCmdEndRenderPass");
  auto cmd_bind_pipeline =
      device_proc<PFN_vkCmdBindPipeline>(device, "vkCmdBindPipeline");
  auto cmd_bind_vertex_buffers = device_proc<PFN_vkCmdBindVertexBuffers>(
      device, "vkCmdBindVertexBuffers");
  auto cmd_draw = device_proc<PFN_vkCmdDraw>(device, "vkCmdDraw");
  auto cmd_copy_image_to_buffer = device_proc<PFN_vkCmdCopyImageToBuffer>(
      device, "vkCmdCopyImageToBuffer");
  auto cmd_pipeline_barrier =
      device_proc<PFN_vkCmdPipelineBarrier>(device, "vkCmdPipelineBarrier");

  auto queue_submit = device_proc<PFN_vkQueueSubmit>(device, "vkQueueSubmit");
  auto create_fence = device_proc<PFN_vkCreateFence>(device, "vkCreateFence");
  auto destroy_fence = device_proc<PFN_vkDestroyFence>(device, "vkDestroyFence");
  auto reset_fences = device_proc<PFN_vkResetFences>(device, "vkResetFences");
  auto wait_for_fences = device_proc<PFN_vkWaitForFences>(device, "vkWaitForFences");

  require("vkCreateWaylandSurfaceKHR",
          reinterpret_cast<PFN_vkVoidFunction>(
              instance_proc<CreateWaylandSurfaceFn>(instance,
                                                    "vkCreateWaylandSurfaceKHR")));
  require("vkCreateSwapchainKHR",
          reinterpret_cast<PFN_vkVoidFunction>(create_swapchain));
  require("vkAcquireNextImageKHR",
          reinterpret_cast<PFN_vkVoidFunction>(acquire_next_image));
  require("vkQueuePresentKHR", reinterpret_cast<PFN_vkVoidFunction>(queue_present));
  require("vkGetSwapchainImagesKHR",
          reinterpret_cast<PFN_vkVoidFunction>(get_swapchain_images));
  require("vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
          reinterpret_cast<PFN_vkVoidFunction>(get_surface_capabilities));
  if (failures() > 0) {
    tear_down(&runtime);
    destroy_surface(runtime.instance, surface, nullptr);
    close_wayland(&wayland);
    return 1;
  }

  // ---- the extensions are advertised --------------------------------------
  {
    uint32_t count = 0;
    enumerate_instance_extensions(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    enumerate_instance_extensions(nullptr, &count, available.data());
    bool has_surface = false;
    bool has_wayland = false;
    for (const VkExtensionProperties& extension : available) {
      if (std::strcmp(extension.extensionName, "VK_KHR_surface") == 0)
        has_surface = true;
      if (std::strcmp(extension.extensionName, "VK_KHR_wayland_surface") == 0)
        has_wayland = true;
    }
    check(has_surface, "VK_KHR_surface is advertised");
    check(has_wayland, "VK_KHR_wayland_surface is advertised");

    uint32_t device_count2 = 0;
    enumerate_device_extensions(runtime.physical, nullptr, &device_count2, nullptr);
    std::vector<VkExtensionProperties> device_available(device_count2);
    enumerate_device_extensions(runtime.physical, nullptr, &device_count2,
                                device_available.data());
    bool has_swapchain = false;
    for (const VkExtensionProperties& extension : device_available) {
      if (std::strcmp(extension.extensionName, "VK_KHR_swapchain") == 0)
        has_swapchain = true;
    }
    check(has_swapchain, "VK_KHR_swapchain is advertised");
  }

  // ---- the surface --------------------------------------------------------
  VkBool32 supported = VK_FALSE;
  check(get_surface_support(runtime.physical, 0, surface, &supported) == VK_SUCCESS &&
            supported == VK_TRUE,
        "the queue family can present to this surface");

  {
    VkSurfaceCapabilitiesKHR capabilities{};
    check(get_surface_capabilities(runtime.physical, surface, &capabilities) ==
              VK_SUCCESS,
          "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    check(capabilities.minImageCount >= 2,
          "at least two images are required, so one can be shown while the next "
          "is drawn");
    check(capabilities.currentExtent.width == 0xFFFFFFFFu &&
              capabilities.currentExtent.height == 0xFFFFFFFFu,
          "the extent is the application's to choose");
    check(capabilities.supportedTransforms ==
              VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR &&
              capabilities.currentTransform == VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
          "only the identity transform is offered");
    check((capabilities.supportedUsageFlags &
           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0,
          "swapchain images can be colour attachments");
  }

  {
    uint32_t count = 0;
    check(get_surface_formats(runtime.physical, surface, &count, nullptr) ==
              VK_SUCCESS &&
              count == 1,
          "one surface format is offered");
    VkSurfaceFormatKHR format{};
    check(get_surface_formats(runtime.physical, surface, &count, &format) ==
              VK_SUCCESS &&
              format.format == VK_FORMAT_R8G8B8A8_UNORM &&
              format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
          "the surface format is R8G8B8A8_UNORM, sRGB non-linear");

    uint32_t mode_count = 0;
    check(get_surface_present_modes(runtime.physical, surface, &mode_count,
                                    nullptr) == VK_SUCCESS &&
              mode_count == 1,
          "one present mode is offered");
    VkPresentModeKHR mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    check(get_surface_present_modes(runtime.physical, surface, &mode_count,
                                    &mode) == VK_SUCCESS &&
              mode == VK_PRESENT_MODE_FIFO_KHR,
          "the present mode is FIFO");
  }

  // ---- the swapchain ------------------------------------------------------
  VkSwapchainCreateInfoKHR swapchain_info{};
  swapchain_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swapchain_info.surface = surface;
  swapchain_info.minImageCount = 3;
  swapchain_info.imageFormat = VK_FORMAT_R8G8B8A8_UNORM;
  swapchain_info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  swapchain_info.imageExtent = {kWidth, kHeight};
  swapchain_info.imageArrayLayers = 1;
  swapchain_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  swapchain_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  swapchain_info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swapchain_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  swapchain_info.clipped = VK_TRUE;
  swapchain_info.oldSwapchain = VK_NULL_HANDLE;

  // The refusal battery, identical to the X11 test's: every item this driver
  // could do approximately and chooses not to.
  {
    VkSwapchainKHR refused = VK_NULL_HANDLE;
    VkSwapchainCreateInfoKHR bad = swapchain_info;

    bad.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    check(create_swapchain(device, &bad, nullptr, &refused) ==
                  VK_ERROR_FORMAT_NOT_SUPPORTED &&
              refused == VK_NULL_HANDLE,
          "a B8G8R8A8 swapchain is refused rather than channel swapped");

    bad = swapchain_info;
    bad.imageColorSpace = VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT;
    check(create_swapchain(device, &bad, nullptr, &refused) ==
              VK_ERROR_FORMAT_NOT_SUPPORTED,
          "a display-P3 swapchain is refused");

    bad = swapchain_info;
    bad.minImageCount = 1;
    check(create_swapchain(device, &bad, nullptr, &refused) ==
              VK_ERROR_INITIALIZATION_FAILED,
          "a one-image swapchain is refused");

    bad = swapchain_info;
    bad.presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
    check(create_swapchain(device, &bad, nullptr, &refused) ==
              VK_ERROR_FEATURE_NOT_PRESENT,
          "MAILBOX is refused rather than presented as FIFO");

    bad = swapchain_info;
    bad.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_STORAGE_BIT;
    check(create_swapchain(device, &bad, nullptr, &refused) ==
              VK_ERROR_FEATURE_NOT_PRESENT,
          "a storage swapchain image is refused");

    bad = swapchain_info;
    bad.preTransform = VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR;
    check(create_swapchain(device, &bad, nullptr, &refused) ==
              VK_ERROR_FEATURE_NOT_PRESENT,
          "a rotated swapchain is refused");

    bad = swapchain_info;
    bad.compositeAlpha = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
    check(create_swapchain(device, &bad, nullptr, &refused) ==
              VK_ERROR_FEATURE_NOT_PRESENT,
          "pre-multiplied alpha is refused");
  }

  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  check(create_swapchain(device, &swapchain_info, nullptr, &swapchain) ==
            VK_SUCCESS &&
            swapchain != VK_NULL_HANDLE,
        "vkCreateSwapchainKHR");

  uint32_t image_count = 0;
  check(get_swapchain_images(device, swapchain, &image_count, nullptr) ==
            VK_SUCCESS &&
            image_count == swapchain_info.minImageCount,
        "vkGetSwapchainImagesKHR reports the images asked for");
  std::vector<VkImage> images(image_count);
  get_swapchain_images(device, swapchain, &image_count, images.data());
  {
    bool distinct = true;
    for (uint32_t i = 0; i < image_count; ++i) {
      for (uint32_t j = i + 1; j < image_count; ++j) {
        if (images[i] == images[j])
          distinct = false;
      }
    }
    check(distinct, "the swapchain images are distinct handles");
  }

  // ---- objects ------------------------------------------------------------
  VkRenderPass render_pass = VK_NULL_HANDLE;
  {
    VkAttachmentDescription attachment{};
    attachment.format = VK_FORMAT_R8G8B8A8_UNORM;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference reference{};
    reference.attachment = 0;
    reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &reference;

    VkRenderPassCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = 1;
    info.pAttachments = &attachment;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;

    check(create_render_pass(device, &info, nullptr, &render_pass) == VK_SUCCESS &&
              render_pass != VK_NULL_HANDLE,
          "a render pass whose final layout is PRESENT_SRC_KHR");
  }

  std::vector<VkImageView> views(image_count, VK_NULL_HANDLE);
  std::vector<VkFramebuffer> framebuffers(image_count, VK_NULL_HANDLE);
  {
    bool views_ok = true;
    bool framebuffers_ok = true;
    for (uint32_t i = 0; i < image_count; ++i) {
      VkImageViewCreateInfo view_info{};
      view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      view_info.image = images[i];
      view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
      view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      view_info.subresourceRange.levelCount = 1;
      view_info.subresourceRange.layerCount = 1;
      if (create_image_view(device, &view_info, nullptr, &views[i]) != VK_SUCCESS)
        views_ok = false;

      VkFramebufferCreateInfo framebuffer_info{};
      framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebuffer_info.renderPass = render_pass;
      framebuffer_info.attachmentCount = 1;
      framebuffer_info.pAttachments = &views[i];
      framebuffer_info.width = kWidth;
      framebuffer_info.height = kHeight;
      framebuffer_info.layers = 1;
      if (create_framebuffer(device, &framebuffer_info, nullptr,
                             &framebuffers[i]) != VK_SUCCESS)
        framebuffers_ok = false;
    }
    check(views_ok, "a view of each swapchain image");
    check(framebuffers_ok, "a framebuffer for each swapchain image");
  }

  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  {
    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    create_pipeline_layout(device, &layout_info, nullptr, &layout);

    const std::vector<uint32_t> vertex_code = read_spirv(triangle_vert.c_str());
    const std::vector<uint32_t> fragment_code = read_spirv(triangle_frag.c_str());
    if (vertex_code.empty() || fragment_code.empty()) {
      alarm(0);
      tear_down(&runtime);
      destroy_surface(runtime.instance, surface, nullptr);
      close_wayland(&wayland);
      return 1;
    }

    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    VkShaderModule vertex_module = VK_NULL_HANDLE;
    module_info.codeSize = vertex_code.size() * sizeof(uint32_t);
    module_info.pCode = vertex_code.data();
    create_shader_module(device, &module_info, nullptr, &vertex_module);
    VkShaderModule fragment_module = VK_NULL_HANDLE;
    module_info.codeSize = fragment_code.size() * sizeof(uint32_t);
    module_info.pCode = fragment_code.data();
    create_shader_module(device, &module_info, nullptr, &fragment_module);

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertex_module;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragment_module;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = kVertexStride;
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributes[2] = {};
    attributes[0].location = 0;
    attributes[0].binding = 0;
    attributes[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributes[0].offset = 0;
    attributes[1].location = 1;
    attributes[1].binding = 0;
    attributes[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributes[1].offset = sizeof(float) * 2;

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = &binding;
    vertex_input.vertexAttributeDescriptionCount = 2;
    vertex_input.pVertexAttributeDescriptions = attributes;

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.width = static_cast<float>(kWidth);
    viewport.height = static_cast<float>(kHeight);
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.extent = {kWidth, kHeight};

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                      VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT |
                                      VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;

    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = &stages[0];
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &raster;
    pipeline_info.pMultisampleState = &multisample;
    pipeline_info.pColorBlendState = &blend;
    pipeline_info.layout = layout;
    pipeline_info.renderPass = render_pass;

    check(create_graphics_pipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                    nullptr, &pipeline) == VK_SUCCESS &&
              pipeline != VK_NULL_HANDLE,
          "a graphics pipeline for the swapchain render pass");

    destroy_shader_module(device, vertex_module, nullptr);
    destroy_shader_module(device, fragment_module, nullptr);
  }

  // ---- the vertex buffer and the readback buffer --------------------------
  VkBuffer vertex_buffer = VK_NULL_HANDLE;
  VkDeviceMemory vertex_memory = VK_NULL_HANDLE;
  {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = sizeof(kVertices);
    info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_buffer(device, &info, nullptr, &vertex_buffer);

    VkMemoryRequirements requirements{};
    get_buffer_requirements(device, vertex_buffer, &requirements);
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = runtime.host_memory_type;
    allocate_memory(device, &allocation, nullptr, &vertex_memory);

    void* mapped = nullptr;
    map_memory(device, vertex_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
    std::memcpy(mapped, kVertices, sizeof(kVertices));
    check(bind_buffer_memory(device, vertex_buffer, vertex_memory, 0) == VK_SUCCESS,
          "the vertex buffer is bound");
  }

  const size_t readback_size =
      static_cast<size_t>(kWidth) * kHeight * kBytesPerPixel;
  VkBuffer readback = VK_NULL_HANDLE;
  VkDeviceMemory readback_memory = VK_NULL_HANDLE;
  void* readback_mapped = nullptr;
  {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = readback_size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_buffer(device, &info, nullptr, &readback);

    VkMemoryRequirements requirements{};
    get_buffer_requirements(device, readback, &requirements);
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = runtime.host_memory_type;
    allocate_memory(device, &allocation, nullptr, &readback_memory);
    bind_buffer_memory(device, readback, readback_memory, 0);
    map_memory(device, readback_memory, 0, VK_WHOLE_SIZE, 0, &readback_mapped);
  }

  VkCommandPool pool = VK_NULL_HANDLE;
  {
    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.queueFamilyIndex = 0;
    create_command_pool(device, &info, nullptr, &pool);
  }
  VkCommandBuffer commands = VK_NULL_HANDLE;
  {
    VkCommandBufferAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool = pool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    allocate_command_buffers(device, &info, &commands);
  }
  VkFence fence = VK_NULL_HANDLE;
  {
    VkFenceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    create_fence(device, &info, nullptr, &fence);
  }

  // ---- draw into one image, then present it ------------------------------
  uint32_t index = 0;
  check(acquire_next_image(device, swapchain, UINT64_MAX, VK_NULL_HANDLE,
                           VK_NULL_HANDLE, &index) == VK_SUCCESS &&
            index < image_count,
        "vkAcquireNextImageKHR returns one of the swapchain's images");

  {
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    begin_command_buffer(commands, &begin);

    const VkClearValue clear = clear_value(kClearBytes);
    VkRenderPassBeginInfo pass{};
    pass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    pass.renderPass = render_pass;
    pass.framebuffer = framebuffers[index];
    pass.renderArea.extent = {kWidth, kHeight};
    pass.clearValueCount = 1;
    pass.pClearValues = &clear;
    cmd_begin_render_pass(commands, &pass, VK_SUBPASS_CONTENTS_INLINE);

    cmd_bind_pipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const VkDeviceSize offset = 0;
    cmd_bind_vertex_buffers(commands, 0, 1, &vertex_buffer, &offset);
    cmd_draw(commands, 3, 1, 0, 0);

    cmd_end_render_pass(commands);

    VkImageMemoryBarrier to_transfer{};
    to_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_transfer.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_transfer.image = images[index];
    to_transfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    to_transfer.subresourceRange.levelCount = 1;
    to_transfer.subresourceRange.layerCount = 1;
    to_transfer.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    cmd_pipeline_barrier(commands, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &to_transfer);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {kWidth, kHeight, 1};
    cmd_copy_image_to_buffer(commands, images[index],
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1,
                             &region);

    end_command_buffer(commands);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commands;
    check(queue_submit(queue, 1, &submit, fence) == VK_SUCCESS,
          "the frame is submitted");
    wait_for_fences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    reset_fences(device, 1, &fence);
  }

  {
    const auto* pixels = static_cast<const uint8_t*>(readback_mapped);
    Image drawn;
    drawn.data = pixels;
    drawn.width = kWidth;
    drawn.height = kHeight;
    drawn.pixel_bytes = kBytesPerPixel;
    drawn.label = "the swapchain image";

    std::string difference;
    const int mismatches = compare_with_triangle(drawn, &difference);
    if (mismatches != 0)
      std::printf("     %d pixel(s) differ; first: %s\n", mismatches,
                  difference.c_str());
    check(mismatches == 0,
          "the image holds the triangle the vertex data describes");

    const uint8_t* corner =
        pixels + (static_cast<size_t>(kHeight) - 1) * kWidth * kBytesPerPixel;
    check(std::abs(corner[0] - kClearBytes[0]) <= kTolerance &&
              std::abs(corner[1] - kClearBytes[1]) <= kTolerance &&
              std::abs(corner[2] - kClearBytes[2]) <= kTolerance,
          "the render pass's clear reached the pixels the triangle misses");
  }

  // ---- present, and compare the frame that was about to be shown ----------
  {
    const char* dump_path = "/tmp/opencode/kniz-wayland.ppm";
    setenv("KNIZ_PRESENT_DUMP", dump_path, 1);

    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain;
    present.pImageIndices = &index;
    check(queue_present(queue, &present) == VK_SUCCESS,
          "vkQueuePresentKHR presents the image");
    unsetenv("KNIZ_PRESENT_DUMP");

    Ppm frame;
    const bool read = read_ppm(dump_path, &frame);
    check(read, "the presented frame was captured");
    if (read) {
      if (frame.width != kWidth || frame.height != kHeight) {
        std::printf("     the frame is %ux%u, the window is %ux%u\n", frame.width,
                    frame.height, kWidth, kHeight);
      }
      check(frame.width == kWidth && frame.height == kHeight,
            "the presented frame is the size of the swapped image");

      Image shown;
      shown.data = frame.rgb.data();
      shown.width = frame.width;
      shown.height = frame.height;
      shown.pixel_bytes = 3;
      shown.label = "the presented frame";

      std::string difference;
      const int mismatches = compare_with_triangle(shown, &difference);
      if (mismatches != 0)
        std::printf("     %d pixel(s) differ; first: %s\n", mismatches,
                    difference.c_str());
      check(mismatches == 0,
            "the presented frame is the image, the right way up");
      std::remove(dump_path);
    }
  }

  // ---- acquisition is round robin ----------------------------------------
  {
    std::vector<uint32_t> acquired;
    bool ok = true;
    for (uint32_t i = 0; i < image_count + 1; ++i) {
      uint32_t next = 0;
      if (acquire_next_image(device, swapchain, UINT64_MAX, VK_NULL_HANDLE,
                             VK_NULL_HANDLE, &next) != VK_SUCCESS) {
        ok = false;
        break;
      }
      acquired.push_back(next);
    }
    bool cycles = ok && acquired.size() == image_count + 1;
    for (uint32_t i = 1; cycles && i < acquired.size(); ++i) {
      if (acquired[i] != (acquired[i - 1] + 1) % image_count)
        cycles = false;
    }
    check(cycles, "acquisition walks the images in order and wraps");
  }

  // ---- presenting an index the swapchain does not have -------------------
  {
    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain;
    uint32_t bogus = image_count + 3;
    present.pImageIndices = &bogus;
    VkResult result = VK_SUCCESS;
    present.pResults = &result;
    const VkResult overall = queue_present(queue, &present);
    check(overall != VK_SUCCESS && result != VK_SUCCESS,
          "presenting an image the swapchain does not have is refused");
  }

  // ---- a smaller swapchain is resized into the window --------------------
  {
    // On Wayland the replacement swapchain with a different extent exercises
    // what the X11 test cannot: wl_egl_window_resize.  The WindowSurface is
    // reused -- "the window did not change" -- and the drawable's size follows
    // the new extent, so the captured frame is 160x120 holding the small clear
    // colour in every pixel.
    VkSwapchainCreateInfoKHR small_info = swapchain_info;
    small_info.minImageCount = 2;
    small_info.imageExtent = {kSmallWidth, kSmallHeight};

    VkSwapchainKHR small = VK_NULL_HANDLE;
    check(create_swapchain(device, &small_info, nullptr, &small) == VK_SUCCESS,
          "a second swapchain, smaller than the window");

    uint32_t small_count = 0;
    get_swapchain_images(device, small, &small_count, nullptr);
    std::vector<VkImage> small_images(small_count);
    get_swapchain_images(device, small, &small_count, small_images.data());

    uint32_t small_index = 0;
    acquire_next_image(device, small, UINT64_MAX, VK_NULL_HANDLE, VK_NULL_HANDLE,
                       &small_index);

    VkImageView small_view = VK_NULL_HANDLE;
    VkFramebuffer small_framebuffer = VK_NULL_HANDLE;
    {
      VkImageViewCreateInfo view_info{};
      view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      view_info.image = small_images[small_index];
      view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
      view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      view_info.subresourceRange.levelCount = 1;
      view_info.subresourceRange.layerCount = 1;
      create_image_view(device, &view_info, nullptr, &small_view);

      VkFramebufferCreateInfo framebuffer_info{};
      framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebuffer_info.renderPass = render_pass;
      framebuffer_info.attachmentCount = 1;
      framebuffer_info.pAttachments = &small_view;
      framebuffer_info.width = kSmallWidth;
      framebuffer_info.height = kSmallHeight;
      framebuffer_info.layers = 1;
      create_framebuffer(device, &framebuffer_info, nullptr, &small_framebuffer);
    }

    reset_command_buffer(commands, 0);
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    begin_command_buffer(commands, &begin);

    const VkClearValue clear = clear_value(kSmallClearBytes);
    VkRenderPassBeginInfo pass{};
    pass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    pass.renderPass = render_pass;
    pass.framebuffer = small_framebuffer;
    pass.renderArea.extent = {kSmallWidth, kSmallHeight};
    pass.clearValueCount = 1;
    pass.pClearValues = &clear;
    cmd_begin_render_pass(commands, &pass, VK_SUBPASS_CONTENTS_INLINE);
    cmd_end_render_pass(commands);

    end_command_buffer(commands);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commands;
    check(queue_submit(queue, 1, &submit, fence) == VK_SUCCESS,
          "the small frame is submitted");
    wait_for_fences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    reset_fences(device, 1, &fence);

    const char* dump_path = "/tmp/opencode/kniz-wayland-small.ppm";
    setenv("KNIZ_PRESENT_DUMP", dump_path, 1);
    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.swapchainCount = 1;
    present.pSwapchains = &small;
    present.pImageIndices = &small_index;
    check(queue_present(queue, &present) == VK_SUCCESS,
          "vkQueuePresentKHR presents the small image");
    unsetenv("KNIZ_PRESENT_DUMP");

    Ppm frame;
    const bool read = read_ppm(dump_path, &frame);
    check(read, "the small presented frame was captured");
    if (read) {
      check(frame.width == kSmallWidth && frame.height == kSmallHeight,
            "the drawable followed the smaller swapchain's extent");

      bool uniform = true;
      for (size_t i = 0; i < frame.rgb.size(); ++i) {
        const uint8_t expected = (i % 3 == 0)   ? kSmallClearBytes[0]
                                 : (i % 3 == 1) ? kSmallClearBytes[1]
                                                : kSmallClearBytes[2];
        if (std::abs(static_cast<int>(frame.rgb[i]) - static_cast<int>(expected)) >
            kTolerance)
          uniform = false;
      }
      check(uniform, "every pixel of the frame is the small clear colour");
      std::remove(dump_path);
    }

    destroy_framebuffer(device, small_framebuffer, nullptr);
    destroy_image_view(device, small_view, nullptr);
    destroy_swapchain(device, small, nullptr);
  }

  // ---- unrestricted teardown, the wayland way ----------------------------
  destroy_swapchain(device, swapchain, nullptr);
  for (uint32_t i = 0; i < image_count; ++i) {
    destroy_framebuffer(device, framebuffers[i], nullptr);
    destroy_image_view(device, views[i], nullptr);
  }
  destroy_render_pass(device, render_pass, nullptr);
  destroy_pipeline(device, pipeline, nullptr);
  destroy_pipeline_layout(device, layout, nullptr);
  destroy_buffer(device, vertex_buffer, nullptr);
  free_memory(device, vertex_memory, nullptr);
  destroy_buffer(device, readback, nullptr);
  free_memory(device, readback_memory, nullptr);
  destroy_fence(device, fence, nullptr);
  destroy_command_pool(device, pool, nullptr);

  // The surface goes first, while the context is still alive: this is where
  // the EGL surface and the wl_egl_window behind it are destroyed, in that
  // order.  Then the device, then the instance.
  destroy_surface(runtime.instance, surface, nullptr);
  tear_down(&runtime);
  close_wayland(&wayland);

  alarm(0);
  return failures() == 0 ? 0 : 1;
}
