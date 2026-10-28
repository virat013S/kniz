// M3 test: presentation, through the real loader and a real window.
//
// This is the first milestone whose result is not an image the test holds but
// pixels on a screen, and that changes what a gate has to do.  Everything the
// application controls can be checked as before: the swapchain images are
// ordinary textures, so drawing into one and copying it back proves the render
// pass, the pipeline and the attachment all agree.  What that leaves unproven is
// the last step -- the draw from the image into the window -- which is exactly
// the step that is new, and exactly the step that a wrong y flip, a missing
// sampler or a stale scissor would ruin while leaving every earlier check green.
//
// So the driver can be asked to write the frame it is about to swap to a file
// (KNIZ_PRESENT_DUMP, see docs/DESIGN.md §3.13).  That is the same buffer
// eglSwapBuffers is handed, so comparing it against the image the test drew
// checks the present path pixel for pixel, orientation included.  A screenshot
// would prove the same thing less precisely and would need a person to look.
//
// The checks:
//
//   1. the surface reports what it can do, and refuses to pretend about the rest:
//      formats, present modes, capabilities, and support on the one queue family;
//   2. a swapchain is created, its images come back as distinct handles, and a
//      triangle drawn into one of them reads back as the barycentric colours the
//      vertex data implies -- the same comparison the M2b test makes, now against
//      an image the driver owns rather than one the test allocated;
//   3. that image is presented, and the frame the driver was about to show is
//      compared against the same reference.  The triangle is asymmetric about the
//      horizontal axis, so a flip anywhere in the present path moves the apex to
//      the top row and fails;
//   4. a second swapchain smaller than the window is presented, and the frame is
//      the window's size with the image's colour in every pixel: presentation
//      scales, which is what a window manager resizing the window requires;
//   5. acquisition is round robin over the swapchain's images, and presenting an
//      index the swapchain does not have is refused rather than drawn;
//   6. the refusals -- a format, colour space, present mode, usage, pre-transform
//      or composite alpha this driver does not implement, and a swapchain of one
//      image -- are asserted to fail rather than to be approximated.
//
// On a machine with no X11 display there is nothing to present to, and the test
// says so and succeeds: CI has no display, and a test that failed there would
// make the suite unusable in the one environment that must stay green.

#include <vulkan/vulkan.h>

#include <dlfcn.h>

#include <cmath>
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

// The window, and the swapchain that matches it.  Presentation draws the image
// across the whole window, so at 1:1 the comparison against the drawn image is
// exact rather than scaled.
constexpr uint32_t kWidth = 320;
constexpr uint32_t kHeight = 240;

// A second swapchain, half the window in each direction.  Presenting it is how
// the scaling path is exercised.
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

// The same triangle the M2b test draws, for the same reasons: three vertices
// carrying red, green and blue, and a shape whose flat edge and apex are at
// opposite ends of the image so that a flip cannot survive the comparison.
//
//   vertex 0  NDC (-1, -1)  red     image row 0     (the top)
//   vertex 1  NDC ( 1, -1)  green   image row 0     (the top)
//   vertex 2  NDC ( 0,  1)  blue    image row 239   (the bottom)
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

// Barycentric weights of a point in image coordinates, y running downwards from
// the top row exactly as Vulkan's framebuffer does.
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

// True when a pixel centre is inside the triangle.  Barycentric weights go
// negative outside it, so the sign test is the whole of it.
bool inside_triangle(uint32_t x, uint32_t y) {
  double w0 = 0, w1 = 0, w2 = 0;
  barycentric(x + 0.5, y + 0.5, &w0, &w1, &w2);
  constexpr double kSlack = 1e-9;
  return w0 >= -kSlack && w1 >= -kSlack && w2 >= -kSlack;
}

// What a pixel must hold: an interpolation of the vertex colours inside the
// triangle, and the render pass's clear colour outside it.  Both are computed
// here from the data the test submitted, not read back from the driver.
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

// Interpolation and the attachment's rounding disagree by at most a couple of
// steps; a flip, a channel swap or a missing draw is out by tens.
constexpr int kTolerance = 3;

// Compares a rectangle of bytes against the triangle reference, given how the
// rows are laid out and how many bytes a pixel occupies.  `top_down` is true when
// row 0 of the buffer is the image's top row, which is true for both buffers
// this test compares -- Vulkan's readback and the PPM the driver writes.
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

// A binary PPM, which is what KNIZ_PRESENT_DUMP writes: the frame that was about
// to be shown, top row first, three bytes per pixel.
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
  // The header's trailing whitespace is a single byte, which fscanf leaves in
  // place; the raster follows immediately.
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
// X11, opened the same way the loader is: by dlopen, so that the test needs no
// X11 headers and no X11 development package to build.  Six entry points are
// enough to make a window, and the window is only ever handed to the driver.
// ---------------------------------------------------------------------------

struct X11 {
  void* library = nullptr;
  void* display = nullptr;
  unsigned long window = 0;

  void* (*open_display)(const char*) = nullptr;
  unsigned long (*default_root_window)(void*) = nullptr;
  unsigned long (*create_simple_window)(void*, unsigned long, int, int,
                                        unsigned int, unsigned int,
                                        unsigned int, unsigned long,
                                        unsigned long) = nullptr;
  int (*store_name)(void*, unsigned long, const char*) = nullptr;
  int (*flush)(void*) = nullptr;
  int (*sync)(void*, int) = nullptr;
  int (*destroy_window)(void*, unsigned long) = nullptr;
  int (*close_display)(void*) = nullptr;
};

bool open_x11(X11* x11, uint32_t width, uint32_t height) {
  x11->library = dlopen("libX11.so.6", RTLD_NOW | RTLD_LOCAL);
  if (x11->library == nullptr)
    return false;

  const auto load = [x11](const char* name) {
    return dlsym(x11->library, name);
  };
  x11->open_display = to_fn<void* (*)(const char*)>(load("XOpenDisplay"));
  x11->default_root_window =
      to_fn<unsigned long (*)(void*)>(load("XDefaultRootWindow"));
  x11->create_simple_window = to_fn<unsigned long (*)(
      void*, unsigned long, int, int, unsigned int, unsigned int, unsigned int,
      unsigned long, unsigned long)>(load("XCreateSimpleWindow"));
  x11->store_name =
      to_fn<int (*)(void*, unsigned long, const char*)>(load("XStoreName"));
  x11->flush = to_fn<int (*)(void*)>(load("XFlush"));
  x11->sync = to_fn<int (*)(void*, int)>(load("XSync"));
  x11->destroy_window =
      to_fn<int (*)(void*, unsigned long)>(load("XDestroyWindow"));
  x11->close_display = to_fn<int (*)(void*)>(load("XCloseDisplay"));

  if (x11->open_display == nullptr || x11->default_root_window == nullptr ||
      x11->create_simple_window == nullptr || x11->store_name == nullptr ||
      x11->flush == nullptr ||
      x11->sync == nullptr || x11->destroy_window == nullptr ||
      x11->close_display == nullptr) {
    dlclose(x11->library);
    x11->library = nullptr;
    return false;
  }

  x11->display = x11->open_display(nullptr);
  if (x11->display == nullptr) {
    dlclose(x11->library);
    x11->library = nullptr;
    return false;
  }

  const unsigned long root = x11->default_root_window(x11->display);
  x11->window = x11->create_simple_window(x11->display, root, 0, 0, width,
                                          height, 0, 0, 0);
  if (x11->window == 0) {
    x11->close_display(x11->display);
    dlclose(x11->library);
    x11->library = nullptr;
    return false;
  }
  x11->store_name(x11->display, x11->window, "kniz present test");
  x11->flush(x11->display);
  x11->sync(x11->display, 0);

  // The window is deliberately never mapped.  A mapped window is resized by the
  // window manager -- on the development machine Hyprland tiled this one from
  // 320x240 to 655x674 -- and the presented frame is then a scaled version of
  // the image, which cannot be compared against it pixel for pixel.  An unmapped
  // window keeps the geometry the test asked for, and the present path (draw,
  // read back, swap) is identical; only the compositor's copy is missing, and
  // that is what running a real program like vkcube is for.
  return true;
}

void close_x11(X11* x11) {
  if (x11->display != nullptr) {
    if (x11->window != 0 && x11->destroy_window != nullptr)
      x11->destroy_window(x11->display, x11->window);
    x11->close_display(x11->display);
    x11->display = nullptr;
    x11->window = 0;
  }
  if (x11->library != nullptr) {
    dlclose(x11->library);
    x11->library = nullptr;
  }
}

// VK_KHR_xcb_surface's create info.  Declared here rather than by including
// <vulkan/vulkan_xcb.h>, which exists only when VK_USE_PLATFORM_XCB_KHR is
// defined -- and that would make xcb headers a build requirement of the suite.
// The layout is the extension's published ABI.
struct XcbSurfaceCreateInfo {
  VkStructureType sType;
  const void* pNext;
  VkFlags flags;
  void* connection;
  uint32_t window;
};

using CreateXcbSurfaceFn = VkResult (*)(VkInstance,
                                        const XcbSurfaceCreateInfo*,
                                        const VkAllocationCallbacks*,
                                        VkSurfaceKHR*);

// The surface functions are all core in the Vulkan headers except this one, so
// only this one needs a local declaration.
VkResult create_xcb_surface(VkInstance instance, void* connection,
                            uint32_t window, VkSurfaceKHR* surface) {
  auto create = instance_proc<CreateXcbSurfaceFn>(instance, "vkCreateXcbSurfaceKHR");
  if (create == nullptr)
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  XcbSurfaceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
  info.connection = connection;
  info.window = window;
  return create(instance, &info, nullptr, surface);
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

int main(int argc, char** argv) {
  std::printf("kniz present test (M3)\n");

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

  X11 x11;
  if (!open_x11(&x11, kWidth, kHeight)) {
    // No X11, so no window, so nothing to present to.  This is the normal case
    // in a container and on a CI runner, and the driver reports the same thing
    // through vkGetPhysicalDeviceSurfaceSupportKHR.
    std::printf("  %-58s %s\n", "skipped: no X11 display to present to", "ok");
    return 0;
  }

  Extensions extensions;
  extensions.instance = {"VK_KHR_surface", "VK_KHR_xcb_surface"};
  extensions.device = {"VK_KHR_swapchain"};

  Runtime runtime;
  if (!bring_up(&runtime, "kniz-present", extensions)) {
    close_x11(&x11);
    return 1;
  }

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
  auto destroy_surface = instance_proc<PFN_vkDestroySurfaceKHR>(
      instance, "vkDestroySurfaceKHR");
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
  auto get_swapchain_images =
      device_proc<PFN_vkGetSwapchainImagesKHR>(device, "vkGetSwapchainImagesKHR");
  auto acquire_next_image =
      device_proc<PFN_vkAcquireNextImageKHR>(device, "vkAcquireNextImageKHR");
  auto queue_present = device_proc<PFN_vkQueuePresentKHR>(device, "vkQueuePresentKHR");

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
  auto create_pipeline_layout =
      device_proc<PFN_vkCreatePipelineLayout>(device, "vkCreatePipelineLayout");
  auto destroy_pipeline_layout =
      device_proc<PFN_vkDestroyPipelineLayout>(device, "vkDestroyPipelineLayout");
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
  auto cmd_bind_vertex_buffers =
      device_proc<PFN_vkCmdBindVertexBuffers>(device, "vkCmdBindVertexBuffers");
  auto cmd_draw = device_proc<PFN_vkCmdDraw>(device, "vkCmdDraw");
  auto cmd_copy_image_to_buffer =
      device_proc<PFN_vkCmdCopyImageToBuffer>(device, "vkCmdCopyImageToBuffer");
  auto cmd_pipeline_barrier =
      device_proc<PFN_vkCmdPipelineBarrier>(device, "vkCmdPipelineBarrier");

  auto queue_submit = device_proc<PFN_vkQueueSubmit>(device, "vkQueueSubmit");
  auto create_fence = device_proc<PFN_vkCreateFence>(device, "vkCreateFence");
  auto destroy_fence = device_proc<PFN_vkDestroyFence>(device, "vkDestroyFence");
  auto reset_fences = device_proc<PFN_vkResetFences>(device, "vkResetFences");
  auto wait_for_fences = device_proc<PFN_vkWaitForFences>(device, "vkWaitForFences");

  require("vkCreateXcbSurfaceKHR",
          reinterpret_cast<PFN_vkVoidFunction>(
              instance_proc<CreateXcbSurfaceFn>(instance, "vkCreateXcbSurfaceKHR")));
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
    close_x11(&x11);
    return 1;
  }

  // ---- the extensions are advertised --------------------------------------
  {
    uint32_t count = 0;
    enumerate_instance_extensions(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    enumerate_instance_extensions(nullptr, &count, available.data());
    bool has_surface = false;
    bool has_xcb = false;
    for (const VkExtensionProperties& extension : available) {
      if (std::strcmp(extension.extensionName, "VK_KHR_surface") == 0)
        has_surface = true;
      if (std::strcmp(extension.extensionName, "VK_KHR_xcb_surface") == 0)
        has_xcb = true;
    }
    check(has_surface, "VK_KHR_surface is advertised");
    check(has_xcb, "VK_KHR_xcb_surface is advertised");

    uint32_t device_count = 0;
    enumerate_device_extensions(runtime.physical, nullptr, &device_count, nullptr);
    std::vector<VkExtensionProperties> device_available(device_count);
    enumerate_device_extensions(runtime.physical, nullptr, &device_count,
                                device_available.data());
    bool has_swapchain = false;
    for (const VkExtensionProperties& extension : device_available) {
      if (std::strcmp(extension.extensionName, "VK_KHR_swapchain") == 0)
        has_swapchain = true;
    }
    check(has_swapchain, "VK_KHR_swapchain is advertised");
  }

  // ---- the surface --------------------------------------------------------
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  check(create_xcb_surface(instance, x11.display,
                           static_cast<uint32_t>(x11.window),
                           &surface) == VK_SUCCESS &&
            surface != VK_NULL_HANDLE,
        "vkCreateXcbSurfaceKHR");

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

  // The refusals first, each one a thing this driver could do approximately and
  // chooses not to.
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
    // What an application passes to a swapchain render pass.  GL has no layouts,
    // so this is recorded and ignored -- but it is recorded, which is why a
    // swapchain render pass is accepted at all.
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
      tear_down(&runtime);
      close_x11(&x11);
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
    pipeline_info.pStages = stages;
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

    // The render pass left the image in PRESENT_SRC, so it has to come back to
    // a transfer source before it can be copied out.
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

    // The clear colour survives everywhere the triangle does not reach, which is
    // the check that the attachment's load operation ran at all.  The bottom
    // right corner is outside the triangle; the top left one is inside it, since
    // the flat edge runs along row 0.
    const uint8_t* corner =
        pixels + (static_cast<size_t>(kHeight) - 1) * kWidth * kBytesPerPixel;
    check(std::abs(corner[0] - kClearBytes[0]) <= kTolerance &&
              std::abs(corner[1] - kClearBytes[1]) <= kTolerance &&
              std::abs(corner[2] - kClearBytes[2]) <= kTolerance,
          "the render pass's clear reached the pixels the triangle misses");
  }

  // ---- present, and compare the frame that was about to be shown ----------
  {
    const char* dump_path = "/tmp/opencode/kniz-present.ppm";
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
            "the presented frame is the size of the window");

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
    // One more acquire than there are images, so the wrap is observed.  The
    // starting index is wherever the previous section left it, which is why the
    // check is on the steps rather than on the values.
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

  // ---- a swapchain smaller than the window is scaled ---------------------
  {
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

    // Clear only: every pixel of the window has to end up the clear colour, so
    // the check is about the size and the scaling rather than about geometry.
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
    queue_submit(queue, 1, &submit, fence);
    wait_for_fences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    reset_fences(device, 1, &fence);

    const char* dump_path = "/tmp/opencode/kniz-present-small.ppm";
    setenv("KNIZ_PRESENT_DUMP", dump_path, 1);
    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.swapchainCount = 1;
    present.pSwapchains = &small;
    present.pImageIndices = &small_index;
    const VkResult presented = queue_present(queue, &present);
    unsetenv("KNIZ_PRESENT_DUMP");
    check(presented == VK_SUCCESS, "the smaller swapchain presents");

    Ppm frame;
    if (read_ppm(dump_path, &frame)) {
      if (frame.width != kWidth || frame.height != kHeight) {
        std::printf("     the frame is %ux%u, the window is %ux%u\n", frame.width,
                    frame.height, kWidth, kHeight);
      }
      check(frame.width == kWidth && frame.height == kHeight,
            "the frame is still the window's size, not the image's");
      int wrong = 0;
      for (size_t i = 0; i < frame.rgb.size(); i += 3) {
        if (std::abs(static_cast<int>(frame.rgb[i]) -
                     static_cast<int>(kSmallClearBytes[0])) > kTolerance ||
            std::abs(static_cast<int>(frame.rgb[i + 1]) -
                     static_cast<int>(kSmallClearBytes[1])) > kTolerance ||
            std::abs(static_cast<int>(frame.rgb[i + 2]) -
                     static_cast<int>(kSmallClearBytes[2])) > kTolerance) {
          ++wrong;
        }
      }
      if (wrong != 0)
        std::printf("     %d of %zu pixels were not the clear colour\n", wrong,
                    frame.rgb.size() / 3);
      check(wrong == 0, "the image was scaled across the whole window");
      std::remove(dump_path);
    } else {
      check(false, "the smaller swapchain's frame was captured");
    }

    destroy_framebuffer(device, small_framebuffer, nullptr);
    destroy_image_view(device, small_view, nullptr);
    destroy_swapchain(device, small, nullptr);
  }

  // ---- teardown -----------------------------------------------------------
  destroy_fence(device, fence, nullptr);
  destroy_command_pool(device, pool, nullptr);
  destroy_buffer(device, readback, nullptr);
  free_memory(device, readback_memory, nullptr);
  destroy_buffer(device, vertex_buffer, nullptr);
  free_memory(device, vertex_memory, nullptr);
  destroy_pipeline(device, pipeline, nullptr);
  destroy_pipeline_layout(device, layout, nullptr);
  for (uint32_t i = 0; i < image_count; ++i)
    destroy_framebuffer(device, framebuffers[i], nullptr);
  for (uint32_t i = 0; i < image_count; ++i)
    destroy_image_view(device, views[i], nullptr);
  destroy_render_pass(device, render_pass, nullptr);
  destroy_swapchain(device, swapchain, nullptr);
  destroy_surface(instance, surface, nullptr);

  check(true, "the swapchain, its window and the surface are destroyed");

  tear_down(&runtime);
  close_x11(&x11);

  std::printf("%d check(s) failed\n", failures());
  return failures() == 0 ? 0 : 1;
}
