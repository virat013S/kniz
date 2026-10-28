// Presentation: VK_KHR_surface, VK_KHR_xcb_surface and VK_KHR_swapchain.
//
// The shape of the translation is forced by one fact: GL cannot make a texture
// *be* a window's front buffer.  So a swapchain image is an ordinary GL texture
// -- the same object M2a created for VkImage, which is why drawing into one
// needs no new code at all -- and vkQueuePresentKHR is a draw from that texture
// into the window's default framebuffer, followed by eglSwapBuffers.
//
// Two consequences are worth stating before the code:
//
//   * The present is a draw rather than a blit because the two ends need not
//     share a format.  glBlitFramebuffer requires compatible formats, and the
//     window's is chosen by the window system, not by us; a blit would either
//     refuse configs or swap channels silently.  A textured triangle lets GL do
//     the conversion the way it does for every other draw.
//
//   * The y flip lives here and nowhere else.  GL's window coordinates count up
//     from the bottom, so a texture's row 0 is the bottom row of the window,
//     while in Kniz row 0 is Vulkan's *top* row.  That is why every other path
//     in this driver draws without flipping (§4, M2b) and this one does not:
//     the window is the only surface whose row order is the window system's
//     rather than Vulkan's.
//
// Nothing here talks to the window system directly.  A surface stores the native
// handle the application gave us, EGL turns it into a drawable, and the only
// questions asked of the window afterwards -- its size -- are asked of EGL.

#include "kniz_swapchain.h"

#include "kniz_common.h"
#include "kniz_gl.h"
#include "kniz_image.h"
#include "kniz_objects.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace kniz {
namespace {

// The one swapchain format Kniz offers.
//
// Presentation is exact for it: the image is a GL RGBA8 texture, and the draw
// writes it to the window's default framebuffer, whose channel order and sRGB
// encoding are the window system's business.  An sRGB swapchain format is
// refused rather than accepted, because a UNORM window framebuffer would then
// need the encoding applied by hand and getting that wrong is a colour that
// looks almost right.  B8G8R8A8 would work through the same draw, but nothing
// has yet been verified with a channel order other than the one the tests use,
// so it is not offered.
constexpr VkFormat kSwapchainFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkColorSpaceKHR kSwapchainColorSpace =
    VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

// FIFO is the one mode the specification requires every implementation to
// support, and it is the one Kniz can actually honour: presentation is
// synchronous and eglSwapBuffers blocks when the queue is full.  Mailbox and
// immediate would both be lies -- there is no queue to drop or skip.
constexpr VkPresentModeKHR kPresentMode = VK_PRESENT_MODE_FIFO_KHR;

// 16384 is what every GL implementation since 3.x guarantees for
// GL_MAX_TEXTURE_SIZE, so it is the largest image Kniz can promise.
constexpr uint32_t kMaxImageExtent = 16384;

// The present shaders.  This is the only GLSL Kniz writes by hand -- everything
// else arrives as SPIR-V and is translated -- so it is compiled directly rather
// than through kniz_shader.cpp.
//
// The triangle is generated from gl_VertexID, so there is no vertex buffer and
// no attribute to bind.  Its three corners are (0,0), (2,0) and (0,2) in
// texcoord space, which map to (-1,-1), (3,-1) and (-1,3): a triangle that
// covers the viewport with room to spare, which is why the rasteriser never has
// to deal with a seam down the middle of the window.
constexpr const char* kPresentVertexShader = R"(#version 450
layout(location = 0) out vec2 texcoord;
void main() {
  vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  texcoord = corner;
  gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
)";

constexpr const char* kPresentFragmentShader = R"(#version 450
layout(binding = 0) uniform sampler2D source_image;
layout(location = 0) in vec2 texcoord;
layout(location = 0) out vec4 colour;
void main() {
  // The one y flip in the driver.  The window's bottom row is texcoord.y = 0,
  // and the image's bottom row is v = 1.
  colour = texture(source_image, vec2(texcoord.x, 1.0 - texcoord.y));
}
)";

gl::Uint compile_stage(gl::Enum type, const char* source, const char* label) {
  gl::Uint shader = gl::f.CreateShader(type);
  if (shader == 0) {
    KNIZ_ERR("could not create the present %s shader", label);
    return 0;
  }
  gl::f.ShaderSource(shader, 1, &source, nullptr);
  gl::f.CompileShader(shader);

  gl::Int ok = 0;
  gl::f.GetShaderiv(shader, gl::kCompileStatus, &ok);
  if (ok == 0) {
    char log[1024] = {};
    gl::f.GetShaderInfoLog(shader, sizeof(log), nullptr, log);
    KNIZ_ERR("the present %s shader did not compile: %s", label, log);
    gl::f.DeleteShader(shader);
    return 0;
  }
  return shader;
}

// Builds the program, vertex array and sampler a swapchain presents with.
// Returns false with the GL side left clean.
bool create_present_state(Swapchain* swapchain) {
  gl::Uint vertex = compile_stage(gl::kVertexShader, kPresentVertexShader,
                                  "vertex");
  if (vertex == 0)
    return false;

  gl::Uint fragment = compile_stage(gl::kFragmentShader,
                                    kPresentFragmentShader, "fragment");
  if (fragment == 0) {
    gl::f.DeleteShader(vertex);
    return false;
  }

  gl::Uint program = gl::f.CreateProgram();
  gl::f.AttachShader(program, vertex);
  gl::f.AttachShader(program, fragment);
  gl::f.LinkProgram(program);
  // The shaders are owned by the program once attached, so they are deleted
  // either way.
  gl::f.DeleteShader(vertex);
  gl::f.DeleteShader(fragment);

  gl::Int linked = 0;
  gl::f.GetProgramiv(program, gl::kLinkStatus, &linked);
  if (linked == 0) {
    char log[1024] = {};
    gl::f.GetProgramInfoLog(program, sizeof(log), nullptr, log);
    KNIZ_ERR("the present program did not link: %s", log);
    gl::f.DeleteProgram(program);
    return false;
  }
  swapchain->present_program = program;

  gl::f.GenVertexArrays(1, &swapchain->present_array);

  // Linear rather than nearest, so a window that is not the swapchain's size
  // scales smoothly instead of showing whichever texel it landed on.  Clamped,
  // because the triangle's texcoords reach exactly 0 and 1 at the corners and
  // a repeat sampler would fold the opposite edge in along them.
  gl::f.GenSamplers(1, &swapchain->present_sampler);
  gl::f.SamplerParameteri(swapchain->present_sampler, gl::kTextureMinFilter,
                          gl::kLinear);
  gl::f.SamplerParameteri(swapchain->present_sampler, gl::kTextureMagFilter,
                          gl::kLinear);
  gl::f.SamplerParameteri(swapchain->present_sampler, gl::kTextureWrapS,
                          gl::kClampToEdge);
  gl::f.SamplerParameteri(swapchain->present_sampler, gl::kTextureWrapT,
                          gl::kClampToEdge);

  return true;
}

// A swapchain image is a texture with no memory object behind it: the
// application never allocates or binds memory for one, and Vulkan says so by
// requiring that the images come from vkGetSwapchainImagesKHR.
Image* create_swapchain_image(Device* device, VkExtent2D extent) {
  Image* image = new (std::nothrow) Image();
  if (image == nullptr)
    return nullptr;

  image->device = device;
  image->type = VK_IMAGE_TYPE_2D;
  image->format = kSwapchainFormat;
  image->extent = {extent.width, extent.height, 1};
  image->mip_levels = 1;
  image->array_layers = 1;
  image->usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

  gl::f.GenTextures(1, &image->gl_texture);
  if (image->gl_texture == 0) {
    delete image;
    return nullptr;
  }
  if (!image_create_storage(image)) {
    gl::f.DeleteTextures(1, &image->gl_texture);
    delete image;
    return nullptr;
  }
  return image;
}

void destroy_image(Image* image) {
  if (image == nullptr)
    return;
  if (image->gl_texture != 0)
    gl::f.DeleteTextures(1, &image->gl_texture);
  delete image;
}

void destroy_swapchain_objects(Swapchain* swapchain) {
  for (Image* image : swapchain->images)
    destroy_image(image);
  swapchain->images.clear();

  if (swapchain->present_sampler != 0)
    gl::f.DeleteSamplers(1, &swapchain->present_sampler);
  if (swapchain->present_array != 0)
    gl::f.DeleteVertexArrays(1, &swapchain->present_array);
  if (swapchain->present_program != 0)
    gl::f.DeleteProgram(swapchain->present_program);
}

// Writes the frame that is about to be shown to the file named by
// KNIZ_PRESENT_DUMP, as a PPM.
//
// This exists because presentation is the one thing in the driver whose result
// is normally only visible to a person.  A screenshot tool would prove it, but
// only by eye; reading back the very buffer that is about to be swapped lets the
// test compare it against the image it drew, pixel for pixel, including the
// orientation.  It is off unless the variable is set, and it is the only reason
// glReadPixels is resolved at all.
void dump_presented_frame(int width, int height) {
  const char* path = std::getenv("KNIZ_PRESENT_DUMP");
  if (path == nullptr || *path == '\0')
    return;
  if (gl::f.ReadPixels == nullptr || gl::f.PixelStorei == nullptr)
    return;

  std::vector<uint8_t> pixels(static_cast<size_t>(width) *
                              static_cast<size_t>(height) * 4);
  gl::f.PixelStorei(gl::kPackAlignment, 1);
  gl::f.ReadPixels(0, 0, width, height, gl::kRgba, gl::kUnsignedByte,
                   pixels.data());

  // Written beside the destination and renamed into place, because this runs on
  // every present: a reader that opens the file while a frame is half written
  // gets a truncated PPM, and the whole point of the dump is that a frame can be
  // looked at without racing the application.
  std::string temporary(path);
  temporary += ".tmp";
  std::FILE* file = std::fopen(temporary.c_str(), "wb");
  if (file == nullptr) {
    KNIZ_WARN("could not open %s for KNIZ_PRESENT_DUMP", temporary.c_str());
    return;
  }
  std::fprintf(file, "P6\n%d %d\n255\n", width, height);
  // glReadPixels starts at the bottom of the window; a PPM starts at the top.
  for (int y = height - 1; y >= 0; --y) {
    const uint8_t* row = pixels.data() + static_cast<size_t>(y) * width * 4;
    for (int x = 0; x < width; ++x)
      std::fwrite(row + x * 4, 1, 3, file);
  }
  const bool written = std::fclose(file) == 0;
  if (!written || std::rename(temporary.c_str(), path) != 0) {
    KNIZ_WARN("could not write the presented frame to %s", path);
    std::remove(temporary.c_str());
    return;
  }
  KNIZ_DBG("presented frame written to %s", path);
}

// Draws one image into the window.  Called with a GL context current.
bool present_image(Swapchain* swapchain, uint32_t index) {
  if (index >= swapchain->images.size()) {
    KNIZ_ERR("presented image index %u, but the swapchain has %zu image(s)",
             index, swapchain->images.size());
    return false;
  }

  // The context has to be current on the window drawable before anything below
  // means what it says: the default framebuffer is that surface's.
  gl::WindowSurface window = swapchain->surface->window;
  if (!gl::bind_window_surface(window))
    return false;

  int width = 0;
  int height = 0;
  const bool sized = gl::window_surface_size(window, &width, &height);
  if (!sized || width <= 0 || height <= 0) {
    // A minimised window has no drawable to draw into.  Vulkan's answer to that
    // is a surface with zero extent and a present that does nothing, which is
    // what this is; the next acquire still hands out an image.
    KNIZ_DBG("not presenting: the window has no drawable size");
    gl::bind_headless_surface();
    return sized;
  }

  // Everything the present draw touches has to be set explicitly: the
  // application's last draw left the scissor, the masks and the enables in
  // whatever state its pipeline asked for, and a present that inherited them
  // would show part of a frame, or none of it.
  gl::f.BindFramebuffer(gl::kFramebuffer, 0);
  gl::f.Viewport(0, 0, width, height);
  gl::f.Disable(gl::kScissorTest);
  gl::f.Disable(gl::kDepthTest);
  gl::f.Disable(gl::kStencilTest);
  gl::f.Disable(gl::kBlend);
  gl::f.Disable(gl::kCullFace);
  gl::f.Disable(gl::kDither);
  gl::f.Disable(gl::kFramebufferSrgb);
  gl::f.ColorMask(1, 1, 1, 1);

  gl::f.UseProgram(swapchain->present_program);
  gl::f.BindVertexArray(swapchain->present_array);
  gl::f.ActiveTexture(gl::kTexture0);
  gl::f.BindTexture(gl::kTexture2D, swapchain->images[index]->gl_texture);
  gl::f.BindSampler(0, swapchain->present_sampler);
  gl::f.DrawArrays(gl::kTriangles, 0, 3);

  dump_presented_frame(width, height);

  // Leave the sampler unbound rather than leaving it on unit 0: a pipeline that
  // samples from unit 0 binds its own sampler first, so this is tidiness rather
  // than correctness, but the texture unit namespace is shared state.
  gl::f.BindSampler(0, 0);
  gl::f.BindVertexArray(0);
  gl::f.UseProgram(0);

  const bool drew = !gl::report_errors("present");
  const bool swapped = drew && gl::swap_window_surface(window);
  gl::bind_headless_surface();
  return swapped;
}

}  // namespace

bool swapchain_extension_supported(const char* name) {
  return name != nullptr && std::strcmp(name, "VK_KHR_swapchain") == 0;
}

// ---------------------------------------------------------------------------
// Surfaces
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateXcbSurfaceKHR(
    VkInstance instanceHandle, const XcbSurfaceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface) {
  if (pSurface == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pSurface = VK_NULL_HANDLE;

  Instance* instance = from_handle(instanceHandle);
  if (instance == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (pCreateInfo == nullptr || pCreateInfo->connection == nullptr ||
      pCreateInfo->window == 0) {
    KNIZ_ERR("vkCreateXcbSurfaceKHR needs both a connection and a window");
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (pAllocator != nullptr)
    KNIZ_WARN("host allocation callbacks are not supported and will be ignored");

  Surface* surface = nullptr;
  try {
    surface = new Surface();
  } catch (const std::bad_alloc&) {
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  }
  surface->instance = instance;
  surface->native_window = pCreateInfo->window;

  *pSurface = to_handle(surface);
  KNIZ_DBG("xcb surface created for window 0x%x", pCreateInfo->window);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroySurfaceKHR(VkInstance instanceHandle,
                                             VkSurfaceKHR surfaceHandle,
                                             const VkAllocationCallbacks*) {
  (void)instanceHandle;
  Surface* surface = from_handle(surfaceHandle);
  if (surface == nullptr)
    return;
  if (surface->window != nullptr && gl::make_current())
    gl::destroy_window_surface(surface->window);
  surface->magic = 0;
  delete surface;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceSupportKHR(
    VkPhysicalDevice physicalHandle, uint32_t queueFamilyIndex,
    VkSurfaceKHR surfaceHandle, VkBool32* pSupported) {
  if (pSupported == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  PhysicalDevice* physical = from_handle(physicalHandle);
  Surface* surface = from_handle(surfaceHandle);
  if (physical == nullptr || surface == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  *pSupported = VK_FALSE;
  if (queueFamilyIndex != 0)
    return VK_SUCCESS;

  // Reported honestly rather than optimistically: on a surfaceless context
  // there is no way to put a pixel on a screen, and an application that asks
  // deserves to be told before it builds a swapchain around the answer.
  *pSupported = physical->instance->gl.window_capable ? VK_TRUE : VK_FALSE;
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceCapabilitiesKHR(
    VkPhysicalDevice physicalHandle, VkSurfaceKHR surfaceHandle,
    VkSurfaceCapabilitiesKHR* pCapabilities) {
  if (pCapabilities == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  PhysicalDevice* physical = from_handle(physicalHandle);
  Surface* surface = from_handle(surfaceHandle);
  if (physical == nullptr || surface == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  *pCapabilities = VkSurfaceCapabilitiesKHR{};
  pCapabilities->minImageCount = 2;
  // No upper bound: the swapchain is a list of textures and there is nothing
  // to exhaust but memory.
  pCapabilities->maxImageCount = 0;

  // The window's size is the window system's to decide, and asking for it here
  // would mean a round trip through a connection Kniz deliberately does not
  // hold.  Reporting "undefined" hands the choice to the application, which is
  // exactly what Vulkan's 0xFFFFFFFF means; presentation then scales to
  // whatever the window turns out to be.
  pCapabilities->currentExtent.width = 0xFFFFFFFFu;
  pCapabilities->currentExtent.height = 0xFFFFFFFFu;
  pCapabilities->minImageExtent = {1, 1};
  pCapabilities->maxImageExtent = {kMaxImageExtent, kMaxImageExtent};
  pCapabilities->maxImageArrayLayers = 1;

  pCapabilities->supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  pCapabilities->currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  pCapabilities->supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  pCapabilities->supportedUsageFlags =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceFormatsKHR(
    VkPhysicalDevice physicalHandle, VkSurfaceKHR surfaceHandle,
    uint32_t* pSurfaceFormatCount, VkSurfaceFormatKHR* pSurfaceFormats) {
  if (pSurfaceFormatCount == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  PhysicalDevice* physical = from_handle(physicalHandle);
  Surface* surface = from_handle(surfaceHandle);
  if (physical == nullptr || surface == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  constexpr uint32_t kCount = 1;
  const VkResult result =
      enumerate_count(kCount, pSurfaceFormatCount, pSurfaceFormats);
  if (result == VK_SUCCESS && pSurfaceFormats != nullptr) {
    pSurfaceFormats[0].format = kSwapchainFormat;
    pSurfaceFormats[0].colorSpace = kSwapchainColorSpace;
  }
  return result;
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfacePresentModesKHR(
    VkPhysicalDevice physicalHandle, VkSurfaceKHR surfaceHandle,
    uint32_t* pPresentModeCount, VkPresentModeKHR* pPresentModes) {
  if (pPresentModeCount == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  PhysicalDevice* physical = from_handle(physicalHandle);
  Surface* surface = from_handle(surfaceHandle);
  if (physical == nullptr || surface == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  constexpr uint32_t kCount = 1;
  const VkResult result =
      enumerate_count(kCount, pPresentModeCount, pPresentModes);
  if (result == VK_SUCCESS && pPresentModes != nullptr)
    pPresentModes[0] = kPresentMode;
  return result;
}

// ---------------------------------------------------------------------------
// Swapchains
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateSwapchainKHR(
    VkDevice deviceHandle, const VkSwapchainCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain) {
  if (pSwapchain == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pSwapchain = VK_NULL_HANDLE;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (pCreateInfo == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (pAllocator != nullptr)
    KNIZ_WARN("host allocation callbacks are not supported and will be ignored");

  Surface* surface = from_handle(pCreateInfo->surface);
  if (surface == nullptr) {
    KNIZ_ERR("vkCreateSwapchainKHR was given a surface this driver did not "
             "create");
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (!device_can_run_gl(device))
    return VK_ERROR_INITIALIZATION_FAILED;
  if (!gl::make_current())
    return VK_ERROR_INITIALIZATION_FAILED;

  if (pCreateInfo->imageFormat != kSwapchainFormat) {
    KNIZ_ERR("swapchain format %d is not supported; this driver presents "
             "VK_FORMAT_R8G8B8A8_UNORM",
             static_cast<int>(pCreateInfo->imageFormat));
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  }
  if (pCreateInfo->imageColorSpace != kSwapchainColorSpace) {
    KNIZ_ERR("swapchain colour space %d is not supported; this driver presents "
             "VK_COLOR_SPACE_SRGB_NONLINEAR_KHR",
             static_cast<int>(pCreateInfo->imageColorSpace));
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  }
  if (pCreateInfo->imageExtent.width == 0 ||
      pCreateInfo->imageExtent.height == 0) {
    KNIZ_ERR("swapchain extent %ux%u is empty",
             pCreateInfo->imageExtent.width, pCreateInfo->imageExtent.height);
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (pCreateInfo->imageExtent.width > kMaxImageExtent ||
      pCreateInfo->imageExtent.height > kMaxImageExtent) {
    KNIZ_ERR("swapchain extent %ux%u exceeds the %u pixel maximum",
             pCreateInfo->imageExtent.width, pCreateInfo->imageExtent.height,
             kMaxImageExtent);
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (pCreateInfo->imageArrayLayers != 1) {
    KNIZ_ERR("swapchains with %u layers are not implemented; only 1 is",
             pCreateInfo->imageArrayLayers);
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if ((pCreateInfo->imageUsage &
       ~(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
         VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) != 0) {
    KNIZ_ERR("swapchain image usage 0x%x is not supported; colour attachment "
             "and transfer source only",
             static_cast<unsigned>(pCreateInfo->imageUsage));
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if (pCreateInfo->preTransform != VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) {
    KNIZ_ERR("swapchain pre-transform 0x%x is not supported; only identity is",
             static_cast<unsigned>(pCreateInfo->preTransform));
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if (pCreateInfo->compositeAlpha != VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) {
    KNIZ_ERR("composite alpha 0x%x is not supported; only opaque is",
             static_cast<unsigned>(pCreateInfo->compositeAlpha));
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if (pCreateInfo->presentMode != kPresentMode) {
    KNIZ_ERR("present mode %d is not supported; this driver presents FIFO",
             static_cast<int>(pCreateInfo->presentMode));
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if (pCreateInfo->minImageCount < 2) {
    KNIZ_ERR("a swapchain of %u image(s) cannot be presented; at least 2 are "
             "needed",
             pCreateInfo->minImageCount);
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (pCreateInfo->minImageCount > 64) {
    KNIZ_ERR("a swapchain of %u images is beyond anything Kniz will allocate",
             pCreateInfo->minImageCount);
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (!device->instance->gl.window_capable) {
    KNIZ_ERR("cannot create a swapchain: this OpenGL context has no window "
             "system to present to (it is a headless context, which is the "
             "normal case in a container)");
    return VK_ERROR_SURFACE_LOST_KHR;
  }
  if (pCreateInfo->oldSwapchain != VK_NULL_HANDLE) {
    // Vulkan retires the old swapchain when the new one is created but leaves
    // destroying it to the application.  Nothing is shared between the two
    // here, so "retiring" it means exactly nothing, which is worth saying.
    KNIZ_DBG("creating a swapchain to replace an existing one");
  }

  Swapchain* swapchain = nullptr;
  try {
    swapchain = new Swapchain();
  } catch (const std::bad_alloc&) {
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  }
  swapchain->device = device;
  swapchain->surface = surface;
  swapchain->format = pCreateInfo->imageFormat;
  swapchain->extent = pCreateInfo->imageExtent;
  swapchain->present_mode = pCreateInfo->presentMode;

  if (surface->window == nullptr) {
    surface->window = gl::create_window_surface(surface->native_window);
    if (surface->window == nullptr) {
      delete swapchain;
      return VK_ERROR_SURFACE_LOST_KHR;
    }
  }

  if (!create_present_state(swapchain)) {
    destroy_swapchain_objects(swapchain);
    delete swapchain;
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  swapchain->images.reserve(pCreateInfo->minImageCount);
  for (uint32_t i = 0; i < pCreateInfo->minImageCount; ++i) {
    Image* image = create_swapchain_image(device, pCreateInfo->imageExtent);
    if (image == nullptr) {
      KNIZ_ERR("could not create swapchain image %u", i);
      destroy_swapchain_objects(swapchain);
      delete swapchain;
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    swapchain->images.push_back(image);
  }

  *pSwapchain = to_handle(swapchain);
  KNIZ_INFO("swapchain created: %ux%u, %u image(s), FIFO, presented to window "
            "0x%llx",
            swapchain->extent.width, swapchain->extent.height,
            pCreateInfo->minImageCount,
            static_cast<unsigned long long>(surface->native_window));
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroySwapchainKHR(
    VkDevice deviceHandle, VkSwapchainKHR swapchainHandle,
    const VkAllocationCallbacks*) {
  Device* device = from_handle(deviceHandle);
  Swapchain* swapchain = from_handle(swapchainHandle);
  if (device == nullptr || swapchain == nullptr)
    return;

  if (gl::make_current())
    destroy_swapchain_objects(swapchain);

  swapchain->magic = 0;
  delete swapchain;
}

VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainImagesKHR(
    VkDevice deviceHandle, VkSwapchainKHR swapchainHandle,
    uint32_t* pSwapchainImageCount, VkImage* pSwapchainImages) {
  if (pSwapchainImageCount == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  Device* device = from_handle(deviceHandle);
  Swapchain* swapchain = from_handle(swapchainHandle);
  if (device == nullptr || swapchain == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  const uint32_t count = static_cast<uint32_t>(swapchain->images.size());
  const VkResult result =
      enumerate_count(count, pSwapchainImageCount, pSwapchainImages);
  if (result == VK_SUCCESS && pSwapchainImages != nullptr) {
    for (uint32_t i = 0; i < count; ++i)
      pSwapchainImages[i] = to_handle(swapchain->images[i]);
  }
  return result;
}

VKAPI_ATTR VkResult VKAPI_CALL AcquireNextImageKHR(
    VkDevice deviceHandle, VkSwapchainKHR swapchainHandle, uint64_t timeout,
    VkSemaphore semaphore, VkFence fenceHandle, uint32_t* pImageIndex) {
  if (pImageIndex == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  Device* device = from_handle(deviceHandle);
  Swapchain* swapchain = from_handle(swapchainHandle);
  if (device == nullptr || swapchain == nullptr || swapchain->images.empty())
    return VK_ERROR_INITIALIZATION_FAILED;

  // Kniz presents synchronously: by the time vkQueuePresentKHR returns, the
  // image has been drawn to the window and swapped, so nothing is in flight and
  // an image is always available.  That is why the timeout is irrelevant and
  // why the semaphore is ignored -- the same trade every other submission in
  // this driver makes.
  (void)timeout;
  (void)semaphore;

  if (fenceHandle != VK_NULL_HANDLE) {
    Fence* fence = from_handle(fenceHandle);
    if (fence == nullptr)
      return VK_ERROR_INITIALIZATION_FAILED;
    fence->signaled = true;
  }

  *pImageIndex = swapchain->next_image;
  swapchain->next_image =
      (swapchain->next_image + 1) % static_cast<uint32_t>(swapchain->images.size());
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL QueuePresentKHR(
    VkQueue queueHandle, const VkPresentInfoKHR* pPresentInfo) {
  Queue* queue = from_handle(queueHandle);
  if (queue == nullptr || pPresentInfo == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  Device* device = queue->device;
  if (pPresentInfo->swapchainCount > 0 && pPresentInfo->pSwapchains == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (!device_can_run_gl(device))
    return VK_ERROR_INITIALIZATION_FAILED;
  if (!gl::make_current())
    return VK_ERROR_INITIALIZATION_FAILED;

  // Wait semaphores are ignored, for the reason given in vkAcquireNextImageKHR:
  // the work they would wait on has already finished.
  VkResult first_error = VK_SUCCESS;
  for (uint32_t i = 0; i < pPresentInfo->swapchainCount; ++i) {
    Swapchain* swapchain = from_handle(pPresentInfo->pSwapchains[i]);
    VkResult result = VK_SUCCESS;

    if (swapchain == nullptr) {
      result = VK_ERROR_INITIALIZATION_FAILED;
    } else if (pPresentInfo->pImageIndices == nullptr) {
      result = VK_ERROR_INITIALIZATION_FAILED;
    } else if (!present_image(swapchain, pPresentInfo->pImageIndices[i])) {
      // A failed swap is a dead drawable far more often than a driver bug: the
      // window was destroyed, or the display connection went away.
      result = VK_ERROR_SURFACE_LOST_KHR;
    }

    if (pPresentInfo->pResults != nullptr)
      pPresentInfo->pResults[i] = result;
    if (result != VK_SUCCESS && first_error == VK_SUCCESS)
      first_error = result;
  }

  return first_error;
}

}  // namespace kniz
