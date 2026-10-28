// M2b test: drawing, end to end, through the real loader.
//
// Every milestone so far translated one Vulkan object into one OpenGL object: a
// buffer is a byte array on both sides, an image is a texture on both sides.
// Drawing is where that stops being true.  A Vulkan pipeline is an immutable
// object holding a dozen pieces of state that OpenGL spreads across the program,
// the vertex array object and the context itself, and a render pass describes
// work OpenGL has no concept of.  So this is the first milestone whose failure
// mode is *silently wrong pixels* rather than an obvious error: a triangle drawn
// upside down is still a triangle, and an image whose colours have been swapped
// between channels still looks like an image.
//
// The checks are therefore numeric and adversarial, in the same spirit as the
// compute and image tests:
//
//   1. a triangle whose three vertices carry red, green and blue is compared
//      against barycentric coordinates computed here, from the vertex positions
//      the buffer holds.  Interpolated colours exercise the whole path: the
//      attribute formats, the binding offsets, the varying locations, the
//      interpolation itself and the write to the attachment;
//   2. the pixel positions the triangle covers are checked individually, above
//      and below its middle, so a vertical flip shows up as a numeric
//      disagreement rather than as an image that still contains a triangle;
//   3. the same image is produced four more ways -- vkCmdDrawIndexed with binding
//      offsets and a negative base vertex, back-face culling with the winding
//      called counter-clockwise, a dynamic viewport, and a render area covering
//      half the framebuffer -- because each of those resolves the geometry
//      through a different piece of the translation;
//   4. a fragment shader that writes gl_FragCoord back out asks the rasteriser
//      where it thinks each fragment is, which is the one question no amount of
//      care on the input side can fake the answer to.  Its z channel is where
//      the clip space depth fixup becomes observable: the vertex shader writes
//      Vulkan depth 0.25 and 0.25 has to come back, where a missing rewrite of
//      gl_Position.z would show the same 0.25 read as 0.625 of GL's depth range.
//      A depth attachment is not needed to ask this question;
//   5. the colour write mask, which unlike a load-op clear *is* pipeline state:
//      a pipeline writing only red and green must leave blue and alpha as the
//      clear left them, and the pass after it must clear every channel anyway;
//   6. the refusals -- a second subpass, a depth attachment, blending, an integer
//      vertex format, a geometry stage, a pipeline used with the wrong render
//      pass -- are asserted to refuse rather than to approximate.

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "harness.h"

using namespace kniz_test;

namespace {

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 64;
constexpr uint32_t kBytesPerPixel = 4;
constexpr size_t kImageBytes = static_cast<size_t>(kWidth) * kHeight * kBytesPerPixel;

// Two clear colours, both given as k/255 so that the float handed to the render
// pass and the byte the attachment ends up holding are the same number.
constexpr uint8_t kClearBytes[4] = {17, 34, 51, 255};
constexpr uint8_t kPreviousBytes[4] = {200, 100, 50, 255};

VkClearValue clear_value(const uint8_t bytes[4]) {
  VkClearValue value{};
  value.color.float32[0] = bytes[0] / 255.0f;
  value.color.float32[1] = bytes[1] / 255.0f;
  value.color.float32[2] = bytes[2] / 255.0f;
  value.color.float32[3] = bytes[3] / 255.0f;
  return value;
}

// One vertex, as the test describes it.  The two attributes do not live in one
// interleaved buffer, though: the positions are in one buffer and the colours in
// another, with a stride each, so the replay path has to cope with attributes
// that come from different buffers with different layouts.
struct Vertex {
  float position[2];
  float color[4];
};

// The spacing between consecutive vertices in each of those buffers.
constexpr uint32_t kPositionStride = sizeof(float) * 2;
constexpr uint32_t kColorStride = sizeof(float) * 4;

// Each buffer begins with a padding element, so both bindings carry a non-zero
// offset.  The indices are {1, 2, 3} and the base vertex is -1, which cancels it.
constexpr uint64_t kPositionOffset = kPositionStride;
constexpr uint64_t kColorOffset = kColorStride;

// The triangle.  Its three vertices carry red, green and blue, so every interior
// texel has three different colour components and a channel mix-up cannot pass.
//
// The shape is also deliberately asymmetric about the horizontal axis: the flat
// edge is at NDC y = -1 and the apex at NDC y = +1.  Vulkan puts NDC y = -1 in
// row 0 of the image, so the flat edge belongs at the *top* of the readback and
// the apex at the bottom.  A flip anywhere in the chain moves the apex to the
// first row, where the per-texel checks below cannot miss it.
//
//   vertex 0  NDC (-1, -1)  red     framebuffer (0,    0)
//   vertex 1  NDC ( 1, -1)  green   framebuffer (64,   0)
//   vertex 2  NDC ( 0,  1)  blue    framebuffer (32,  64)
const Vertex kVertices[3] = {
    {{-1.0f, -1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{1.0f, -1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
    {{0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},
};

// A vertex index's position in the framebuffer, in texels.  This is the only
// place the test does that arithmetic, and reproducing it is part of what the
// driver is being asked to do.
void framebuffer_position(const Vertex& vertex, double* x, double* y) {
  *x = (static_cast<double>(vertex.position[0]) + 1.0) * 0.5 * kWidth;
  *y = (static_cast<double>(vertex.position[1]) + 1.0) * 0.5 * kHeight;
}

// Barycentric weights of a point in framebuffer coordinates.  y runs downwards,
// exactly as it does in the image the readback produces.
void barycentric(double px, double py, double* w0, double* w1, double* w2) {
  double x0 = 0, y0 = 0, x1 = 0, y1 = 0, x2 = 0, y2 = 0;
  framebuffer_position(kVertices[0], &x0, &y0);
  framebuffer_position(kVertices[1], &x1, &y1);
  framebuffer_position(kVertices[2], &x2, &y2);

  const double denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
  *w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denominator;
  *w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denominator;
  *w2 = 1.0 - *w0 - *w1;
}

bool inside_triangle(double px, double py) {
  double w0 = 0, w1 = 0, w2 = 0;
  barycentric(px, py, &w0, &w1, &w2);
  constexpr double kSlack = 1e-9;
  return w0 >= -kSlack && w1 >= -kSlack && w2 >= -kSlack;
}

// What the triangle's colour is at a texel centre: an independent computation
// from the vertex data, not a recording of what the driver produced.
void expected_triangle_color(uint32_t x, uint32_t y, uint8_t out[4]) {
  double w0 = 0, w1 = 0, w2 = 0;
  barycentric(x + 0.5, y + 0.5, &w0, &w1, &w2);
  for (int channel = 0; channel < 4; ++channel) {
    const double value = w0 * kVertices[0].color[channel] +
                         w1 * kVertices[1].color[channel] +
                         w2 * kVertices[2].color[channel];
    out[channel] = static_cast<uint8_t>(std::lround(value * 255.0));
  }
}

const uint8_t* texel(const std::vector<uint8_t>& bytes, uint32_t x, uint32_t y) {
  return bytes.data() + (static_cast<size_t>(y) * kWidth + x) * kBytesPerPixel;
}

std::string texel_text(const uint8_t* value) {
  char text[64];
  std::snprintf(text, sizeof text, "(%u, %u, %u, %u)", value[0], value[1], value[2],
                value[3]);
  return text;
}

// The unorm8 attachment rounds once, at the end, so a disagreement of one or two
// is arithmetic and a disagreement of more is a mistake.  Every check below is
// chosen so that the mistake it is looking for moves a channel by far more than
// that.
constexpr int kTolerance = 2;

// Barycentric interpolation happens in floating point on the GPU and is rounded
// to eight bits per channel here and once more in the attachment.  Two is the
// largest disagreement that can come out of that and still be ordinary rounding;
// a colour routed to the wrong channel, or interpolated with the wrong weights,
// is out by tens.
void check_color(const std::vector<uint8_t>& pixels, uint32_t x, uint32_t y,
                 const uint8_t expected[4], const char* what) {
  const uint8_t* actual = texel(pixels, x, y);
  for (int channel = 0; channel < 4; ++channel) {
    if (std::abs(static_cast<int>(actual[channel]) -
                 static_cast<int>(expected[channel])) <= kTolerance) {
      continue;
    }
    const std::string message =
        std::string(what) + ": texel (" + std::to_string(x) + ", " +
        std::to_string(y) + ") is " + texel_text(actual) + ", expected " +
        texel_text(expected);
    check(false, message.c_str());
    return;
  }
  check(true, what);
}

bool is_color(const std::vector<uint8_t>& pixels, uint32_t x, uint32_t y,
              const uint8_t expected[4]) {
  const uint8_t* actual = texel(pixels, x, y);
  for (int channel = 0; channel < 4; ++channel) {
    if (std::abs(static_cast<int>(actual[channel]) -
                 static_cast<int>(expected[channel])) > kTolerance) {
      return false;
    }
  }
  return true;
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
  std::printf("kniz draw test (M2b)\n");

  const char* shader_dir = argc > 1 ? argv[1] : ".";

  if (!icd_selected())
    return 2;

  void* loader = open_loader();
  if (loader == nullptr)
    return 2;
  if (!resolve_get_instance_proc_addr(loader))
    return 2;

  Runtime runtime;
  if (!bring_up(&runtime, "kniz-draw"))
    return 1;

  VkDevice device = runtime.device;
  VkQueue queue = runtime.queue;

  // ---- entry points -------------------------------------------------------
  auto create_buffer = device_proc<PFN_vkCreateBuffer>(device, "vkCreateBuffer");
  auto destroy_buffer = device_proc<PFN_vkDestroyBuffer>(device, "vkDestroyBuffer");
  auto get_buffer_requirements = device_proc<PFN_vkGetBufferMemoryRequirements>(
      device, "vkGetBufferMemoryRequirements");
  auto bind_buffer_memory =
      device_proc<PFN_vkBindBufferMemory>(device, "vkBindBufferMemory");
  auto allocate_memory = device_proc<PFN_vkAllocateMemory>(device, "vkAllocateMemory");
  auto free_memory = device_proc<PFN_vkFreeMemory>(device, "vkFreeMemory");
  auto map_memory = device_proc<PFN_vkMapMemory>(device, "vkMapMemory");
  auto unmap_memory = device_proc<PFN_vkUnmapMemory>(device, "vkUnmapMemory");

  auto create_image = device_proc<PFN_vkCreateImage>(device, "vkCreateImage");
  auto destroy_image = device_proc<PFN_vkDestroyImage>(device, "vkDestroyImage");
  auto get_image_requirements = device_proc<PFN_vkGetImageMemoryRequirements>(
      device, "vkGetImageMemoryRequirements");
  auto bind_image_memory =
      device_proc<PFN_vkBindImageMemory>(device, "vkBindImageMemory");
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

  auto cmd_begin_render_pass =
      device_proc<PFN_vkCmdBeginRenderPass>(device, "vkCmdBeginRenderPass");
  auto cmd_end_render_pass =
      device_proc<PFN_vkCmdEndRenderPass>(device, "vkCmdEndRenderPass");
  auto cmd_bind_pipeline =
      device_proc<PFN_vkCmdBindPipeline>(device, "vkCmdBindPipeline");
  auto cmd_bind_vertex_buffers =
      device_proc<PFN_vkCmdBindVertexBuffers>(device, "vkCmdBindVertexBuffers");
  auto cmd_bind_index_buffer =
      device_proc<PFN_vkCmdBindIndexBuffer>(device, "vkCmdBindIndexBuffer");
  auto cmd_set_viewport = device_proc<PFN_vkCmdSetViewport>(device, "vkCmdSetViewport");
  auto cmd_set_scissor = device_proc<PFN_vkCmdSetScissor>(device, "vkCmdSetScissor");
  auto cmd_draw = device_proc<PFN_vkCmdDraw>(device, "vkCmdDraw");
  auto cmd_draw_indexed = device_proc<PFN_vkCmdDrawIndexed>(device, "vkCmdDrawIndexed");
  auto cmd_clear_color_image =
      device_proc<PFN_vkCmdClearColorImage>(device, "vkCmdClearColorImage");
  auto cmd_copy_image_to_buffer =
      device_proc<PFN_vkCmdCopyImageToBuffer>(device, "vkCmdCopyImageToBuffer");
  auto cmd_pipeline_barrier =
      device_proc<PFN_vkCmdPipelineBarrier>(device, "vkCmdPipelineBarrier");

  auto queue_submit = device_proc<PFN_vkQueueSubmit>(device, "vkQueueSubmit");
  auto create_fence = device_proc<PFN_vkCreateFence>(device, "vkCreateFence");
  auto destroy_fence = device_proc<PFN_vkDestroyFence>(device, "vkDestroyFence");
  auto reset_fences = device_proc<PFN_vkResetFences>(device, "vkResetFences");
  auto wait_for_fences = device_proc<PFN_vkWaitForFences>(device, "vkWaitForFences");

  require("vkCreateGraphicsPipelines",
          reinterpret_cast<PFN_vkVoidFunction>(create_graphics_pipelines));
  require("vkCmdDraw", reinterpret_cast<PFN_vkVoidFunction>(cmd_draw));
  require("vkCmdDrawIndexed", reinterpret_cast<PFN_vkVoidFunction>(cmd_draw_indexed));
  require("vkCreateRenderPass",
          reinterpret_cast<PFN_vkVoidFunction>(create_render_pass));
  require("vkCreateFramebuffer",
          reinterpret_cast<PFN_vkVoidFunction>(create_framebuffer));
  require("vkCmdBeginRenderPass",
          reinterpret_cast<PFN_vkVoidFunction>(cmd_begin_render_pass));
  if (failures() > 0) {
    tear_down(&runtime);
    return 1;
  }

  // ---- the render target --------------------------------------------------
  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
  image_info.extent.width = kWidth;
  image_info.extent.height = kHeight;
  image_info.extent.depth = 1;
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory image_memory = VK_NULL_HANDLE;
  VkImageView image_view = VK_NULL_HANDLE;
  VkRenderPass render_pass = VK_NULL_HANDLE;
  VkRenderPass other_render_pass = VK_NULL_HANDLE;
  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer commands = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkBuffer readback = VK_NULL_HANDLE;
  VkDeviceMemory readback_memory = VK_NULL_HANDLE;
  VkBuffer positions = VK_NULL_HANDLE;
  VkDeviceMemory positions_memory = VK_NULL_HANDLE;
  VkBuffer colors = VK_NULL_HANDLE;
  VkDeviceMemory colors_memory = VK_NULL_HANDLE;
  VkBuffer indices = VK_NULL_HANDLE;
  VkDeviceMemory indices_memory = VK_NULL_HANDLE;
  VkShaderModule vertex_module = VK_NULL_HANDLE;
  VkShaderModule fragment_module = VK_NULL_HANDLE;
  VkShaderModule coord_vertex_module = VK_NULL_HANDLE;
  VkShaderModule coord_fragment_module = VK_NULL_HANDLE;

  // Allocates a host visible block and binds a buffer to it, returning the
  // mapping so the caller can write the buffer's initial contents.
  auto make_buffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer* buffer,
                         VkDeviceMemory* memory, void** mapped) -> bool {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (create_buffer(device, &info, nullptr, buffer) != VK_SUCCESS)
      return false;

    VkMemoryRequirements requirements{};
    get_buffer_requirements(device, *buffer, &requirements);

    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = runtime.host_memory_type;
    if (allocate_memory(device, &allocation, nullptr, memory) != VK_SUCCESS)
      return false;
    if (bind_buffer_memory(device, *buffer, *memory, 0) != VK_SUCCESS)
      return false;
    if (map_memory(device, *memory, 0, VK_WHOLE_SIZE, 0, mapped) != VK_SUCCESS)
      return false;
    return true;
  };

  // ---- objects ------------------------------------------------------------
  check(create_image(device, &image_info, nullptr, &image) == VK_SUCCESS,
        "vkCreateImage (the render target)");
  {
    VkMemoryRequirements requirements{};
    get_image_requirements(device, image, &requirements);
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = runtime.host_memory_type;
    check(allocate_memory(device, &allocation, nullptr, &image_memory) == VK_SUCCESS &&
              bind_image_memory(device, image, image_memory, 0) == VK_SUCCESS,
          "the render target has texture storage");
  }

  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
  view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  check(create_image_view(device, &view_info, nullptr, &image_view) == VK_SUCCESS,
        "vkCreateImageView (the render target)");

  {
    void* mapped = nullptr;
    check(make_buffer(kImageBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &readback,
                      &readback_memory, &mapped) &&
              mapped != nullptr,
          "the readback buffer is host visible");
    std::memset(mapped, 0, kImageBytes);
    unmap_memory(device, readback_memory);
  }

  // ---- the render pass ----------------------------------------------------
  //
  // One attachment, cleared on load, and a single subpass that writes it.  The
  // triangle needs nothing else.
  auto make_render_pass = [&](VkRenderPass* out,
                              const VkAttachmentDescription& attachment,
                              const VkSubpassDescription& subpass) -> VkResult {
    VkRenderPassCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = 1;
    info.pAttachments = &attachment;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    return create_render_pass(device, &info, nullptr, out);
  };

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  {
    VkAttachmentDescription attachment{};
    attachment.format = VK_FORMAT_R8G8B8A8_UNORM;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &reference;

    check(make_render_pass(&render_pass, attachment, subpass) == VK_SUCCESS,
          "vkCreateRenderPass");
    check(make_render_pass(&other_render_pass, attachment, subpass) == VK_SUCCESS,
          "a second, identical render pass (for the mismatch refusal)");
  }

  {
    VkFramebufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    info.renderPass = render_pass;
    info.attachmentCount = 1;
    info.pAttachments = &image_view;
    info.width = kWidth;
    info.height = kHeight;
    info.layers = 1;
    check(create_framebuffer(device, &info, nullptr, &framebuffer) == VK_SUCCESS,
          "vkCreateFramebuffer");
  }

  // ---- refusals that are properties of the render pass --------------------
  {
    VkAttachmentDescription attachment{};
    attachment.format = VK_FORMAT_R8G8B8A8_UNORM;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription two[2] = {subpass, subpass};
    VkRenderPassCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = 1;
    info.pAttachments = &attachment;
    info.subpassCount = 2;
    info.pSubpasses = two;
    VkRenderPass unused = VK_NULL_HANDLE;
    check(create_render_pass(device, &info, nullptr, &unused) ==
              VK_ERROR_FEATURE_NOT_PRESENT,
          "a render pass with two subpasses is refused");

    // A depth *format* in a colour slot is refused: GL would attach it where the
    // subpass does not draw, and every pixel would come back as whatever the depth
    // test wrote.  (A depth attachment in the depth slot arrived in M3b, and
    // tests/depth.cpp is where that lives.)
    VkAttachmentDescription depth_attachment = attachment;
    depth_attachment.format = VK_FORMAT_D16_UNORM;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.pAttachments = &depth_attachment;
    check(create_render_pass(device, &info, nullptr, &unused) ==
              VK_ERROR_INITIALIZATION_FAILED,
          "a depth format in a colour attachment is refused");

    VkAttachmentReference input{0, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkSubpassDescription with_input = subpass;
    with_input.inputAttachmentCount = 1;
    with_input.pInputAttachments = &input;
    info.pAttachments = &attachment;
    info.pSubpasses = &with_input;
    check(create_render_pass(device, &info, nullptr, &unused) ==
              VK_ERROR_FEATURE_NOT_PRESENT,
          "an input attachment is refused");

    // A load operation on an attachment the subpass never writes has no GL
    // equivalent, and is refused where the render pass is created rather than at
    // the first vkCmdBeginRenderPass that would have to perform it.
    VkSubpassDescription writes_nothing = subpass;
    writes_nothing.colorAttachmentCount = 0;
    writes_nothing.pColorAttachments = nullptr;
    info.pSubpasses = &writes_nothing;
    check(create_render_pass(device, &info, nullptr, &unused) ==
              VK_ERROR_FEATURE_NOT_PRESENT,
          "a subpass with no colour attachment is refused");
  }

  // ---- buffers ------------------------------------------------------------
  //
  // The positions and the colours live in two buffers, bound at two bindings,
  // and both are bound with a non-zero offset: element 0 of each is padding.
  // A vertex shader vertex index therefore reads element (index + 1), which is
  // what the offset is for.
  {
    void* mapped = nullptr;
    check(make_buffer(kPositionStride * 4, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      &positions, &positions_memory, &mapped),
          "the positions buffer");
    auto* floats = static_cast<float*>(mapped);
    for (int i = 0; i < 2; ++i)
      floats[i] = -99.0f;  // padding, never read
    for (int i = 0; i < 3; ++i) {
      floats[2 + i * 2 + 0] = kVertices[i].position[0];
      floats[2 + i * 2 + 1] = kVertices[i].position[1];
    }
    unmap_memory(device, positions_memory);

    check(make_buffer(kColorStride * 4, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &colors,
                      &colors_memory, &mapped),
          "the colours buffer");
    floats = static_cast<float*>(mapped);
    for (int i = 0; i < 4; ++i)
      floats[i] = -99.0f;
    for (int i = 0; i < 3; ++i) {
      for (int c = 0; c < 4; ++c)
        floats[4 + i * 4 + c] = kVertices[i].color[c];
    }
    unmap_memory(device, colors_memory);

    // Indices {1, 2, 3} with a vertexOffset of -1 select the same three
    // vertices as {0, 1, 2}, so the indexed phase must produce exactly the image
    // the direct draw did -- while exercising the base vertex arithmetic.
    check(make_buffer(sizeof(uint16_t) * 3, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &indices,
                      &indices_memory, &mapped),
          "the index buffer");
    auto* index_values = static_cast<uint16_t*>(mapped);
    index_values[0] = 1;
    index_values[1] = 2;
    index_values[2] = 3;
    unmap_memory(device, indices_memory);
  }

  // ---- shader modules -----------------------------------------------------
  auto make_module = [&](const char* name, VkShaderModule* module) -> bool {
    const std::string path = std::string(shader_dir) + "/" + name;
    const std::vector<uint32_t> words = read_spirv(path.c_str());
    if (words.empty())
      return false;
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = words.size() * sizeof(uint32_t);
    info.pCode = words.data();
    return create_shader_module(device, &info, nullptr, module) == VK_SUCCESS;
  };

  check(make_module("triangle.vert.spv", &vertex_module) &&
            make_module("triangle.frag.spv", &fragment_module),
        "the triangle shader modules");
  check(make_module("fragcoord.vert.spv", &coord_vertex_module) &&
            make_module("fragcoord.frag.spv", &coord_fragment_module),
        "the gl_FragCoord shader modules");

  {
    VkPipelineLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    check(create_pipeline_layout(device, &info, nullptr, &layout) == VK_SUCCESS,
          "vkCreatePipelineLayout (no descriptors)");
  }

  if (failures() > 0) {
    tear_down(&runtime);
    return 1;
  }

  // ---- graphics pipelines -------------------------------------------------
  struct PipelineDesc {
    VkShaderModule vertex = VK_NULL_HANDLE;
    VkShaderModule fragment = VK_NULL_HANDLE;
    bool vertex_input = true;
    VkCullModeFlags cull = VK_CULL_MODE_NONE;
    VkFrontFace front = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool dynamic_viewport = false;
    bool blending = false;
    bool geometry_stage = false;
    VkFormat attribute_format = VK_FORMAT_R32G32_SFLOAT;
    VkColorComponentFlags color_write = VK_COLOR_COMPONENT_R_BIT |
                                        VK_COLOR_COMPONENT_G_BIT |
                                        VK_COLOR_COMPONENT_B_BIT |
                                        VK_COLOR_COMPONENT_A_BIT;
    VkRenderPass pass = VK_NULL_HANDLE;
  };

  auto build_pipeline = [&](const PipelineDesc& desc, VkPipeline* out) -> VkResult {
    VkPipelineShaderStageCreateInfo stages[3]{};
    uint32_t stage_count = 0;
    stages[stage_count].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[stage_count].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[stage_count].module = desc.vertex;
    stages[stage_count].pName = "main";
    ++stage_count;
    stages[stage_count].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[stage_count].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[stage_count].module = desc.fragment;
    stages[stage_count].pName = "main";
    ++stage_count;
    if (desc.geometry_stage) {
      stages[stage_count].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stages[stage_count].stage = VK_SHADER_STAGE_GEOMETRY_BIT;
      stages[stage_count].module = desc.vertex;
      stages[stage_count].pName = "main";
      ++stage_count;
    }

    // Positions at binding 0 and colours at binding 1, each with a stride of its
    // own because they are separate buffers, and each with element 0 unwritten:
    // the offsets bound below are what skip it.
    VkVertexInputBindingDescription bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].stride = kPositionStride;
    bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindings[1].binding = 1;
    bindings[1].stride = kColorStride;
    bindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributes[2]{};
    attributes[0].location = 0;
    attributes[0].binding = 0;
    attributes[0].format = desc.attribute_format;
    attributes[0].offset = 0;
    attributes[1].location = 1;
    attributes[1].binding = 1;
    attributes[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributes[1].offset = 0;

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    if (desc.vertex_input) {
      vertex_input.vertexBindingDescriptionCount = 2;
      vertex_input.pVertexBindingDescriptions = bindings;
      vertex_input.vertexAttributeDescriptionCount = 2;
      vertex_input.pVertexAttributeDescriptions = attributes;
    }

    VkPipelineInputAssemblyStateCreateInfo assembly{};
    assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(kWidth);
    viewport.height = static_cast<float>(kHeight);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{{0, 0}, {kWidth, kHeight}};

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;
    if (!desc.dynamic_viewport) {
      viewport_state.pViewports = &viewport;
      viewport_state.pScissors = &scissor;
    }

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = desc.cull;
    raster.frontFace = desc.front;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.blendEnable = desc.blending ? VK_TRUE : VK_FALSE;
    blend_attachment.colorWriteMask = desc.color_write;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;

    VkDynamicState dynamic[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    if (desc.dynamic_viewport) {
      dynamic_state.dynamicStateCount = 2;
      dynamic_state.pDynamicStates = dynamic;
    }

    VkGraphicsPipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount = stage_count;
    info.pStages = stages;
    info.pVertexInputState = &vertex_input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport_state;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic_state;
    info.layout = layout;
    info.renderPass = desc.pass;

    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult result =
        create_graphics_pipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
    if (out != nullptr)
      *out = pipeline;
    return result;
  };

  VkPipeline triangle_pipeline = VK_NULL_HANDLE;
  VkPipeline indexed_pipeline = VK_NULL_HANDLE;
  VkPipeline cull_ccw_pipeline = VK_NULL_HANDLE;
  VkPipeline cull_cw_pipeline = VK_NULL_HANDLE;
  VkPipeline coord_pipeline = VK_NULL_HANDLE;
  VkPipeline red_green_pipeline = VK_NULL_HANDLE;

  {
    PipelineDesc desc;
    desc.vertex = vertex_module;
    desc.fragment = fragment_module;
    desc.pass = render_pass;
    check(build_pipeline(desc, &triangle_pipeline) == VK_SUCCESS,
          "vkCreateGraphicsPipelines (the triangle)");

    // Same state, but bound to the other render pass: used below to check that a
    // pipeline and a framebuffer that do not belong together are refused rather
    // than drawn.
    PipelineDesc indexed = desc;
    indexed.pass = other_render_pass;
    check(build_pipeline(indexed, &indexed_pipeline) == VK_SUCCESS,
          "vkCreateGraphicsPipelines (for the other render pass)");

    PipelineDesc ccw = desc;
    ccw.cull = VK_CULL_MODE_BACK_BIT;
    ccw.front = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    check(build_pipeline(ccw, &cull_ccw_pipeline) == VK_SUCCESS,
          "vkCreateGraphicsPipelines (culling back faces, counter-clockwise front)");

    PipelineDesc cw = desc;
    cw.cull = VK_CULL_MODE_BACK_BIT;
    cw.front = VK_FRONT_FACE_CLOCKWISE;
    check(build_pipeline(cw, &cull_cw_pipeline) == VK_SUCCESS,
          "vkCreateGraphicsPipelines (culling back faces, clockwise front)");

    PipelineDesc coord;
    coord.vertex = coord_vertex_module;
    coord.fragment = coord_fragment_module;
    coord.vertex_input = false;
    coord.dynamic_viewport = true;
    coord.pass = render_pass;
    check(build_pipeline(coord, &coord_pipeline) == VK_SUCCESS,
          "vkCreateGraphicsPipelines (gl_FragCoord, dynamic viewport, no attributes)");

    // Writes red and green but not blue, so the pass that uses it leaves the
    // colour write mask with a channel switched off.
    PipelineDesc masked = desc;
    masked.color_write =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
    check(build_pipeline(masked, &red_green_pipeline) == VK_SUCCESS,
          "vkCreateGraphicsPipelines (writing only red and green)");
  }

  // ---- refusals that are properties of the pipeline -----------------------
  {
    PipelineDesc desc;
    desc.vertex = vertex_module;
    desc.fragment = fragment_module;
    desc.pass = render_pass;
    VkPipeline unused = VK_NULL_HANDLE;

    PipelineDesc blended = desc;
    blended.blending = true;
    check(build_pipeline(blended, &unused) == VK_ERROR_FEATURE_NOT_PRESENT,
          "alpha blending is refused");

    PipelineDesc geometry = desc;
    geometry.geometry_stage = true;
    check(build_pipeline(geometry, &unused) == VK_ERROR_FEATURE_NOT_PRESENT,
          "a geometry stage is refused");

    PipelineDesc integer = desc;
    integer.attribute_format = VK_FORMAT_R32G32_SINT;
    check(build_pipeline(integer, &unused) == VK_ERROR_FORMAT_NOT_SUPPORTED,
          "an integer vertex format is refused");
  }

  // ---- command buffer and fence -------------------------------------------
  {
    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // The pool's command buffer is recorded, submitted and recorded again for
    // every phase of the test, and re-beginning an executable command buffer is
    // only allowed when its pool carries this flag.
    info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex = 0;
    check(create_command_pool(device, &info, nullptr, &pool) == VK_SUCCESS,
          "vkCreateCommandPool");

    VkCommandBufferAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocation.commandPool = pool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    check(allocate_command_buffers(device, &allocation, &commands) == VK_SUCCESS,
          "vkAllocateCommandBuffers");

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    check(create_fence(device, &fence_info, nullptr, &fence) == VK_SUCCESS,
          "vkCreateFence");
  }

  if (failures() > 0) {
    tear_down(&runtime);
    return 1;
  }

  // ---- one draw, submitted and read back ----------------------------------
  //
  // Every phase below goes through this: record a render pass and a draw, copy
  // the attachment into the readback buffer, submit, wait, and hand back the
  // image.  Recording, submitting and reading back in one place is what makes
  // the phases differ only in the thing each one is testing.
  struct DrawOptions {
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool vertex_buffers = true;
    bool indexed = false;
    bool dynamic_state = false;
    bool pre_clear = false;
    VkRect2D render_area{{0, 0}, {kWidth, kHeight}};
    VkClearValue clear{};
  };

  auto render = [&](const DrawOptions& options, std::vector<uint8_t>* out) -> bool {
    out->assign(kImageBytes, 0);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (begin_command_buffer(commands, &begin) != VK_SUCCESS)
      return false;

    if (options.pre_clear) {
      // Give the attachment known contents first, so that a phase which renders
      // into only part of it can tell "untouched" from "cleared".
      const VkClearValue previous = clear_value(kPreviousBytes);
      VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      cmd_clear_color_image(commands, image, VK_IMAGE_LAYOUT_GENERAL, &previous.color,
                            1, &range);

      VkImageMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      barrier.image = image;
      barrier.subresourceRange = range;
      cmd_pipeline_barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr,
                           0, nullptr, 1, &barrier);
    }

    VkRenderPassBeginInfo pass_begin{};
    pass_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    pass_begin.renderPass = render_pass;
    pass_begin.framebuffer = framebuffer;
    pass_begin.renderArea = options.render_area;
    pass_begin.clearValueCount = 1;
    pass_begin.pClearValues = &options.clear;
    cmd_begin_render_pass(commands, &pass_begin, VK_SUBPASS_CONTENTS_INLINE);

    cmd_bind_pipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, options.pipeline);

    if (options.vertex_buffers) {
      // Element 0 of each buffer is padding, so the bindings carry the offset of
      // one element -- the same offset the index buffer's {1, 2, 3} indices and
      // the -1 base vertex cancel out.
      VkBuffer buffers[2] = {positions, colors};
      VkDeviceSize offsets[2] = {kPositionOffset, kColorOffset};
      cmd_bind_vertex_buffers(commands, 0, 2, buffers, offsets);
    }
    if (options.indexed) {
      cmd_bind_index_buffer(commands, indices, 0, VK_INDEX_TYPE_UINT16);
    }
    if (options.dynamic_state) {
      VkViewport viewport{};
      viewport.width = static_cast<float>(kWidth);
      viewport.height = static_cast<float>(kHeight);
      viewport.maxDepth = 1.0f;
      cmd_set_viewport(commands, 0, 1, &viewport);
      VkRect2D scissor{{0, 0}, {kWidth, kHeight}};
      cmd_set_scissor(commands, 0, 1, &scissor);
    }

    if (options.indexed) {
      cmd_draw_indexed(commands, 3, 1, 0, -1, 0);
    } else {
      cmd_draw(commands, 3, 1, 0, 0);
    }

    cmd_end_render_pass(commands);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {kWidth, kHeight, 1};
    cmd_copy_image_to_buffer(commands, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            readback, 1, &region);

    if (end_command_buffer(commands) != VK_SUCCESS)
      return false;

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commands;
    if (reset_fences(device, 1, &fence) != VK_SUCCESS)
      return false;
    if (queue_submit(queue, 1, &submit, fence) != VK_SUCCESS)
      return false;
    if (wait_for_fences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
      return false;

    void* mapped = nullptr;
    if (map_memory(device, readback_memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)
      return false;
    std::memcpy(out->data(), mapped, kImageBytes);
    unmap_memory(device, readback_memory);
    return true;
  };

  // ---- phase 1: the triangle, with a static viewport and two vertex buffers
  std::vector<uint8_t> phase1_pixels;
  {
    DrawOptions options;
    options.pipeline = triangle_pipeline;
    options.clear = clear_value(kClearBytes);

    std::vector<uint8_t> pixels;
    check(render(options, &pixels), "phase 1 renders and reads back");
    phase1_pixels = pixels;

    // The triangle's three corners are at framebuffer (0, 0), (64, 0) and
    // (32, 64): the flat edge is in row 0 and the apex in the last row.  These
    // four probes are chosen so that a vertical flip swaps every one of them,
    // which is what makes the coverage checks below evidence rather than
    // decoration.
    check(inside_triangle(32.5, 4.5) && inside_triangle(32.5, 60.5) &&
              !inside_triangle(2.5, 60.5) && !inside_triangle(61.5, 60.5),
          "the reference triangle is wide at the top and narrow at the bottom");

    // Colour from an independent barycentric evaluation of the vertex data, not
    // from what the driver is expected to print.
    const uint32_t samples[][2] = {{32, 4}, {8, 8}, {56, 8}, {32, 30}, {32, 56}};
    for (const auto& sample : samples) {
      uint8_t expected[4];
      expected_triangle_color(sample[0], sample[1], expected);
      const std::string what = "phase 1: texel (" + std::to_string(sample[0]) +
                               ", " + std::to_string(sample[1]) +
                               ") matches barycentric interpolation";
      check_color(pixels, sample[0], sample[1], expected, what.c_str());
    }

    // Outside the triangle, the load operation's clear colour is what is left.
    const uint32_t outside[][2] = {{2, 60}, {61, 60}, {2, 55}, {61, 55}, {0, 63}};
    for (const auto& sample : outside) {
      const std::string what = "phase 1: texel (" + std::to_string(sample[0]) +
                               ", " + std::to_string(sample[1]) +
                               ") is the clear colour";
      check(is_color(pixels, sample[0], sample[1], kClearBytes), what.c_str());
    }
  }

  // ---- phase 2: the same triangle, indexed, with a negative base vertex -----
  //
  // Element 0 of each vertex buffer is padding and both bindings carry an offset
  // of one vertex; the indices are {1, 2, 3} and the base vertex is -1, so the
  // same three vertices come out.  Every one of those three pieces of arithmetic
  // moves the geometry if it is wrong, so agreeing with phase 1 texel for texel
  // is the check.
  {
    DrawOptions options;
    options.pipeline = triangle_pipeline;
    options.indexed = true;
    options.clear = clear_value(kClearBytes);

    std::vector<uint8_t> pixels;
    check(render(options, &pixels), "phase 2 renders and reads back");

    int differing = 0;
    int first_x = 0;
    int first_y = 0;
    for (uint32_t y = 0; y < kHeight; ++y) {
      for (uint32_t x = 0; x < kWidth; ++x) {
        const uint8_t* a = texel(phase1_pixels, x, y);
        const uint8_t* b = texel(pixels, x, y);
        for (int c = 0; c < 4; ++c) {
          if (std::abs(int(a[c]) - int(b[c])) > 1) {
            if (differing == 0) {
              first_x = static_cast<int>(x);
              first_y = static_cast<int>(y);
            }
            ++differing;
            break;
          }
        }
      }
    }

    char message[192];
    std::snprintf(message, sizeof message,
                  "phase 2: indexed drawing with binding offsets and a base "
                  "vertex reproduces phase 1 (%d texel(s) differ, first at (%d, "
                  "%d))",
                  differing, first_x, first_y);
    check(differing == 0, message);
  }

  // ---- phase 3: culling, checked from both sides --------------------------
  //
  // Vulkan measures a framebuffer from its top-left and OpenGL measures a
  // framebuffer object from its bottom-left, but the two agree about which row a
  // vertex at NDC y = -1 lands in, so neither the front face nor the winding
  // order needs flipping.  This is the check that would catch it if they did: the
  // triangle's signed area in framebuffer coordinates is positive, which both
  // APIs call counter-clockwise, so culling back faces with a counter-clockwise
  // front face must leave the image exactly as phase 1 drew it, and calling the
  // same winding clockwise must leave nothing but the clear.
  {
    auto render_to_pixels = [&](VkPipeline pipeline) -> std::vector<uint8_t> {
      DrawOptions options;
      options.pipeline = pipeline;
      options.clear = clear_value(kClearBytes);
      std::vector<uint8_t> pixels;
      if (!render(options, &pixels))
        return {};
      return pixels;
    };

    const std::vector<uint8_t> ccw = render_to_pixels(cull_ccw_pipeline);
    check(!ccw.empty(), "phase 3: the counter-clockwise draw renders and reads back");

    int differing = 0;
    int first_x = 0;
    int first_y = 0;
    if (!ccw.empty()) {
      for (uint32_t y = 0; y < kHeight; ++y) {
        for (uint32_t x = 0; x < kWidth; ++x) {
          const uint8_t* a = texel(phase1_pixels, x, y);
          const uint8_t* b = texel(ccw, x, y);
          for (int c = 0; c < 4; ++c) {
            if (std::abs(int(a[c]) - int(b[c])) > 1) {
              if (differing == 0) {
                first_x = static_cast<int>(x);
                first_y = static_cast<int>(y);
              }
              ++differing;
              break;
            }
          }
        }
      }
    }

    char message[224];
    std::snprintf(message, sizeof message,
                  "phase 3: culling back faces with a counter-clockwise front "
                  "face keeps the triangle that phase 1 drew (%d texel(s) "
                  "differ, first at (%d, %d))",
                  differing, first_x, first_y);
    check(!ccw.empty() && differing == 0, message);

    const std::vector<uint8_t> cw = render_to_pixels(cull_cw_pipeline);
    check(!cw.empty(), "phase 3: the clockwise draw renders and reads back");

    int left = 0;
    if (!cw.empty()) {
      for (uint32_t y = 0; y < kHeight; ++y) {
        for (uint32_t x = 0; x < kWidth; ++x) {
          if (!is_color(cw, x, y, kClearBytes))
            ++left;
        }
      }
    }
    std::snprintf(message, sizeof message,
                  "phase 3: calling that winding clockwise culls the triangle "
                  "entirely (%d texel(s) were drawn)",
                  left);
    check(!cw.empty() && left == 0, message);
  }

  // ---- phase 4: where the rasteriser thinks its fragments are -------------
  //
  // The fragment shader writes gl_FragCoord straight out, so this asks the
  // rasteriser a question about itself.  x and y are checked against the texel
  // they came from, which is what makes a flip anywhere in the chain visible;
  // z is the clip space conversion, which is the one thing no amount of care
  // with vertex data could confirm.
  {
    DrawOptions options;
    options.pipeline = coord_pipeline;
    options.vertex_buffers = false;
    options.dynamic_state = true;
    options.clear = clear_value(kClearBytes);

    std::vector<uint8_t> pixels;
    check(render(options, &pixels), "phase 4 renders and reads back");

    int mismatches = 0;
    int first_x = 0;
    int first_y = 0;
    uint8_t first_expected[4] = {0, 0, 0, 0};
    uint8_t first_actual[4] = {0, 0, 0, 0};
    for (uint32_t y = 0; y < kHeight; ++y) {
      for (uint32_t x = 0; x < kWidth; ++x) {
        uint8_t expected[4] = {
            static_cast<uint8_t>(std::lround((x + 0.5) / kWidth * 255.0)),
            static_cast<uint8_t>(std::lround((y + 0.5) / kHeight * 255.0)),
            // The vertex shader wrote depth 0.25 in Vulkan's clip space.  Without
            // SPIRV-Cross's fixup this channel would read 0.625.
            static_cast<uint8_t>(std::lround(0.25 * 255.0)), 255};
        const uint8_t* actual = texel(pixels, x, y);
        bool matches = true;
        for (int c = 0; c < 4; ++c) {
          if (std::abs(int(actual[c]) - int(expected[c])) > kTolerance)
            matches = false;
        }
        if (!matches) {
          if (mismatches == 0) {
            first_x = static_cast<int>(x);
            first_y = static_cast<int>(y);
            std::memcpy(first_expected, expected, 4);
            std::memcpy(first_actual, actual, 4);
          }
          ++mismatches;
        }
      }
    }

    char message[256];
    std::snprintf(message, sizeof message,
                  "phase 4: all %zu texels report their own gl_FragCoord (%d did "
                  "not; first at (%d, %d): got %s, expected %s)",
                  kImageBytes / kBytesPerPixel, mismatches, first_x, first_y,
                  texel_text(first_actual).c_str(),
                  texel_text(first_expected).c_str());
    check(mismatches == 0, message);

    // The full viewport is covered only if all three gl_VertexIndex values
    // produced their vertex, so this doubles as the check on that builtin.
    check(!is_color(pixels, 0, 0, kClearBytes),
          "phase 4: a triangle built from gl_VertexIndex alone covers the "
          "viewport");
  }

  // ---- phase 5: a render area smaller than the framebuffer ----------------
  //
  // Vulkan confines a render pass to its render area: neither the load op nor
  // any fragment may touch a texel outside it.  OpenGL's nearest equivalent is
  // the scissor rectangle, and the pipeline's own static scissor here is the
  // whole framebuffer, so the render area has to be intersected with it rather
  // than merely set at the start of the pass.
  {
    DrawOptions options;
    options.pipeline = triangle_pipeline;
    options.pre_clear = true;
    options.render_area = VkRect2D{{0, 0}, {kWidth / 2, kHeight}};
    options.clear = clear_value(kClearBytes);

    std::vector<uint8_t> pixels;
    check(render(options, &pixels), "phase 5 renders and reads back");

    check(!is_color(pixels, 20, 32, kClearBytes),
          "phase 5: the left half is drawn into");
    check(is_color(pixels, 40, 32, kPreviousBytes),
          "phase 5: the right half keeps what was there before, because it is "
          "outside the render area");
    check(is_color(pixels, 1, 8, kClearBytes),
          "phase 5: the load-op clear covers the render area and no more");
    check(is_color(pixels, 32, 32, kPreviousBytes),
          "phase 5: a fragment crossing the render area's edge is clipped, not "
          "merely scissored by the pipeline");

    // The last drawn column and the first one the render area excludes, one
    // texel apart, so a render area that is off by one is caught.
    uint8_t expected[4];
    expected_triangle_color(31, 32, expected);
    check(is_color(pixels, 31, 32, expected),
          "phase 5: the last column inside the render area is drawn");
  }

  // ---- phase 6: a pipeline and a framebuffer that do not belong together --
  //
  // Vulkan allows a pipeline to be used only with a compatible render pass.
  // Kniz refuses the draw at recording time, so the image must come back as
  // nothing but the clear colour -- which is an observable consequence, rather
  // than a shrug at a missing error path.
  {
    DrawOptions options;
    options.pipeline = indexed_pipeline;  // built for other_render_pass
    options.clear = clear_value(kClearBytes);

    std::vector<uint8_t> pixels;
    check(render(options, &pixels), "phase 6 renders and reads back");

    int drawn = 0;
    for (uint32_t y = 0; y < kHeight; ++y) {
      for (uint32_t x = 0; x < kWidth; ++x) {
        if (!is_color(pixels, x, y, kClearBytes))
          ++drawn;
      }
    }
    check(drawn == 0,
          "phase 6: a draw with a pipeline built for another render pass is "
          "refused, leaving only the clear");
  }

  // ---- phase 7: the colour write mask, and the clear that outlives it -------
  //
  // A Vulkan pipeline's colour write mask restricts what its own fragments write;
  // it must not restrict anything else, because the next pass's load-op clear is
  // not pipeline state at all.  OpenGL keeps the mask in the context alongside
  // every draw, so the two can be conflated.  Phase 7 is the check with teeth: it
  // draws with a mask that leaves blue and alpha alone, and fails if the mask is
  // ignored.  Phase 8 follows it and requires the next clear to have written every
  // channel — a property that holds on the drivers tested whether or not the clear
  // consults the mask, so it is recorded as intent rather than as proof.
  //
  // The two phases use different clear colours on purpose, so that a clear which
  // did inherit the stale mask would leave the *first* clear's blue behind and be
  // visible rather than coincidentally correct.
  {
    DrawOptions options;
    options.pipeline = red_green_pipeline;
    options.clear = clear_value(kPreviousBytes);

    std::vector<uint8_t> pixels;
    check(render(options, &pixels), "phase 7 renders and reads back");

    // Inside the triangle, where the write mask matters: the colour the mask
    // allows, and the clear's blue and alpha where it does not.
    uint8_t expected[4];
    expected_triangle_color(32, 30, expected);
    const uint8_t masked[4] = {expected[0], expected[1], kPreviousBytes[2],
                               kPreviousBytes[3]};
    check_color(pixels, 32, 30, masked,
                "phase 7: a pipeline writing only red and green leaves the "
                "clear's blue and alpha alone");
  }

  {
    DrawOptions options;
    options.pipeline = triangle_pipeline;
    options.clear = clear_value(kClearBytes);

    std::vector<uint8_t> pixels;
    check(render(options, &pixels), "phase 8 renders and reads back");

    check(is_color(pixels, 2, 60, kClearBytes),
          "phase 8: a load-op clear writes every channel even though the "
          "previous pass left the write mask with blue and alpha switched off");

    uint8_t expected[4];
    expected_triangle_color(32, 30, expected);
    check_color(pixels, 32, 30, expected,
                "phase 8: a pipeline with a full write mask still draws every "
                "channel");
  }

  // ---- teardown -----------------------------------------------------------
  if (destroy_pipeline != nullptr) {
    destroy_pipeline(device, triangle_pipeline, nullptr);
    destroy_pipeline(device, indexed_pipeline, nullptr);
    destroy_pipeline(device, cull_ccw_pipeline, nullptr);
    destroy_pipeline(device, cull_cw_pipeline, nullptr);
    destroy_pipeline(device, coord_pipeline, nullptr);
    destroy_pipeline(device, red_green_pipeline, nullptr);
  }
  if (destroy_shader_module != nullptr) {
    destroy_shader_module(device, vertex_module, nullptr);
    destroy_shader_module(device, fragment_module, nullptr);
    destroy_shader_module(device, coord_vertex_module, nullptr);
    destroy_shader_module(device, coord_fragment_module, nullptr);
  }
  if (destroy_pipeline_layout != nullptr)
    destroy_pipeline_layout(device, layout, nullptr);
  if (destroy_framebuffer != nullptr)
    destroy_framebuffer(device, framebuffer, nullptr);
  if (destroy_render_pass != nullptr) {
    destroy_render_pass(device, render_pass, nullptr);
    destroy_render_pass(device, other_render_pass, nullptr);
  }
  if (destroy_image_view != nullptr)
    destroy_image_view(device, image_view, nullptr);
  if (destroy_image != nullptr)
    destroy_image(device, image, nullptr);
  if (free_memory != nullptr) {
    free_memory(device, image_memory, nullptr);
    free_memory(device, readback_memory, nullptr);
    free_memory(device, positions_memory, nullptr);
    free_memory(device, colors_memory, nullptr);
    free_memory(device, indices_memory, nullptr);
  }
  if (destroy_buffer != nullptr) {
    destroy_buffer(device, readback, nullptr);
    destroy_buffer(device, positions, nullptr);
    destroy_buffer(device, colors, nullptr);
    destroy_buffer(device, indices, nullptr);
  }
  if (destroy_fence != nullptr)
    destroy_fence(device, fence, nullptr);
  if (destroy_command_pool != nullptr)
    destroy_command_pool(device, pool, nullptr);

  tear_down(&runtime);

  if (failures() > 0) {
    std::printf("FAILED: %d check(s)\n", failures());
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
