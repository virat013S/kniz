// M3b test: depth attachments, the depth test and depth writes.
//
// Everything before this milestone drew with the depth buffer switched off, so
// geometry was ordered by whatever the application happened to submit last.  A
// depth attachment is the first piece of state whose failure mode is invisible
// rather than loud: a scene drawn without a depth test still looks like a scene,
// just with the wrong surfaces in front.  So the checks here are a chain in which
// every phase's outcome depends on the buffer the previous phase left behind, and
// each link is chosen so that a specific plausible mistake flips it:
//
//   1. a pass that clears depth to 0.5 and draws at 0.75 must reject the draw --
//      if the clear used 1.0 instead of the value the render pass was given, the
//      draw lands, and the colour is red instead of the clear colour;
//   2. a draw at 0.25 must pass and write its own depth, so the test is not
//      simply rejecting everything;
//   3. a second pass with loadOp LOAD must still hold 0.25, so 0.75 is rejected
//      again -- a depth attachment that is quietly reallocated or re-cleared per
//      pass shows up here as red;
//   4. a nearer draw must pass against that persisted value;
//   5. a pass that clears again must put 0.5 back, which is checked in the depth
//      readback rather than in the colour: after phase 4 the buffer holds 0.1, and
//      0.9 is rejected whether the buffer is 0.1 or 0.5;
//   6. a pipeline with the test disabled and writes enabled must still write
//      depth, which is what Vulkan means and what a driver that maps
//      depthWriteEnable onto glDepthMask only when the test is on gets wrong;
//   7. a later pass must see that written value, so 0.7 is accepted against 0.9
//      and rejected against the 0.5 the clear would otherwise have left;
//   8. the same pass with VK_COMPARE_OP_GREATER must accept a farther draw, which
//      no amount of correct LESS handling can fake;
//   9. a pipeline with the test on and writes off must test but not write.
//
// The depth readback is the other half of the evidence: every phase checks the
// depth image itself, converted the same way a 16-bit unorm attachment converts,
// so "the test passed" and "the buffer holds what the test implies" are separate
// assertions rather than one.
//
// The refusals at the end are the other side of the same coin.  A depth image
// asked for as a sampled texture, a colour format in the depth slot, a stencil
// test, a depth bounds test: each is refused where it is created, with the reason
// the driver cannot honour it.

#include <vulkan/vulkan.h>

#include <cmath>
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

// R8G8B8A8_UNORM and D16_UNORM, so the readbacks are four bytes and two bytes per
// texel and every expected value below can be written as the number it is.
constexpr size_t kColourBytes =
    static_cast<size_t>(kWidth) * kHeight * 4;
constexpr size_t kDepthBytes = static_cast<size_t>(kWidth) * kHeight * 2;

// Colours given as k/255 so the float the pipeline is handed and the byte the
// attachment ends up holding are the same number.
constexpr uint8_t kBlack[4] = {0, 0, 0, 255};
constexpr uint8_t kRed[4] = {255, 0, 0, 255};
constexpr uint8_t kGreen[4] = {0, 255, 0, 255};
constexpr uint8_t kBlue[4] = {0, 0, 255, 255};
constexpr uint8_t kWhite[4] = {255, 255, 255, 255};

// The depths the phases use.  The buffer holds window depth in [0, 1] and a
// 16-bit unorm stores round(v * 65535), which is what unorm16() reproduces.
constexpr float kTenth = 0.1f;
constexpr float kQuarter = 0.25f;
constexpr float kHalf = 0.5f;
constexpr float kSevenTenths = 0.7f;
constexpr float kThreeQuarters = 0.75f;
constexpr float kNineTenths = 0.9f;

// The depth a fragment with this window depth leaves in a D16 attachment.
uint16_t unorm16(float value) {
  return static_cast<uint16_t>(std::lround(static_cast<double>(value) * 65535.0));
}

// The vertex shader's push constant block: a colour and the depth to draw it at.
struct Push {
  float colour[4];
  float z;
};
static_assert(sizeof(Push) == 20, "vec4 then float, under std430");

VkClearValue colour_clear(const uint8_t bytes[4]) {
  VkClearValue value{};
  value.color.float32[0] = bytes[0] / 255.0f;
  value.color.float32[1] = bytes[1] / 255.0f;
  value.color.float32[2] = bytes[2] / 255.0f;
  value.color.float32[3] = bytes[3] / 255.0f;
  return value;
}

VkClearValue depth_clear(float depth) {
  VkClearValue value{};
  value.depthStencil.depth = depth;
  value.depthStencil.stencil = 0;
  return value;
}

Push push_for(const uint8_t colour[4], float z) {
  Push push{};
  for (int i = 0; i < 4; ++i)
    push.colour[i] = colour[i] / 255.0f;
  push.z = z;
  return push;
}

std::string texel_text(const uint8_t* value) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "(%u, %u, %u, %u)", value[0], value[1],
                value[2], value[3]);
  return buffer;
}

bool is_colour(const std::vector<uint8_t>& pixels, uint32_t x, uint32_t y,
               const uint8_t expected[4]) {
  const size_t offset = (static_cast<size_t>(y) * kWidth + x) * 4;
  return std::memcmp(pixels.data() + offset, expected, 4) == 0;
}

void check_uniform_colour(const std::vector<uint8_t>& pixels,
                          const uint8_t expected[4], const char* what) {
  int differing = 0;
  uint32_t first_x = 0;
  uint32_t first_y = 0;
  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 0; x < kWidth; ++x) {
      if (is_colour(pixels, x, y, expected))
        continue;
      if (differing == 0) {
        first_x = x;
        first_y = y;
      }
      ++differing;
    }
  }
  if (differing == 0) {
    check(true, what);
    return;
  }
  const size_t offset = (static_cast<size_t>(first_y) * kWidth + first_x) * 4;
  const std::string detail =
      std::string(what) + ": " + std::to_string(differing) + " of " +
      std::to_string(kWidth * kHeight) + " texels differ, first at (" +
      std::to_string(first_x) + ", " + std::to_string(first_y) + ") which is " +
      texel_text(pixels.data() + offset) + " rather than " +
      texel_text(expected);
  check(false, detail.c_str());
}

// Every texel of the depth attachment, within a couple of counts of the value a
// 16-bit unorm conversion of `expected_depth` produces.
void check_uniform_depth(const std::vector<uint8_t>& depth, float expected_depth,
                         const char* what) {
  const uint16_t expected = unorm16(expected_depth);
  int differing = 0;
  uint32_t first_x = 0;
  uint32_t first_y = 0;
  uint16_t first_value = 0;
  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 0; x < kWidth; ++x) {
      const size_t offset = (static_cast<size_t>(y) * kWidth + x) * 2;
      uint16_t value = 0;
      std::memcpy(&value, depth.data() + offset, sizeof(value));
      const int difference = static_cast<int>(value) - static_cast<int>(expected);
      if (difference >= -2 && difference <= 2)
        continue;
      if (differing == 0) {
        first_x = x;
        first_y = y;
        first_value = value;
      }
      ++differing;
    }
  }
  if (differing == 0) {
    check(true, what);
    return;
  }
  const std::string detail =
      std::string(what) + ": " + std::to_string(differing) + " of " +
      std::to_string(kWidth * kHeight) + " texels differ, first at (" +
      std::to_string(first_x) + ", " + std::to_string(first_y) + ") which is " +
      std::to_string(first_value) + " rather than " + std::to_string(expected) +
      " (depth " + std::to_string(expected_depth) + ")";
  check(false, detail.c_str());
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
  std::printf("kniz depth test (M3b)\n");

  const char* shader_dir = argc > 1 ? argv[1] : ".";

  if (!icd_selected())
    return 2;

  void* loader = open_loader();
  if (loader == nullptr)
    return 2;
  if (!resolve_get_instance_proc_addr(loader))
    return 2;

  Runtime runtime;
  if (!bring_up(&runtime, "kniz-depth"))
    return 1;

  VkDevice device = runtime.device;
  VkQueue queue = runtime.queue;

  // ---- entry points -------------------------------------------------------
  auto get_format_properties = instance_proc<PFN_vkGetPhysicalDeviceFormatProperties>(
      runtime.instance, "vkGetPhysicalDeviceFormatProperties");
  auto get_image_format_properties =
      instance_proc<PFN_vkGetPhysicalDeviceImageFormatProperties>(
          runtime.instance, "vkGetPhysicalDeviceImageFormatProperties");
  require("vkGetPhysicalDeviceFormatProperties",
          reinterpret_cast<PFN_vkVoidFunction>(get_format_properties));
  require("vkGetPhysicalDeviceImageFormatProperties",
          reinterpret_cast<PFN_vkVoidFunction>(get_image_format_properties));

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
  auto allocate_memory = device_proc<PFN_vkAllocateMemory>(device, "vkAllocateMemory");
  auto free_memory = device_proc<PFN_vkFreeMemory>(device, "vkFreeMemory");
  auto map_memory = device_proc<PFN_vkMapMemory>(device, "vkMapMemory");
  auto unmap_memory = device_proc<PFN_vkUnmapMemory>(device, "vkUnmapMemory");

  auto create_buffer = device_proc<PFN_vkCreateBuffer>(device, "vkCreateBuffer");
  auto destroy_buffer = device_proc<PFN_vkDestroyBuffer>(device, "vkDestroyBuffer");
  auto get_buffer_requirements = device_proc<PFN_vkGetBufferMemoryRequirements>(
      device, "vkGetBufferMemoryRequirements");
  auto bind_buffer_memory =
      device_proc<PFN_vkBindBufferMemory>(device, "vkBindBufferMemory");

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
  auto cmd_push_constants =
      device_proc<PFN_vkCmdPushConstants>(device, "vkCmdPushConstants");
  auto cmd_draw = device_proc<PFN_vkCmdDraw>(device, "vkCmdDraw");
  auto cmd_copy_image_to_buffer =
      device_proc<PFN_vkCmdCopyImageToBuffer>(device, "vkCmdCopyImageToBuffer");

  auto queue_submit = device_proc<PFN_vkQueueSubmit>(device, "vkQueueSubmit");
  auto create_fence = device_proc<PFN_vkCreateFence>(device, "vkCreateFence");
  auto destroy_fence = device_proc<PFN_vkDestroyFence>(device, "vkDestroyFence");
  auto reset_fences = device_proc<PFN_vkResetFences>(device, "vkResetFences");
  auto wait_for_fences = device_proc<PFN_vkWaitForFences>(device, "vkWaitForFences");

  require("vkCreateImage", reinterpret_cast<PFN_vkVoidFunction>(create_image));
  require("vkCreateRenderPass",
          reinterpret_cast<PFN_vkVoidFunction>(create_render_pass));
  require("vkCreateFramebuffer",
          reinterpret_cast<PFN_vkVoidFunction>(create_framebuffer));
  require("vkCreateGraphicsPipelines",
          reinterpret_cast<PFN_vkVoidFunction>(create_graphics_pipelines));
  require("vkCmdDraw", reinterpret_cast<PFN_vkVoidFunction>(cmd_draw));
  require("vkCmdPushConstants",
          reinterpret_cast<PFN_vkVoidFunction>(cmd_push_constants));
  if (failures() > 0) {
    tear_down(&runtime);
    return 1;
  }

  // ---- the format report vkcube branches on --------------------------------
  //
  // This is not decoration.  Until M3b, vkGetPhysicalDeviceFormatProperties
  // answered zero for every format, and vkcube asks it whether R8G8B8A8_SRGB can
  // be sampled before deciding how to build its texture.  Told no, it takes the
  // linear-tiling path, which Kniz refuses, and never creates the optimal image
  // at all -- then presents a null image view three steps later.  The report has
  // to agree with what vkCreateImage accepts.
  {
    VkFormatProperties properties{};
    get_format_properties(runtime.physical, VK_FORMAT_R8G8B8A8_SRGB, &properties);
    const VkFormatFeatureFlags expected =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
        VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    check((properties.optimalTilingFeatures & expected) == expected,
          "a colour format is reported as sampleable, attachable and copyable");
    check(properties.linearTilingFeatures == 0,
          "linear tiling is reported as having no features, because it is refused");

    get_format_properties(runtime.physical, VK_FORMAT_D16_UNORM, &properties);
    check((properties.optimalTilingFeatures &
           VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0,
          "a depth format is reported as a depth attachment");
    check((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ==
              0,
          "a depth format is not reported as sampleable, because sampling one is "
          "refused");

    get_format_properties(runtime.physical, VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
                          &properties);
    check(properties.optimalTilingFeatures == 0 &&
              properties.linearTilingFeatures == 0 &&
              properties.bufferFeatures == 0,
          "a format with no OpenGL counterpart is reported as having no features");

    // vkGetPhysicalDeviceImageFormatProperties is the question vkcube actually
    // asks, and it has to answer the same way as vkCreateImage would.
    VkImageFormatProperties image_properties{};
    check(get_image_format_properties(
              runtime.physical, VK_FORMAT_D16_UNORM, VK_IMAGE_TYPE_2D,
              VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              0, &image_properties) == VK_SUCCESS,
          "a D16 image as a depth attachment is supported");
    check(get_image_format_properties(
              runtime.physical, VK_FORMAT_D16_UNORM, VK_IMAGE_TYPE_2D,
              VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT, 0,
              &image_properties) == VK_ERROR_FORMAT_NOT_SUPPORTED,
          "a D16 image as a sampled texture is refused by the query as well");
    check(get_image_format_properties(
              runtime.physical, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_2D,
              VK_IMAGE_TILING_OPTIMAL,
              VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 0,
              &image_properties) == VK_ERROR_FORMAT_NOT_SUPPORTED,
          "a colour image as a depth attachment is refused by the query as well");
  }

  // ---- the attachments ----------------------------------------------------
  VkImage colour_image = VK_NULL_HANDLE;
  VkDeviceMemory colour_memory = VK_NULL_HANDLE;
  VkImageView colour_view = VK_NULL_HANDLE;
  VkImage depth_image = VK_NULL_HANDLE;
  VkDeviceMemory depth_memory = VK_NULL_HANDLE;
  VkImageView depth_view = VK_NULL_HANDLE;

  // Allocates host visible memory and binds an image to it.
  auto bind_image = [&](VkImage image, VkDeviceMemory* memory) -> bool {
    VkMemoryRequirements requirements{};
    get_image_requirements(device, image, &requirements);
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = runtime.host_memory_type;
    return allocate_memory(device, &allocation, nullptr, memory) == VK_SUCCESS &&
           bind_image_memory(device, image, *memory, 0) == VK_SUCCESS;
  };

  auto make_image = [&](VkFormat format, VkImageUsageFlags usage, VkImage* image,
                        VkDeviceMemory* memory) -> bool {
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {kWidth, kHeight, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    return create_image(device, &info, nullptr, image) == VK_SUCCESS &&
           bind_image(*image, memory);
  };

  auto make_view = [&](VkImage image, VkFormat format, VkImageAspectFlags aspect,
                       VkImageView* view) -> bool {
    VkImageViewCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image = image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = format;
    info.subresourceRange = {aspect, 0, 1, 0, 1};
    return create_image_view(device, &info, nullptr, view) == VK_SUCCESS;
  };

  check(make_image(VK_FORMAT_R8G8B8A8_UNORM,
                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                   &colour_image, &colour_memory),
        "vkCreateImage (the colour attachment)");
  check(make_image(VK_FORMAT_D16_UNORM,
                   VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                   &depth_image, &depth_memory),
        "vkCreateImage (the depth attachment)");
  check(make_view(colour_image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT,
                  &colour_view),
        "vkCreateImageView (the colour attachment)");
  check(make_view(depth_image, VK_FORMAT_D16_UNORM, VK_IMAGE_ASPECT_DEPTH_BIT,
                  &depth_view),
        "vkCreateImageView (the depth attachment, with the depth aspect)");

  // ---- readback buffers ---------------------------------------------------
  VkBuffer colour_readback = VK_NULL_HANDLE;
  VkDeviceMemory colour_readback_memory = VK_NULL_HANDLE;
  VkBuffer depth_readback = VK_NULL_HANDLE;
  VkDeviceMemory depth_readback_memory = VK_NULL_HANDLE;

  auto make_buffer = [&](VkDeviceSize size, VkBuffer* buffer,
                         VkDeviceMemory* memory) -> bool {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (create_buffer(device, &info, nullptr, buffer) != VK_SUCCESS)
      return false;
    VkMemoryRequirements requirements{};
    get_buffer_requirements(device, *buffer, &requirements);
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = runtime.host_memory_type;
    return allocate_memory(device, &allocation, nullptr, memory) == VK_SUCCESS &&
           bind_buffer_memory(device, *buffer, *memory, 0) == VK_SUCCESS;
  };

  check(make_buffer(kColourBytes, &colour_readback, &colour_readback_memory) &&
            make_buffer(kDepthBytes, &depth_readback, &depth_readback_memory),
        "the readback buffers are host visible");

  // ---- the render passes --------------------------------------------------
  //
  // Two, identical but for the depth attachment's load operation.  Kniz requires
  // a pipeline to name the render pass it is used with -- Vulkan asks only for
  // compatibility, and DESIGN.md records the stricter rule -- so each pipeline
  // below is built against the pass it draws in.
  VkRenderPass clear_pass = VK_NULL_HANDLE;
  VkRenderPass load_pass = VK_NULL_HANDLE;
  VkFramebuffer clear_framebuffer = VK_NULL_HANDLE;
  VkFramebuffer load_framebuffer = VK_NULL_HANDLE;

  auto make_pass = [&](VkAttachmentLoadOp depth_load_op,
                       VkImageLayout depth_initial, VkRenderPass* out) -> VkResult {
    VkAttachmentDescription attachments[2]{};
    attachments[0].format = VK_FORMAT_R8G8B8A8_UNORM;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    attachments[1].format = VK_FORMAT_D16_UNORM;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = depth_load_op;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    // D16_UNORM has no stencil aspect, so Vulkan requires DONT_CARE here.
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = depth_initial;
    attachments[1].finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colour_reference{
        0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depth_reference{
        1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colour_reference;
    subpass.pDepthStencilAttachment = &depth_reference;

    VkRenderPassCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = 2;
    info.pAttachments = attachments;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    return create_render_pass(device, &info, nullptr, out);
  };

  check(make_pass(VK_ATTACHMENT_LOAD_OP_CLEAR, VK_IMAGE_LAYOUT_UNDEFINED,
                  &clear_pass) == VK_SUCCESS,
        "vkCreateRenderPass (colour and depth, both cleared)");
  check(make_pass(VK_ATTACHMENT_LOAD_OP_LOAD,
                  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                  &load_pass) == VK_SUCCESS,
        "vkCreateRenderPass (colour cleared, depth loaded)");

  auto make_framebuffer = [&](VkRenderPass pass, VkFramebuffer* out) -> VkResult {
    VkImageView attachments[2] = {colour_view, depth_view};
    VkFramebufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    info.renderPass = pass;
    info.attachmentCount = 2;
    info.pAttachments = attachments;
    info.width = kWidth;
    info.height = kHeight;
    info.layers = 1;
    return create_framebuffer(device, &info, nullptr, out);
  };

  check(make_framebuffer(clear_pass, &clear_framebuffer) == VK_SUCCESS,
        "vkCreateFramebuffer (colour and depth)");
  check(make_framebuffer(load_pass, &load_framebuffer) == VK_SUCCESS,
        "vkCreateFramebuffer (for the pass that loads depth)");

  // ---- refusals that are properties of the objects ------------------------
  {
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.extent = {kWidth, kHeight, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkImage unused = VK_NULL_HANDLE;
    info.format = VK_FORMAT_D16_UNORM;
    info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_SAMPLED_BIT;
    check(create_image(device, &info, nullptr, &unused) ==
              VK_ERROR_FEATURE_NOT_PRESENT,
          "a depth image that would be sampled is refused");

    info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    check(create_image(device, &info, nullptr, &unused) ==
              VK_ERROR_FEATURE_NOT_PRESENT,
          "a depth image that would also be a colour attachment is refused");

    info.format = VK_FORMAT_R8G8B8A8_UNORM;
    info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    check(create_image(device, &info, nullptr, &unused) ==
              VK_ERROR_FORMAT_NOT_SUPPORTED,
          "a colour format asked for depth attachment usage is refused");

    // A colour format in the subpass's depth slot.
    VkAttachmentDescription attachments[2]{};
    attachments[0].format = VK_FORMAT_R8G8B8A8_UNORM;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[1] = attachments[0];
    attachments[1].format = VK_FORMAT_D32_SFLOAT;

    VkAttachmentReference colour_reference{
        0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depth_reference{
        1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colour_reference;
    subpass.pDepthStencilAttachment = &depth_reference;

    VkRenderPassCreateInfo pass_info{};
    pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    pass_info.attachmentCount = 2;
    pass_info.pAttachments = attachments;
    pass_info.subpassCount = 1;
    pass_info.pSubpasses = &subpass;

    VkRenderPass unused_pass = VK_NULL_HANDLE;
    VkAttachmentDescription swapped[2] = {attachments[0], attachments[1]};
    swapped[1].format = VK_FORMAT_R8G8B8A8_UNORM;
    pass_info.pAttachments = swapped;
    check(create_render_pass(device, &pass_info, nullptr, &unused_pass) ==
              VK_ERROR_INITIALIZATION_FAILED,
          "a colour format in the depth slot is refused");

    // A depth attachment whose stencil aspect alone is cleared: GL clears the two
    // aspects of a packed format with one call.
    attachments[1].format = VK_FORMAT_D24_UNORM_S8_UINT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].initialLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    pass_info.pAttachments = attachments;
    check(create_render_pass(device, &pass_info, nullptr, &unused_pass) ==
              VK_ERROR_FEATURE_NOT_PRESENT,
          "clearing only the stencil aspect of a packed format is refused");

    // The colour image in the depth slot of the framebuffer: the usage bit is
    // what makes this wrong, not the format, so it is refused for the same reason
    // vkCreateImage would have been.
    VkImageView wrong[2] = {colour_view, colour_view};
    VkFramebufferCreateInfo framebuffer_info{};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = clear_pass;
    framebuffer_info.attachmentCount = 2;
    framebuffer_info.pAttachments = wrong;
    framebuffer_info.width = kWidth;
    framebuffer_info.height = kHeight;
    framebuffer_info.layers = 1;
    VkFramebuffer unused_framebuffer = VK_NULL_HANDLE;
    check(create_framebuffer(device, &framebuffer_info, nullptr,
                             &unused_framebuffer) ==
              VK_ERROR_INITIALIZATION_FAILED,
          "a colour image in the framebuffer's depth slot is refused");
  }

  // ---- shaders and layout -------------------------------------------------
  const std::string vertex_path = std::string(shader_dir) + "/depth.vert.spv";
  const std::string fragment_path = std::string(shader_dir) + "/depth.frag.spv";

  auto make_shader_module = [&](const std::string& path) -> VkShaderModule {
    const std::vector<uint32_t> words = read_spirv(path.c_str());
    if (words.empty())
      return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = words.size() * sizeof(uint32_t);
    info.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (create_shader_module(device, &info, nullptr, &module) != VK_SUCCESS)
      return VK_NULL_HANDLE;
    return module;
  };

  VkShaderModule vertex_module = make_shader_module(vertex_path);
  VkShaderModule fragment_module = make_shader_module(fragment_path);
  check(vertex_module != VK_NULL_HANDLE && fragment_module != VK_NULL_HANDLE,
        "the depth shader modules");

  VkPipelineLayout layout = VK_NULL_HANDLE;
  {
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    range.offset = 0;
    range.size = sizeof(Push);

    VkPipelineLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info.pushConstantRangeCount = 1;
    info.pPushConstantRanges = &range;
    check(create_pipeline_layout(device, &info, nullptr, &layout) == VK_SUCCESS,
          "vkCreatePipelineLayout (one push constant range, vertex stage)");
  }

  // ---- pipelines ----------------------------------------------------------
  struct PipelineDesc {
    VkRenderPass pass = VK_NULL_HANDLE;
    bool depth_test = true;
    bool depth_write = true;
    VkCompareOp compare = VK_COMPARE_OP_LESS;
  };

  auto build_pipeline = [&](const PipelineDesc& desc, VkPipeline* out) -> VkResult {
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertex_module;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragment_module;
    stages[1].pName = "main";

    // The vertex shader builds its geometry from gl_VertexIndex, so there is
    // nothing to bind and no attribute to describe.
    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo assembly{};
    assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.width = static_cast<float>(kWidth);
    viewport.height = static_cast<float>(kHeight);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{{0, 0}, {kWidth, kHeight}};

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = desc.depth_test ? VK_TRUE : VK_FALSE;
    depth.depthWriteEnable = desc.depth_write ? VK_TRUE : VK_FALSE;
    depth.depthCompareOp = desc.compare;
    depth.depthBoundsTestEnable = VK_FALSE;
    depth.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.blendEnable = VK_FALSE;
    blend_attachment.colorWriteMask = 0xF;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;

    VkGraphicsPipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertex_input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport_state;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &blend;
    info.layout = layout;
    info.renderPass = desc.pass;

    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult result =
        create_graphics_pipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                                  &pipeline);
    if (out != nullptr)
      *out = pipeline;
    return result;
  };

  VkPipeline clear_less = VK_NULL_HANDLE;
  VkPipeline clear_write_only = VK_NULL_HANDLE;
  VkPipeline load_less = VK_NULL_HANDLE;
  VkPipeline load_greater = VK_NULL_HANDLE;
  VkPipeline load_no_write = VK_NULL_HANDLE;

  {
    PipelineDesc desc;
    desc.pass = clear_pass;
    check(build_pipeline(desc, &clear_less) == VK_SUCCESS,
          "vkCreateGraphicsPipelines (depth test LESS, writing depth)");

    PipelineDesc write_only = desc;
    write_only.depth_test = false;
    check(build_pipeline(write_only, &clear_write_only) == VK_SUCCESS,
          "vkCreateGraphicsPipelines (no depth test, but writing depth)");

    PipelineDesc load = desc;
    load.pass = load_pass;
    check(build_pipeline(load, &load_less) == VK_SUCCESS,
          "vkCreateGraphicsPipelines (for the pass that loads depth)");

    PipelineDesc greater = load;
    greater.compare = VK_COMPARE_OP_GREATER;
    check(build_pipeline(greater, &load_greater) == VK_SUCCESS,
          "vkCreateGraphicsPipelines (depth test GREATER)");

    PipelineDesc no_write = load;
    no_write.depth_write = false;
    check(build_pipeline(no_write, &load_no_write) == VK_SUCCESS,
          "vkCreateGraphicsPipelines (depth test on, writing nothing)");
  }

  // ---- refusals that are properties of the pipeline -----------------------
  {
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertex_module;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragment_module;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo assembly{};
    assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport viewport{};
    viewport.width = static_cast<float>(kWidth);
    viewport.height = static_cast<float>(kHeight);
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{{0, 0}, {kWidth, kHeight}};
    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.pScissors = &scissor;
    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;

    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS;

    VkGraphicsPipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertex_input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport_state;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &blend;
    info.layout = layout;
    info.renderPass = clear_pass;

    VkPipeline unused = VK_NULL_HANDLE;

    // The stencil test is per-face operation triples in GL, and getting it wrong
    // writes to the stencil buffer silently, so it is refused until a test checks
    // it.  Everything else about this pipeline is one the driver accepts.
    depth.stencilTestEnable = VK_TRUE;
    check(create_graphics_pipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                                    &unused) == VK_ERROR_FEATURE_NOT_PRESENT,
          "the stencil test is refused");
    depth.stencilTestEnable = VK_FALSE;

    depth.depthBoundsTestEnable = VK_TRUE;
    check(create_graphics_pipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                                    &unused) == VK_ERROR_FEATURE_NOT_PRESENT,
          "the depth bounds test is refused");
    depth.depthBoundsTestEnable = VK_FALSE;

    // And the same state with both off is accepted, so the two refusals above are
    // about the features rather than about the pipeline being malformed.
    check(create_graphics_pipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                                    &unused) == VK_SUCCESS,
          "the same pipeline with stencil and bounds testing off is accepted");
    if (unused != VK_NULL_HANDLE)
      destroy_pipeline(device, unused, nullptr);
  }

  // ---- command recording --------------------------------------------------
  VkCommandPool pool = VK_NULL_HANDLE;
  {
    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex = 0;
    check(create_command_pool(device, &info, nullptr, &pool) == VK_SUCCESS,
          "vkCreateCommandPool");
  }

  VkCommandBuffer commands = VK_NULL_HANDLE;
  {
    VkCommandBufferAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool = pool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    check(allocate_command_buffers(device, &info, &commands) == VK_SUCCESS,
          "vkAllocateCommandBuffers");
  }

  VkFence fence = VK_NULL_HANDLE;
  {
    VkFenceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    check(create_fence(device, &info, nullptr, &fence) == VK_SUCCESS,
          "vkCreateFence");
  }

  // One phase: clear the colour to black, clear or load depth, draw the
  // full-screen triangle once, then copy both attachments back.
  struct Phase {
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    float clear_depth = 0.0f;
    const uint8_t* colour = kBlack;
    float z = 0.0f;
  };

  auto run = [&](const Phase& phase, std::vector<uint8_t>* out_colour,
                 std::vector<uint8_t>* out_depth) -> bool {
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (begin_command_buffer(commands, &begin) != VK_SUCCESS)
      return false;

    VkClearValue clears[2] = {colour_clear(kBlack), depth_clear(phase.clear_depth)};
    VkRenderPassBeginInfo pass_begin{};
    pass_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    pass_begin.renderPass =
        phase.framebuffer == clear_framebuffer ? clear_pass : load_pass;
    pass_begin.framebuffer = phase.framebuffer;
    pass_begin.renderArea = {{0, 0}, {kWidth, kHeight}};
    pass_begin.clearValueCount = 2;
    pass_begin.pClearValues = clears;
    cmd_begin_render_pass(commands, &pass_begin, VK_SUBPASS_CONTENTS_INLINE);

    cmd_bind_pipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, phase.pipeline);

    const Push push = push_for(phase.colour, phase.z);
    cmd_push_constants(commands, layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(push), &push);

    cmd_draw(commands, 3, 1, 0, 0);
    cmd_end_render_pass(commands);

    VkBufferImageCopy colour_region{};
    colour_region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    colour_region.imageExtent = {kWidth, kHeight, 1};
    cmd_copy_image_to_buffer(commands, colour_image,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             colour_readback, 1, &colour_region);

    VkBufferImageCopy depth_region{};
    depth_region.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    depth_region.imageExtent = {kWidth, kHeight, 1};
    cmd_copy_image_to_buffer(commands, depth_image,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             depth_readback, 1, &depth_region);

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

    out_colour->resize(kColourBytes);
    void* mapped = nullptr;
    if (map_memory(device, colour_readback_memory, 0, VK_WHOLE_SIZE, 0, &mapped) !=
        VK_SUCCESS)
      return false;
    std::memcpy(out_colour->data(), mapped, kColourBytes);
    unmap_memory(device, colour_readback_memory);

    out_depth->resize(kDepthBytes);
    mapped = nullptr;
    if (map_memory(device, depth_readback_memory, 0, VK_WHOLE_SIZE, 0, &mapped) !=
        VK_SUCCESS)
      return false;
    std::memcpy(out_depth->data(), mapped, kDepthBytes);
    unmap_memory(device, depth_readback_memory);
    return true;
  };

  // ---- phase 1: the clear uses the depth the render pass was given ---------
  {
    Phase phase;
    phase.framebuffer = clear_framebuffer;
    phase.pipeline = clear_less;
    phase.clear_depth = kHalf;
    phase.colour = kRed;
    phase.z = kThreeQuarters;

    std::vector<uint8_t> colour;
    std::vector<uint8_t> depth;
    check(run(phase, &colour, &depth), "phase 1 renders and reads back");
    check_uniform_colour(colour, kBlack,
                         "phase 1: 0.75 against a cleared 0.5 is rejected");
    check_uniform_depth(depth, kHalf,
                        "phase 1: the depth attachment holds the clear value 0.5");
  }

  // ---- phase 2: a nearer fragment passes and writes ------------------------
  {
    Phase phase;
    phase.framebuffer = clear_framebuffer;
    phase.pipeline = clear_less;
    phase.clear_depth = kHalf;
    phase.colour = kGreen;
    phase.z = kQuarter;

    std::vector<uint8_t> colour;
    std::vector<uint8_t> depth;
    check(run(phase, &colour, &depth), "phase 2 renders and reads back");
    check_uniform_colour(colour, kGreen,
                         "phase 2: 0.25 against a cleared 0.5 is drawn");
    check_uniform_depth(depth, kQuarter,
                        "phase 2: the drawn depth 0.25 replaced the clear value");
  }

  // ---- phase 3: the depth buffer survives a pass that loads it ------------
  {
    Phase phase;
    phase.framebuffer = load_framebuffer;
    phase.pipeline = load_less;
    phase.clear_depth = 1.0f;  // ignored: this pass loads instead of clearing
    phase.colour = kRed;
    phase.z = kThreeQuarters;

    std::vector<uint8_t> colour;
    std::vector<uint8_t> depth;
    check(run(phase, &colour, &depth), "phase 3 renders and reads back");
    check_uniform_colour(colour, kBlack,
                         "phase 3: 0.75 against the loaded 0.25 is rejected");
    check_uniform_depth(depth, kQuarter,
                        "phase 3: loading left the depth attachment at 0.25");
  }

  // ---- phase 4: a nearer fragment passes against the loaded value ---------
  {
    Phase phase;
    phase.framebuffer = load_framebuffer;
    phase.pipeline = load_less;
    phase.clear_depth = 1.0f;
    phase.colour = kWhite;
    phase.z = kTenth;

    std::vector<uint8_t> colour;
    std::vector<uint8_t> depth;
    check(run(phase, &colour, &depth), "phase 4 renders and reads back");
    check_uniform_colour(colour, kWhite,
                         "phase 4: 0.1 against the loaded 0.25 is drawn");
    check_uniform_depth(depth, kTenth,
                        "phase 4: the depth attachment now holds 0.1");
  }

  // ---- phase 5: clearing again puts the clear value back ------------------
  {
    Phase phase;
    phase.framebuffer = clear_framebuffer;
    phase.pipeline = clear_less;
    phase.clear_depth = kHalf;
    phase.colour = kBlue;
    phase.z = kNineTenths;

    std::vector<uint8_t> colour;
    std::vector<uint8_t> depth;
    check(run(phase, &colour, &depth), "phase 5 renders and reads back");
    check_uniform_colour(colour, kBlack,
                         "phase 5: 0.9 against a cleared 0.5 is rejected");
    check_uniform_depth(depth, kHalf,
                        "phase 5: the clear put 0.5 back over the 0.1 phase 4 "
                        "left behind");
  }

  // ---- phase 6: depth is written even with the test disabled -------------
  {
    Phase phase;
    phase.framebuffer = clear_framebuffer;
    phase.pipeline = clear_write_only;
    phase.clear_depth = kHalf;
    phase.colour = kBlue;
    phase.z = kNineTenths;

    std::vector<uint8_t> colour;
    std::vector<uint8_t> depth;
    check(run(phase, &colour, &depth), "phase 6 renders and reads back");
    check_uniform_colour(colour, kBlue,
                         "phase 6: with the test off the 0.9 draw lands");
    check_uniform_depth(depth, kNineTenths,
                        "phase 6: depthWriteEnable writes 0.9 with the test off");
  }

  // ---- phase 7: a later pass sees that written depth ---------------------
  {
    Phase phase;
    phase.framebuffer = load_framebuffer;
    phase.pipeline = load_less;
    phase.clear_depth = 1.0f;
    phase.colour = kRed;
    phase.z = kSevenTenths;

    std::vector<uint8_t> colour;
    std::vector<uint8_t> depth;
    check(run(phase, &colour, &depth), "phase 7 renders and reads back");
    check_uniform_colour(colour, kRed,
                         "phase 7: 0.7 against the written 0.9 is drawn");
    check_uniform_depth(depth, kSevenTenths,
                        "phase 7: the depth attachment now holds 0.7");
  }

  // ---- phase 8: the compare operation is honoured ------------------------
  {
    Phase phase;
    phase.framebuffer = load_framebuffer;
    phase.pipeline = load_greater;
    phase.clear_depth = 1.0f;
    phase.colour = kBlue;
    phase.z = kNineTenths;

    std::vector<uint8_t> colour;
    std::vector<uint8_t> depth;
    check(run(phase, &colour, &depth), "phase 8 renders and reads back");
    check_uniform_colour(colour, kBlue,
                         "phase 8: 0.9 passes VK_COMPARE_OP_GREATER against 0.7, "
                         "which LESS would have rejected");
    check_uniform_depth(depth, kNineTenths,
                        "phase 8: the greater draw wrote 0.9");
  }

  // ---- phase 9: the test runs but nothing is written ---------------------
  {
    Phase phase;
    phase.framebuffer = load_framebuffer;
    phase.pipeline = load_no_write;
    phase.clear_depth = 1.0f;
    phase.colour = kWhite;
    phase.z = kTenth;

    std::vector<uint8_t> colour;
    std::vector<uint8_t> depth;
    check(run(phase, &colour, &depth), "phase 9 renders and reads back");
    check_uniform_colour(colour, kWhite,
                         "phase 9: 0.1 passes the test against 0.9 and is drawn");
    check_uniform_depth(depth, kNineTenths,
                        "phase 9: with writes off the depth attachment still holds "
                        "0.9");
  }

  // ---- teardown -----------------------------------------------------------
  destroy_fence(device, fence, nullptr);
  destroy_command_pool(device, pool, nullptr);
  destroy_pipeline(device, clear_less, nullptr);
  destroy_pipeline(device, clear_write_only, nullptr);
  destroy_pipeline(device, load_less, nullptr);
  destroy_pipeline(device, load_greater, nullptr);
  destroy_pipeline(device, load_no_write, nullptr);
  destroy_pipeline_layout(device, layout, nullptr);
  destroy_shader_module(device, vertex_module, nullptr);
  destroy_shader_module(device, fragment_module, nullptr);
  destroy_framebuffer(device, clear_framebuffer, nullptr);
  destroy_framebuffer(device, load_framebuffer, nullptr);
  destroy_render_pass(device, clear_pass, nullptr);
  destroy_render_pass(device, load_pass, nullptr);
  destroy_buffer(device, colour_readback, nullptr);
  destroy_buffer(device, depth_readback, nullptr);
  free_memory(device, colour_readback_memory, nullptr);
  free_memory(device, depth_readback_memory, nullptr);
  destroy_image_view(device, colour_view, nullptr);
  destroy_image_view(device, depth_view, nullptr);
  destroy_image(device, colour_image, nullptr);
  destroy_image(device, depth_image, nullptr);
  free_memory(device, colour_memory, nullptr);
  free_memory(device, depth_memory, nullptr);

  tear_down(&runtime);

  if (failures() == 0)
    std::printf("all checks passed\n");
  return failures() == 0 ? 0 : 1;
}
