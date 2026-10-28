// M2d test: samplers and textures, in a compute and a graphics pipeline.
//
// A texture is the first descriptor whose *meaning* depends on state that lives
// outside the descriptor: the sampler.  Vulkan keeps them apart on purpose -- two
// samplers may read one image with different filters, and one sampler may read
// many images -- and GL has the same split in sampler objects, which is why the
// translation is possible at all.  What this test is built around is the ways it
// can still go wrong:
//
//   * The same image read through two samplers with different wrap modes.  If
//     Kniz had put the filter and wrap state on the texture, as a driver without
//     sampler objects must, the two reads would have to agree.
//
//   * Two images at the same binding number in different descriptor sets, which
//     GL cannot express: texture units are one flat namespace, so the mapping has
//     to number them the way it numbers uniform buffers.
//
//   * A separate image and sampler, which GLSL cannot express at all.  They have
//     to become one combined sampler, and the sampler object has to be bound to
//     the unit the *image* was given, because that is where the combined sampler
//     lives.
//
//   * texelFetch on a bare image, which needs a sampler in GLSL that Vulkan does
//     not require, so one has to be invented at a unit no descriptor uses.
//
//   * Filtering itself, in a graphics pipeline, where the sample position of a
//     pixel is fixed by the rasteriser rather than chosen by the test.
//
// Everything is compared against arithmetic done here, never against what the
// driver produced.

#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using namespace kniz_test;

constexpr uint32_t kTargetSize = 64;
constexpr uint32_t kSourceSize = 8;
constexpr uint32_t kSmallSize = 2;

// One invocation per texel of the 8x8 source image.
constexpr uint32_t kComputeCount = kSourceSize * kSourceSize;
constexpr size_t kResultBytes = kComputeCount * 4 * sizeof(float);
constexpr size_t kTargetBytes =
    static_cast<size_t>(kTargetSize) * kTargetSize * 4;
constexpr size_t kSourceBytes =
    static_cast<size_t>(kSourceSize) * kSourceSize * 4;
constexpr size_t kSmallBytes =
    static_cast<size_t>(kSmallSize) * kSmallSize * 4;

// Every buffer is host visible, and 256 bytes is one alignment unit on every
// driver Kniz has met.
constexpr VkDeviceSize kBufferSize = 256;

// The clear colour the target starts from.  Chosen so that it is not a colour
// any texel of the pattern can produce, which is what lets a refused draw be
// detected by reading the target back.
constexpr uint8_t kClear[4] = {17, 34, 51, 255};

// ---------------------------------------------------------------- the images

// Every texel distinct in every channel, so a wrong texel, a wrong row or a
// transposed read all produce a different number.
std::vector<uint8_t> source_pattern() {
  std::vector<uint8_t> bytes(kSourceBytes);
  for (uint32_t y = 0; y < kSourceSize; ++y) {
    for (uint32_t x = 0; x < kSourceSize; ++x) {
      uint8_t* pixel = bytes.data() + (static_cast<size_t>(y) * kSourceSize + x) * 4;
      pixel[0] = static_cast<uint8_t>(x * 32 + 8);
      pixel[1] = static_cast<uint8_t>(y * 32 + 8);
      pixel[2] = static_cast<uint8_t>((x ^ y) * 32 + 8);
      pixel[3] = 255;
    }
  }
  return bytes;
}

// Four distinct colours, one per corner, so a bilinear read in the middle is the
// average of all four and a nearest read is exactly one of them.
std::vector<uint8_t> small_pattern() {
  return {
      40,  0,   0,   255,  // (0, 0)
      0,   80,  0,   255,  // (1, 0)
      0,   0,   120, 255,  // (0, 1)
      60,  60,  0,   255,  // (1, 1)
  };
}

// ---------------------------------------------------------- reference maths

// One texel as a normalised float, with clamp to edge addressing: an index
// outside the image repeats the border texel, which is what a
// VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE sampler does.
void texel(const std::vector<uint8_t>& bytes, uint32_t size, int x, int y,
           double out[4]) {
  x = std::clamp(x, 0, static_cast<int>(size) - 1);
  y = std::clamp(y, 0, static_cast<int>(size) - 1);
  const uint8_t* pixel = bytes.data() + (static_cast<size_t>(y) * size + x) * 4;
  for (int channel = 0; channel < 4; ++channel)
    out[channel] = pixel[channel] / 255.0;
}

// A bilinear sample at a position in texel space, i.e. texel centres are at
// integers.  The four weights are exact binary fractions for every position this
// test asks for, so the result is exactly representable and the comparison below
// can be tight.
void bilinear(const std::vector<uint8_t>& bytes, uint32_t size, double px,
              double py, double out[4]) {
  const int x0 = static_cast<int>(std::floor(px));
  const int y0 = static_cast<int>(std::floor(py));
  const double fx = px - x0;
  const double fy = py - y0;

  double a[4], b[4], c[4], d[4];
  texel(bytes, size, x0, y0, a);
  texel(bytes, size, x0 + 1, y0, b);
  texel(bytes, size, x0, y0 + 1, c);
  texel(bytes, size, x0 + 1, y0 + 1, d);

  for (int channel = 0; channel < 4; ++channel) {
    const double top = a[channel] * (1.0 - fx) + b[channel] * fx;
    const double bottom = c[channel] * (1.0 - fx) + d[channel] * fx;
    out[channel] = top * (1.0 - fy) + bottom * fy;
  }
}

std::string vec_text(const double value[4]) {
  char text[128];
  std::snprintf(text, sizeof text, "(%.4f, %.4f, %.4f, %.4f)", value[0],
                value[1], value[2], value[3]);
  return text;
}

bool close(const double actual[4], const double expected[4], double tolerance) {
  for (int channel = 0; channel < 4; ++channel)
    if (std::fabs(actual[channel] - expected[channel]) > tolerance)
      return false;
  return true;
}

VkClearValue clear_value() {
  VkClearValue value{};
  for (int channel = 0; channel < 4; ++channel)
    value.color.float32[channel] = kClear[channel] / 255.0f;
  return value;
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
  std::printf("kniz sampler test (M2d)\n");

  const char* shader_dir = argc > 1 ? argv[1] : ".";

  if (!icd_selected())
    return 2;

  void* loader = open_loader();
  if (loader == nullptr)
    return 2;
  if (!resolve_get_instance_proc_addr(loader))
    return 2;

  Runtime runtime;
  if (!bring_up(&runtime, "kniz-sampler"))
    return 1;

  VkDevice device = runtime.device;
  VkQueue queue = runtime.queue;

  // ---- entry points --------------------------------------------------------
  auto create_buffer = device_proc<PFN_vkCreateBuffer>(device, "vkCreateBuffer");
  auto get_buffer_requirements = device_proc<PFN_vkGetBufferMemoryRequirements>(
      device, "vkGetBufferMemoryRequirements");
  auto bind_buffer_memory =
      device_proc<PFN_vkBindBufferMemory>(device, "vkBindBufferMemory");
  auto allocate_memory = device_proc<PFN_vkAllocateMemory>(device, "vkAllocateMemory");
  auto map_memory = device_proc<PFN_vkMapMemory>(device, "vkMapMemory");
  auto unmap_memory = device_proc<PFN_vkUnmapMemory>(device, "vkUnmapMemory");

  auto create_image = device_proc<PFN_vkCreateImage>(device, "vkCreateImage");
  auto get_image_requirements = device_proc<PFN_vkGetImageMemoryRequirements>(
      device, "vkGetImageMemoryRequirements");
  auto bind_image_memory =
      device_proc<PFN_vkBindImageMemory>(device, "vkBindImageMemory");
  auto create_image_view =
      device_proc<PFN_vkCreateImageView>(device, "vkCreateImageView");
  auto create_sampler = device_proc<PFN_vkCreateSampler>(device, "vkCreateSampler");

  auto create_render_pass =
      device_proc<PFN_vkCreateRenderPass>(device, "vkCreateRenderPass");
  auto create_framebuffer =
      device_proc<PFN_vkCreateFramebuffer>(device, "vkCreateFramebuffer");

  auto create_shader_module =
      device_proc<PFN_vkCreateShaderModule>(device, "vkCreateShaderModule");
  auto create_set_layout = device_proc<PFN_vkCreateDescriptorSetLayout>(
      device, "vkCreateDescriptorSetLayout");
  auto create_pipeline_layout =
      device_proc<PFN_vkCreatePipelineLayout>(device, "vkCreatePipelineLayout");
  auto create_compute_pipelines =
      device_proc<PFN_vkCreateComputePipelines>(device, "vkCreateComputePipelines");
  auto create_graphics_pipelines =
      device_proc<PFN_vkCreateGraphicsPipelines>(device, "vkCreateGraphicsPipelines");

  auto create_descriptor_pool =
      device_proc<PFN_vkCreateDescriptorPool>(device, "vkCreateDescriptorPool");
  auto allocate_descriptor_sets =
      device_proc<PFN_vkAllocateDescriptorSets>(device, "vkAllocateDescriptorSets");
  auto update_descriptor_sets =
      device_proc<PFN_vkUpdateDescriptorSets>(device, "vkUpdateDescriptorSets");

  auto create_command_pool =
      device_proc<PFN_vkCreateCommandPool>(device, "vkCreateCommandPool");
  auto allocate_command_buffers =
      device_proc<PFN_vkAllocateCommandBuffers>(device, "vkAllocateCommandBuffers");
  auto begin_command_buffer =
      device_proc<PFN_vkBeginCommandBuffer>(device, "vkBeginCommandBuffer");
  auto end_command_buffer =
      device_proc<PFN_vkEndCommandBuffer>(device, "vkEndCommandBuffer");

  auto cmd_pipeline_barrier =
      device_proc<PFN_vkCmdPipelineBarrier>(device, "vkCmdPipelineBarrier");
  auto cmd_copy_buffer_to_image =
      device_proc<PFN_vkCmdCopyBufferToImage>(device, "vkCmdCopyBufferToImage");
  auto cmd_copy_image_to_buffer =
      device_proc<PFN_vkCmdCopyImageToBuffer>(device, "vkCmdCopyImageToBuffer");

  auto cmd_begin_render_pass =
      device_proc<PFN_vkCmdBeginRenderPass>(device, "vkCmdBeginRenderPass");
  auto cmd_end_render_pass =
      device_proc<PFN_vkCmdEndRenderPass>(device, "vkCmdEndRenderPass");
  auto cmd_bind_pipeline =
      device_proc<PFN_vkCmdBindPipeline>(device, "vkCmdBindPipeline");
  auto cmd_bind_descriptor_sets =
      device_proc<PFN_vkCmdBindDescriptorSets>(device, "vkCmdBindDescriptorSets");
  auto cmd_draw = device_proc<PFN_vkCmdDraw>(device, "vkCmdDraw");
  auto cmd_dispatch = device_proc<PFN_vkCmdDispatch>(device, "vkCmdDispatch");

  auto queue_submit = device_proc<PFN_vkQueueSubmit>(device, "vkQueueSubmit");
  auto create_fence = device_proc<PFN_vkCreateFence>(device, "vkCreateFence");
  auto destroy_fence = device_proc<PFN_vkDestroyFence>(device, "vkDestroyFence");
  auto wait_for_fences = device_proc<PFN_vkWaitForFences>(device, "vkWaitForFences");

  require("vkCreateSampler", reinterpret_cast<PFN_vkVoidFunction>(create_sampler));
  require("vkCreateComputePipelines",
          reinterpret_cast<PFN_vkVoidFunction>(create_compute_pipelines));
  require("vkCreateGraphicsPipelines",
          reinterpret_cast<PFN_vkVoidFunction>(create_graphics_pipelines));
  require("vkCmdCopyBufferToImage",
          reinterpret_cast<PFN_vkVoidFunction>(cmd_copy_buffer_to_image));
  if (failures() > 0) {
    tear_down(&runtime);
    return 1;
  }

  // ---- buffers -------------------------------------------------------------
  auto make_buffer = [&](VkBufferUsageFlags usage, VkDeviceSize size,
                         VkBuffer* buffer, VkDeviceMemory* memory,
                         void** mapped) -> bool {
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
    return map_memory(device, *memory, 0, VK_WHOLE_SIZE, 0, mapped) == VK_SUCCESS;
  };

  VkBuffer source_staging = VK_NULL_HANDLE;
  VkDeviceMemory source_staging_memory = VK_NULL_HANDLE;
  VkBuffer small_staging = VK_NULL_HANDLE;
  VkDeviceMemory small_staging_memory = VK_NULL_HANDLE;
  VkBuffer result_buffer = VK_NULL_HANDLE;
  VkDeviceMemory result_memory = VK_NULL_HANDLE;
  VkBuffer readback = VK_NULL_HANDLE;
  VkDeviceMemory readback_memory = VK_NULL_HANDLE;

  const std::vector<uint8_t> source_bytes = source_pattern();
  const std::vector<uint8_t> small_bytes = small_pattern();

  void* mapped = nullptr;
  {
    check(make_buffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kBufferSize,
                      &source_staging, &source_staging_memory, &mapped),
          "the staging buffer for the 8x8 source image");
    std::memset(mapped, 0, kBufferSize);
    std::memcpy(mapped, source_bytes.data(), kSourceBytes);
    unmap_memory(device, source_staging_memory);
  }
  {
    check(make_buffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kBufferSize,
                      &small_staging, &small_staging_memory, &mapped),
          "the staging buffer for the 2x2 image");
    std::memset(mapped, 0, kBufferSize);
    std::memcpy(mapped, small_bytes.data(), kSmallBytes);
    unmap_memory(device, small_staging_memory);
  }
  {
    // The result buffer is as large as the 64 vec4 the shaders write, not one
    // alignment unit: a buffer that is too small loses the writes past its end,
    // and the readback below would then be reading a shorter block than it asked
    // for.
    check(make_buffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kResultBytes,
                      &result_buffer, &result_memory, &mapped),
          "the compute shaders' result buffer");
    std::memset(mapped, 0, kResultBytes);
    unmap_memory(device, result_memory);
  }
  {
    check(make_buffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT, kTargetBytes, &readback,
                      &readback_memory, &mapped),
          "the readback buffer for the render target");
    std::memset(mapped, 0, kTargetBytes);
    unmap_memory(device, readback_memory);
  }

  // ---- images --------------------------------------------------------------
  //
  // Three: the 8x8 source, a 2x2 image for the filtering check, and the 64x64
  // render target.  All are one mip level and one layer, which is what M2d
  // samples; a mipmapped or arrayed image is refused later in the test rather
  // than approximated.
  auto make_image = [&](uint32_t size, VkImageUsageFlags usage, VkImage* image,
                        VkDeviceMemory* memory, VkImageView* view) -> bool {
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_R8G8B8A8_UNORM;
    info.extent = {size, size, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (create_image(device, &info, nullptr, image) != VK_SUCCESS)
      return false;

    VkMemoryRequirements requirements{};
    get_image_requirements(device, *image, &requirements);
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = runtime.host_memory_type;
    if (allocate_memory(device, &allocation, nullptr, memory) != VK_SUCCESS)
      return false;
    if (bind_image_memory(device, *image, *memory, 0) != VK_SUCCESS)
      return false;

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = *image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return create_image_view(device, &view_info, nullptr, view) == VK_SUCCESS;
  };

  VkImage source_image = VK_NULL_HANDLE;
  VkDeviceMemory source_memory = VK_NULL_HANDLE;
  VkImageView source_view = VK_NULL_HANDLE;
  VkImage small_image = VK_NULL_HANDLE;
  VkDeviceMemory small_memory = VK_NULL_HANDLE;
  VkImageView small_view = VK_NULL_HANDLE;
  VkImage target_image = VK_NULL_HANDLE;
  VkDeviceMemory target_memory = VK_NULL_HANDLE;
  VkImageView target_view = VK_NULL_HANDLE;

  check(make_image(kSourceSize,
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                   &source_image, &source_memory, &source_view),
        "vkCreateImage (the 8x8 image, sampled)");
  check(make_image(kSmallSize,
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                   &small_image, &small_memory, &small_view),
        "vkCreateImage (a 2x2 image, for filtering)");
  check(make_image(kTargetSize,
                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                   &target_image, &target_memory, &target_view),
        "vkCreateImage (the render target)");

  // ---- the render pass -----------------------------------------------------
  VkRenderPass render_pass = VK_NULL_HANDLE;
  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  {
    VkAttachmentDescription attachment{};
    attachment.format = VK_FORMAT_R8G8B8A8_UNORM;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &reference;

    VkRenderPassCreateInfo pass_info{};
    pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    pass_info.attachmentCount = 1;
    pass_info.pAttachments = &attachment;
    pass_info.subpassCount = 1;
    pass_info.pSubpasses = &subpass;
    check(create_render_pass(device, &pass_info, nullptr, &render_pass) == VK_SUCCESS,
          "vkCreateRenderPass");

    VkFramebufferCreateInfo framebuffer_info{};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = render_pass;
    framebuffer_info.attachmentCount = 1;
    framebuffer_info.pAttachments = &target_view;
    framebuffer_info.width = kTargetSize;
    framebuffer_info.height = kTargetSize;
    framebuffer_info.layers = 1;
    check(create_framebuffer(device, &framebuffer_info, nullptr, &framebuffer) ==
              VK_SUCCESS,
          "vkCreateFramebuffer");
  }

  // ---- samplers ------------------------------------------------------------
  //
  // Four samplers, each a different piece of state:
  //
  //   nearest_clamp   NEAREST, CLAMP_TO_EDGE   the exact-texel baseline
  //   nearest_repeat  NEAREST, REPEAT          the same image, different wrap
  //   linear_clamp    LINEAR,  CLAMP_TO_EDGE   the filtering case
  //   border          NEAREST, CLAMP_TO_BORDER a border colour, never sampled
  //
  // The first three are used by the phases below.  The fourth exists so that
  // every address mode Kniz claims is exercised at least once through the
  // sampler object, including the one that needs a border colour uploaded.
  auto make_sampler = [&](VkFilter filter, VkSamplerMipmapMode mipmap,
                          VkSamplerAddressMode address,
                          VkBorderColor border, VkSampler* out) -> VkResult {
    VkSamplerCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = filter;
    info.minFilter = filter;
    info.mipmapMode = mipmap;
    info.addressModeU = address;
    info.addressModeV = address;
    info.addressModeW = address;
    info.minLod = 0.0f;
    info.maxLod = 0.25f;
    info.borderColor = border;
    return create_sampler(device, &info, nullptr, out);
  };

  VkSampler nearest_clamp = VK_NULL_HANDLE;
  VkSampler nearest_repeat = VK_NULL_HANDLE;
  VkSampler linear_clamp = VK_NULL_HANDLE;
  VkSampler border = VK_NULL_HANDLE;

  check(make_sampler(VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
                     VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                     VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
                     &nearest_clamp) == VK_SUCCESS,
        "vkCreateSampler (NEAREST, clamp to edge)");
  check(make_sampler(VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
                     VK_SAMPLER_ADDRESS_MODE_REPEAT,
                     VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
                     &nearest_repeat) == VK_SUCCESS,
        "vkCreateSampler (NEAREST, repeat)");
  check(make_sampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST,
                     VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                     VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
                     &linear_clamp) == VK_SUCCESS,
        "vkCreateSampler (LINEAR, clamp to edge)");
  check(make_sampler(VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
                     VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                     VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE, &border) == VK_SUCCESS,
        "vkCreateSampler (clamp to border, opaque white)");

  // ---- descriptor set layouts ---------------------------------------------
  //
  // The compute pipeline's set 0 holds the two samplers over one image and the
  // result buffer, and set 1 holds the 2x2 image at binding 0 -- the same binding
  // number as the source image in set 0.
  auto make_set_layout = [&](const VkDescriptorSetLayoutBinding* bindings,
                             uint32_t count,
                             VkDescriptorSetLayout* out) -> VkResult {
    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = count;
    info.pBindings = bindings;
    return create_set_layout(device, &info, nullptr, out);
  };

  auto single_binding = [](uint32_t binding, VkDescriptorType type,
                           VkShaderStageFlags stages) {
    VkDescriptorSetLayoutBinding result{};
    result.binding = binding;
    result.descriptorType = type;
    result.descriptorCount = 1;
    result.stageFlags = stages;
    return result;
  };

  VkDescriptorSetLayout combined_set_layout = VK_NULL_HANDLE;
  {
    VkDescriptorSetLayoutBinding bindings[3] = {
        single_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                       VK_SHADER_STAGE_COMPUTE_BIT),
        single_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                       VK_SHADER_STAGE_COMPUTE_BIT),
        single_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                       VK_SHADER_STAGE_COMPUTE_BIT),
    };
    check(make_set_layout(bindings, 3, &combined_set_layout) == VK_SUCCESS,
          "a set layout with two combined image samplers and a storage buffer");
  }

  VkDescriptorSetLayout second_set_layout = VK_NULL_HANDLE;
  {
    const VkDescriptorSetLayoutBinding binding =
        single_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                       VK_SHADER_STAGE_COMPUTE_BIT);
    check(make_set_layout(&binding, 1, &second_set_layout) == VK_SUCCESS,
          "a second set with a combined image sampler at the same binding "
          "number");
  }

  VkDescriptorSetLayout separate_set_layout = VK_NULL_HANDLE;
  {
    VkDescriptorSetLayoutBinding bindings[3] = {
        single_binding(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                       VK_SHADER_STAGE_COMPUTE_BIT),
        single_binding(1, VK_DESCRIPTOR_TYPE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT),
        single_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                       VK_SHADER_STAGE_COMPUTE_BIT),
    };
    check(make_set_layout(bindings, 3, &separate_set_layout) == VK_SUCCESS,
          "a set layout with a separate image, a separate sampler and a "
          "storage buffer");
  }

  VkDescriptorSetLayout graphics_set_layout = VK_NULL_HANDLE;
  {
    const VkDescriptorSetLayoutBinding binding =
        single_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                       VK_SHADER_STAGE_FRAGMENT_BIT);
    check(make_set_layout(&binding, 1, &graphics_set_layout) == VK_SUCCESS,
          "a set layout with one combined image sampler for the fragment stage");
  }

  // ---- pipeline layouts ----------------------------------------------------
  VkPipelineLayout combined_layout = VK_NULL_HANDLE;
  {
    const VkDescriptorSetLayout sets[2] = {combined_set_layout, second_set_layout};
    VkPipelineLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info.setLayoutCount = 2;
    info.pSetLayouts = sets;
    check(create_pipeline_layout(device, &info, nullptr, &combined_layout) ==
              VK_SUCCESS,
          "vkCreatePipelineLayout (two sets of combined image samplers)");
  }

  VkPipelineLayout separate_layout = VK_NULL_HANDLE;
  {
    const VkDescriptorSetLayout sets[1] = {separate_set_layout};
    VkPipelineLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info.setLayoutCount = 1;
    info.pSetLayouts = sets;
    check(create_pipeline_layout(device, &info, nullptr, &separate_layout) ==
              VK_SUCCESS,
          "vkCreatePipelineLayout (a separate image and sampler)");
  }

  VkPipelineLayout graphics_layout = VK_NULL_HANDLE;
  {
    const VkDescriptorSetLayout sets[1] = {graphics_set_layout};
    VkPipelineLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info.setLayoutCount = 1;
    info.pSetLayouts = sets;
    check(create_pipeline_layout(device, &info, nullptr, &graphics_layout) ==
              VK_SUCCESS,
          "vkCreatePipelineLayout (one set of combined image samplers)");
  }

  // ---- shader modules ------------------------------------------------------
  auto make_module = [&](const char* name, VkShaderModule* out) -> bool {
    const std::string path = std::string(shader_dir) + "/" + name;
    const std::vector<uint32_t> spirv = read_spirv(path.c_str());
    if (spirv.empty())
      return false;
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = spirv.size() * sizeof(uint32_t);
    info.pCode = spirv.data();
    return create_shader_module(device, &info, nullptr, out) == VK_SUCCESS;
  };

  VkShaderModule sample_module = VK_NULL_HANDLE;
  VkShaderModule separate_module = VK_NULL_HANDLE;
  VkShaderModule vertex_module = VK_NULL_HANDLE;
  VkShaderModule fragment_module = VK_NULL_HANDLE;
  check(make_module("texture_sample.comp.spv", &sample_module),
        "the combined image sampler compute shader");
  check(make_module("texture_separate.comp.spv", &separate_module),
        "the separate image and sampler compute shader");
  check(make_module("texture.vert.spv", &vertex_module) &&
            make_module("texture.frag.spv", &fragment_module),
        "the textured triangle shader modules");

  // ---- pipelines -----------------------------------------------------------
  auto build_compute = [&](VkShaderModule module, VkPipelineLayout layout,
                           VkPipeline* out) -> VkResult {
    VkComputePipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = layout;
    return create_compute_pipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, out);
  };

  VkPipeline sample_pipeline = VK_NULL_HANDLE;
  check(build_compute(sample_module, combined_layout, &sample_pipeline) ==
            VK_SUCCESS,
        "vkCreateComputePipelines (combined image samplers in two sets)");

  VkPipeline separate_pipeline = VK_NULL_HANDLE;
  check(build_compute(separate_module, separate_layout, &separate_pipeline) ==
            VK_SUCCESS,
        "vkCreateComputePipelines (a separate image, a separate sampler and a "
        "texelFetch)");

  auto build_graphics = [&](VkPipelineLayout layout, VkPipeline* out) -> VkResult {
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
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo assembly{};
    assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.width = static_cast<float>(kTargetSize);
    viewport.height = static_cast<float>(kTargetSize);
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{{0, 0}, {kTargetSize, kTargetSize}};

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
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                      VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT |
                                      VK_COLOR_COMPONENT_A_BIT;
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
    info.pColorBlendState = &blend;
    info.layout = layout;
    info.renderPass = render_pass;
    info.subpass = 0;
    return create_graphics_pipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, out);
  };

  VkPipeline graphics_pipeline = VK_NULL_HANDLE;
  check(build_graphics(graphics_layout, &graphics_pipeline) == VK_SUCCESS,
        "vkCreateGraphicsPipelines (a textured triangle)");

  if (failures() > 0) {
    tear_down(&runtime);
    return 1;
  }

  // ---- descriptor pool and sets -------------------------------------------
  VkDescriptorPool pool = VK_NULL_HANDLE;
  {
    VkDescriptorPoolSize sizes[3]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = 8;
    sizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    sizes[1].descriptorCount = 2;
    sizes[2].type = VK_DESCRIPTOR_TYPE_SAMPLER;
    sizes[2].descriptorCount = 2;

    VkDescriptorPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.maxSets = 12;
    info.poolSizeCount = 3;
    info.pPoolSizes = sizes;
    check(create_descriptor_pool(device, &info, nullptr, &pool) == VK_SUCCESS,
          "vkCreateDescriptorPool");
  }

  VkDescriptorSet compute_sets[3] = {};
  {
    const VkDescriptorSetLayout layouts[3] = {combined_set_layout,
                                              second_set_layout,
                                              separate_set_layout};
    VkDescriptorSetAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    info.descriptorPool = pool;
    info.descriptorSetCount = 3;
    info.pSetLayouts = layouts;
    check(allocate_descriptor_sets(device, &info, compute_sets) == VK_SUCCESS,
          "three descriptor sets for the compute pipelines");
  }

  VkDescriptorSet graphics_sets[2] = {};
  {
    const VkDescriptorSetLayout layouts[2] = {graphics_set_layout,
                                              graphics_set_layout};
    VkDescriptorSetAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    info.descriptorPool = pool;
    info.descriptorSetCount = 2;
    info.pSetLayouts = layouts;
    check(allocate_descriptor_sets(device, &info, graphics_sets) == VK_SUCCESS,
          "two descriptor sets for the graphics pipeline");
  }

  // ---- writing the descriptors --------------------------------------------
  auto write_combined = [&](VkDescriptorSet set, uint32_t binding,
                            VkImageView view, VkSampler sampler) {
    VkDescriptorImageInfo image{};
    image.sampler = sampler;
    image.imageView = view;
    image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &image;
    update_descriptor_sets(device, 1, &write, 0, nullptr);
  };

  auto write_buffer = [&](VkDescriptorSet set, uint32_t binding,
                          VkDescriptorType type, VkBuffer buffer) {
    VkDescriptorBufferInfo info{};
    info.buffer = buffer;
    info.offset = 0;
    info.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pBufferInfo = &info;
    update_descriptor_sets(device, 1, &write, 0, nullptr);
  };

  // Set 0 of the combined pipeline: the source image through two samplers, and
  // the result buffer.
  write_combined(compute_sets[0], 0, source_view, nearest_clamp);
  write_combined(compute_sets[0], 1, source_view, nearest_repeat);
  write_buffer(compute_sets[0], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
               result_buffer);
  // Set 1: the 2x2 image at binding 0, the same number as set 0's binding 0.
  write_combined(compute_sets[1], 0, small_view, linear_clamp);

  // The separate pipeline: one image, one sampler, both at binding numbers of
  // their own, and the same result buffer.
  {
    VkDescriptorImageInfo image{};
    image.imageView = source_view;
    image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = compute_sets[2];
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &image;
    update_descriptor_sets(device, 1, &write, 0, nullptr);
  }
  {
    VkDescriptorImageInfo image{};
    image.sampler = linear_clamp;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = compute_sets[2];
    write.dstBinding = 1;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    write.pImageInfo = &image;
    update_descriptor_sets(device, 1, &write, 0, nullptr);
  }
  write_buffer(compute_sets[2], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
               result_buffer);

  // The graphics sets: the same image through NEAREST and through LINEAR.
  write_combined(graphics_sets[0], 0, source_view, nearest_clamp);
  write_combined(graphics_sets[1], 0, source_view, linear_clamp);

  // ---- command buffers -----------------------------------------------------
  VkCommandPool command_pool = VK_NULL_HANDLE;
  {
    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex = 0;
    check(create_command_pool(device, &info, nullptr, &command_pool) == VK_SUCCESS,
          "vkCreateCommandPool");
  }

  auto new_command = [&]() -> VkCommandBuffer {
    VkCommandBufferAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool = command_pool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (allocate_command_buffers(device, &info, &command) != VK_SUCCESS)
      return VK_NULL_HANDLE;
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (begin_command_buffer(command, &begin) != VK_SUCCESS)
      return VK_NULL_HANDLE;
    return command;
  };

  auto submit_and_wait = [&](VkCommandBuffer command) -> bool {
    if (end_command_buffer(command) != VK_SUCCESS)
      return false;

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    if (create_fence(device, &fence_info, nullptr, &fence) != VK_SUCCESS)
      return false;

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    const bool ok =
        queue_submit(queue, 1, &submit, fence) == VK_SUCCESS &&
        wait_for_fences(device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    destroy_fence(device, fence, nullptr);
    return ok;
  };

  auto transition = [&](VkCommandBuffer command, VkImage image,
                        VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    cmd_pipeline_barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
  };

  // ---- uploading the images ------------------------------------------------
  //
  // One submission for both images: undefined -> transfer destination, copy,
  // then transfer destination -> shader read.  Kniz has no image layouts to
  // change, so the barriers are recorded and replayed as memory barriers; the
  // point of writing them is that the test then looks like an application.
  {
    VkCommandBuffer command = new_command();
    check(command != VK_NULL_HANDLE, "a command buffer for the image upload");

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {kSourceSize, kSourceSize, 1};
    transition(command, source_image, VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    cmd_copy_buffer_to_image(command, source_staging, source_image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    transition(command, source_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    region.imageExtent = {kSmallSize, kSmallSize, 1};
    transition(command, small_image, VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    cmd_copy_buffer_to_image(command, small_staging, small_image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    transition(command, small_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    check(submit_and_wait(command), "the images are uploaded");
  }

  if (failures() > 0) {
    tear_down(&runtime);
    return 1;
  }

  // Reads the result buffer back as floats.
  auto read_results = [&](double values[kComputeCount][4]) {
    void* target = nullptr;
    if (map_memory(device, result_memory, 0, VK_WHOLE_SIZE, 0, &target) !=
        VK_SUCCESS)
      return false;
    // The shader wrote floats; the comparison works in double.  Copying the
    // bytes straight into a double array would reinterpret pairs of floats as
    // one double, which is exactly the kind of mistake this test is meant to
    // catch in the driver rather than make itself.
    const float* raw = static_cast<const float*>(target);
    for (uint32_t i = 0; i < kComputeCount; ++i)
      for (int channel = 0; channel < 4; ++channel)
        values[i][channel] = raw[i * 4 + channel];
    unmap_memory(device, result_memory);
    return true;
  };

  // Runs one compute pipeline and compares every invocation against `expected`.
  auto run_compute = [&](const char* what, VkPipeline pipeline,
                         VkPipelineLayout layout, const VkDescriptorSet* sets,
                         uint32_t set_count,
                         const double expected[kComputeCount][4]) {
    VkCommandBuffer command = new_command();
    check(command != VK_NULL_HANDLE, what);
    if (command == VK_NULL_HANDLE)
      return;

    cmd_bind_pipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    cmd_bind_descriptor_sets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0,
                             set_count, sets, 0, nullptr);
    // local_size_x is 64 and there is one invocation per texel, so one group.
    cmd_dispatch(command, 1, 1, 1);
    check(submit_and_wait(command), "the dispatch completes");

    double actual[kComputeCount][4] = {};
    check(read_results(actual), "the result buffer reads back");

    int mismatches = 0;
    std::string first;
    for (uint32_t i = 0; i < kComputeCount; ++i) {
      if (close(actual[i], expected[i], 1e-3))
        continue;
      if (mismatches == 0) {
        first = "invocation " + std::to_string(i) + " read " +
                vec_text(actual[i]) + ", expected " + vec_text(expected[i]);
      }
      ++mismatches;
    }
    if (mismatches > 0)
      std::printf("     %s\n", first.c_str());
    check(mismatches == 0, "every invocation reads the texels it should");
  };

  // ---- phase 1: combined image samplers, two sets -------------------------
  //
  // One image read three ways: texelFetch through a combined sampler, a REPEAT
  // address more than a whole wrap outside the image, and a 2x2 image at the
  // same binding number in the next set.
  {
    double expected[kComputeCount][4] = {};
    double blended[4];
    bilinear(small_bytes, kSmallSize, 0.5, 0.5, blended);
    for (uint32_t i = 0; i < kComputeCount; ++i) {
      double fetched[4];
      texel(source_bytes, kSourceSize, static_cast<int>(i % kSourceSize),
            static_cast<int>(i / kSourceSize), fetched);
      for (int channel = 0; channel < 4; ++channel)
        expected[i][channel] =
            fetched[channel] + 2.0 * fetched[channel] + 4.0 * blended[channel];
    }

    const VkDescriptorSet sets[2] = {compute_sets[0], compute_sets[1]};
    run_compute("the combined image sampler dispatch", sample_pipeline,
                combined_layout, sets, 2, expected);
  }

  // ---- phase 2: a separate image and sampler ------------------------------
  //
  // The bilinear read is at a texel corner, so a sampler object that never
  // arrived would fall back to the texture's own NEAREST state and read one
  // texel instead of the average of four.
  {
    double expected[kComputeCount][4] = {};
    for (uint32_t i = 0; i < kComputeCount; ++i) {
      const int x = static_cast<int>(i % kSourceSize);
      const int y = static_cast<int>(i / kSourceSize);
      double filtered[4];
      double fetched[4];
      // The shader addresses the texel corner: u = x / 8, so the position in
      // texel space is x - 0.5, exactly between four texels.
      bilinear(source_bytes, kSourceSize, x - 0.5, y - 0.5, filtered);
      texel(source_bytes, kSourceSize, x, y, fetched);
      for (int channel = 0; channel < 4; ++channel)
        expected[i][channel] = filtered[channel] + 2.0 * fetched[channel];
    }

    const VkDescriptorSet sets[1] = {compute_sets[2]};
    run_compute("the separate image and sampler dispatch", separate_pipeline,
                separate_layout, sets, 1, expected);
  }

  // ---- phases 3 and 4: sampling in a graphics pipeline --------------------
  //
  // The fragment shader samples at gl_FragCoord.xy / 64, so pixel (c, r) lands at
  // texel-space position (c + 0.5) / 8 - 0.5.  With NEAREST that is texel
  // (c / 8, r / 8) exactly; with LINEAR it is a fixed fraction between two
  // texels, and both are computed here rather than read back.
  auto run_graphics = [&](const char* what, VkDescriptorSet set,
                          bool linear) {
    VkCommandBuffer command = new_command();
    check(command != VK_NULL_HANDLE, what);
    if (command == VK_NULL_HANDLE)
      return;

    VkRenderPassBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    begin.renderPass = render_pass;
    begin.framebuffer = framebuffer;
    begin.renderArea = {{0, 0}, {kTargetSize, kTargetSize}};
    const VkClearValue clear = clear_value();
    begin.clearValueCount = 1;
    begin.pClearValues = &clear;
    cmd_begin_render_pass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
    cmd_bind_pipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);
    cmd_bind_descriptor_sets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                             graphics_layout, 0, 1, &set, 0, nullptr);
    cmd_draw(command, 3, 1, 0, 0);
    cmd_end_render_pass(command);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {kTargetSize, kTargetSize, 1};
    cmd_copy_image_to_buffer(command, target_image,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1,
                             &region);
    check(submit_and_wait(command), "the draw and its readback complete");

    std::vector<uint8_t> actual(kTargetBytes);
    void* source = nullptr;
    if (map_memory(device, readback_memory, 0, VK_WHOLE_SIZE, 0, &source) !=
        VK_SUCCESS) {
      check(false, "the render target reads back");
      return;
    }
    std::memcpy(actual.data(), source, kTargetBytes);
    unmap_memory(device, readback_memory);

    int mismatches = 0;
    std::string first;
    for (uint32_t row = 0; row < kTargetSize; ++row) {
      for (uint32_t column = 0; column < kTargetSize; ++column) {
        // The pixel centre in texel space: u * size - 0.5.
        const double px =
            (column + 0.5) * kSourceSize / kTargetSize - 0.5;
        const double py = (row + 0.5) * kSourceSize / kTargetSize - 0.5;

        double expected[4];
        if (linear)
          bilinear(source_bytes, kSourceSize, px, py, expected);
        else
          texel(source_bytes, kSourceSize,
                static_cast<int>(std::floor(px + 0.5)),
                static_cast<int>(std::floor(py + 0.5)), expected);

        const uint8_t* pixel =
            actual.data() + (static_cast<size_t>(row) * kTargetSize + column) * 4;
        bool matches = true;
        for (int channel = 0; channel < 4; ++channel) {
          // One count of the 8-bit quantisation, and no more: every weight in
          // play is an exact binary fraction.
          if (std::fabs(pixel[channel] / 255.0 - expected[channel]) > 1.5 / 255.0)
            matches = false;
        }
        if (matches)
          continue;
        if (mismatches == 0) {
          char text[192];
          std::snprintf(text, sizeof text,
                        "pixel (%u, %u) read (%u, %u, %u, %u), expected %s",
                        column, row, pixel[0], pixel[1], pixel[2], pixel[3],
                        vec_text(expected).c_str());
          first = text;
        }
        ++mismatches;
      }
    }
    if (mismatches > 0)
      std::printf("     %s\n", first.c_str());
    check(mismatches == 0, "every pixel reads the texels it should");
  };

  run_graphics("a textured draw with a NEAREST sampler", graphics_sets[0], false);
  run_graphics("a textured draw with a LINEAR sampler", graphics_sets[1], true);

  // ---- phase 5: refusals ---------------------------------------------------
  //
  // Each is refused where the application reveals it: at sampler creation, or at
  // the descriptor write, or -- for a shader that cannot be translated -- at
  // pipeline creation.
  {
    VkSamplerCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = VK_FILTER_NEAREST;
    info.minFilter = VK_FILTER_NEAREST;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.maxLod = 0.25f;

    VkSampler rejected = VK_NULL_HANDLE;
    VkSamplerCreateInfo compare = info;
    compare.compareEnable = VK_TRUE;
    compare.compareOp = VK_COMPARE_OP_LESS;
    check(create_sampler(device, &compare, nullptr, &rejected) ==
              VK_ERROR_FEATURE_NOT_PRESENT,
          "a comparison sampler is refused");

    VkSamplerCreateInfo unnormalized = info;
    unnormalized.unnormalizedCoordinates = VK_TRUE;
    check(create_sampler(device, &unnormalized, nullptr, &rejected) ==
              VK_ERROR_FEATURE_NOT_PRESENT,
          "unnormalized coordinates are refused");

    VkSamplerCreateInfo anisotropic = info;
    anisotropic.anisotropyEnable = VK_TRUE;
    anisotropic.maxAnisotropy = 4.0f;
    check(create_sampler(device, &anisotropic, nullptr, &rejected) ==
              VK_ERROR_FEATURE_NOT_PRESENT,
          "anisotropic filtering is refused (the feature is not reported)");

    VkSamplerCreateInfo mirrored = info;
    mirrored.addressModeU = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    check(create_sampler(device, &mirrored, nullptr, &rejected) ==
              VK_ERROR_FEATURE_NOT_PRESENT,
          "mirror clamp to edge is refused (a Vulkan extension Kniz does not "
          "report)");

    VkSamplerCreateInfo lod = info;
    lod.mipLodBias = 0.5f;
    check(create_sampler(device, &lod, nullptr, &rejected) == VK_SUCCESS,
          "a sampler with a mip LOD bias is accepted");
  }

  // A sampled image needs VK_IMAGE_USAGE_SAMPLED_BIT, and an image with more than
  // one mip level cannot be sampled because only level 0 is ever written.  Both
  // are refused at the descriptor write, and the proof is that the draw which
  // would have used the binding is then refused too, leaving the clear colour in
  // the target rather than whatever the driver had bound.
  {
    VkImage unmarked_image = VK_NULL_HANDLE;
    VkDeviceMemory unmarked_memory = VK_NULL_HANDLE;
    VkImageView unmarked_view = VK_NULL_HANDLE;
    check(make_image(kSourceSize, VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                     &unmarked_image, &unmarked_memory, &unmarked_view),
          "an image created without VK_IMAGE_USAGE_SAMPLED_BIT");

    VkImage mipmapped_image = VK_NULL_HANDLE;
    VkDeviceMemory mipmapped_memory = VK_NULL_HANDLE;
    VkImageView mipmapped_view = VK_NULL_HANDLE;
    {
      VkImageCreateInfo info{};
      info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      info.imageType = VK_IMAGE_TYPE_2D;
      info.format = VK_FORMAT_R8G8B8A8_UNORM;
      info.extent = {kSourceSize, kSourceSize, 1};
      info.mipLevels = 2;
      info.arrayLayers = 1;
      info.samples = VK_SAMPLE_COUNT_1_BIT;
      info.tiling = VK_IMAGE_TILING_OPTIMAL;
      info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      check(create_image(device, &info, nullptr, &mipmapped_image) == VK_SUCCESS,
            "an image with two mip levels");

      VkMemoryRequirements requirements{};
      get_image_requirements(device, mipmapped_image, &requirements);
      VkMemoryAllocateInfo allocation{};
      allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocation.allocationSize = requirements.size;
      allocation.memoryTypeIndex = runtime.host_memory_type;
      check(allocate_memory(device, &allocation, nullptr, &mipmapped_memory) ==
                    VK_SUCCESS &&
                bind_image_memory(device, mipmapped_image, mipmapped_memory, 0) ==
                    VK_SUCCESS,
            "the mipmapped image has texture storage");

      VkImageViewCreateInfo view{};
      view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      view.image = mipmapped_image;
      view.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view.format = VK_FORMAT_R8G8B8A8_UNORM;
      view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      check(create_image_view(device, &view, nullptr, &mipmapped_view) ==
                VK_SUCCESS,
            "a view of its level 0");
    }

    struct Case {
      const char* what;
      VkImageView view;
    };
    const Case cases[2] = {
        {"an image without VK_IMAGE_USAGE_SAMPLED_BIT is refused when it is "
         "sampled",
         unmarked_view},
        {"an image with more than one mip level is refused when it is sampled",
         mipmapped_view},
    };

    for (const Case& refused : cases) {
      // A set of its own, so the refused write cannot leave a usable binding
      // behind in one of the sets the phases above used.
      VkDescriptorSet set = VK_NULL_HANDLE;
      VkDescriptorSetAllocateInfo allocate{};
      allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      allocate.descriptorPool = pool;
      allocate.descriptorSetCount = 1;
      allocate.pSetLayouts = &graphics_set_layout;
      check(allocate_descriptor_sets(device, &allocate, &set) == VK_SUCCESS,
            "a descriptor set for the refused write");
      write_combined(set, 0, refused.view, nearest_clamp);

      VkCommandBuffer command = new_command();
      check(command != VK_NULL_HANDLE, refused.what);
      if (command == VK_NULL_HANDLE)
        continue;

      VkRenderPassBeginInfo begin{};
      begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      begin.renderPass = render_pass;
      begin.framebuffer = framebuffer;
      begin.renderArea = {{0, 0}, {kTargetSize, kTargetSize}};
      const VkClearValue clear = clear_value();
      begin.clearValueCount = 1;
      begin.pClearValues = &clear;
      cmd_begin_render_pass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
      cmd_bind_pipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        graphics_pipeline);
      cmd_bind_descriptor_sets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                               graphics_layout, 0, 1, &set, 0, nullptr);
      cmd_draw(command, 3, 1, 0, 0);
      cmd_end_render_pass(command);

      VkBufferImageCopy region{};
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.imageExtent = {kTargetSize, kTargetSize, 1};
      cmd_copy_image_to_buffer(command, target_image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1,
                               &region);
      if (!submit_and_wait(command)) {
        check(false, "the refused draw still completes its submission");
        continue;
      }

      std::vector<uint8_t> actual(kTargetBytes);
      void* source = nullptr;
      if (map_memory(device, readback_memory, 0, VK_WHOLE_SIZE, 0, &source) !=
          VK_SUCCESS) {
        check(false, "the target reads back after the refused draw");
        continue;
      }
      std::memcpy(actual.data(), source, kTargetBytes);
      unmap_memory(device, readback_memory);

      bool untouched = true;
      for (size_t i = 0; i < actual.size(); i += 4)
        for (int channel = 0; channel < 4; ++channel)
          if (actual[i + channel] != kClear[channel])
            untouched = false;
      check(untouched,
            "the refused draw leaves the target at its clear colour");
    }
  }

  // A sampler descriptor that is declared and never written is refused at the
  // draw, which is the same guarantee the buffer tests make: a missing binding is
  // not read as whatever the last command left there.
  {
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorPool = pool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &graphics_set_layout;
    check(allocate_descriptor_sets(device, &allocate, &set) == VK_SUCCESS,
          "a descriptor set that is never written");

    VkCommandBuffer command = new_command();
    check(command != VK_NULL_HANDLE,
          "a draw whose sampler binding was never written is refused");
    if (command != VK_NULL_HANDLE) {
      VkRenderPassBeginInfo begin{};
      begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      begin.renderPass = render_pass;
      begin.framebuffer = framebuffer;
      begin.renderArea = {{0, 0}, {kTargetSize, kTargetSize}};
      const VkClearValue clear = clear_value();
      begin.clearValueCount = 1;
      begin.pClearValues = &clear;
      cmd_begin_render_pass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
      cmd_bind_pipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        graphics_pipeline);
      cmd_bind_descriptor_sets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                               graphics_layout, 0, 1, &set, 0, nullptr);
      cmd_draw(command, 3, 1, 0, 0);
      cmd_end_render_pass(command);

      VkBufferImageCopy region{};
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.imageExtent = {kTargetSize, kTargetSize, 1};
      cmd_copy_image_to_buffer(command, target_image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1,
                               &region);
      if (!submit_and_wait(command)) {
        check(false, "the refused draw still completes its submission");
      } else {
        std::vector<uint8_t> actual(kTargetBytes);
        void* source = nullptr;
        if (map_memory(device, readback_memory, 0, VK_WHOLE_SIZE, 0, &source) ==
            VK_SUCCESS) {
          std::memcpy(actual.data(), source, kTargetBytes);
          unmap_memory(device, readback_memory);
          bool untouched = true;
          for (size_t i = 0; i < actual.size(); i += 4)
            for (int channel = 0; channel < 4; ++channel)
              if (actual[i + channel] != kClear[channel])
                untouched = false;
          check(untouched, "the target keeps its clear colour");
        }
      }
    }
  }

  tear_down(&runtime);

  std::printf("\n%s\n", failures() == 0 ? "all checks passed" : "FAILURES");
  return failures() == 0 ? 0 : 1;
}
