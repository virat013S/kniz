// M2c test: uniform buffers and push constants, in both a compute and a
// graphics pipeline.
//
// A uniform buffer is the first descriptor Kniz supports that is *read* rather
// than written, and that changes the shape of what can go wrong.  A wrong
// storage buffer binding usually shows up as a dispatch that writes somewhere
// implausible; a wrong uniform buffer binding shows up as geometry in the wrong
// place, or colours that are subtly off, which is much easier to explain away.
//
// The two things this test is built around:
//
//   * Two uniform blocks in *different descriptor sets* with the same binding
//     number, which Vulkan allows and GLSL cannot express at all.  GL numbers
//     uniform blocks in one flat namespace, so they have to end up at different
//     binding points; a translation that used the binding number directly would
//     put both at 0, and one buffer would then answer for the other.  In the
//     graphics pipeline that means the vertex shader reads the fragment shader's
//     data, so the geometry moves to wherever the colour says it should be.
//
//   * Push constants, which OpenGL does not have in any form.  Kniz lowers them
//     onto a uniform buffer it owns and uploads to as the recorded commands
//     replay, so their correctness depends on the ordering of the recording, on
//     the reservation of a binding point that no descriptor uses, and on the
//     block layout Vulkan chose matching the block layout GLSL can express.
//
// Everything below is checked numerically against a computation done here, in
// the same spirit as the compute and draw tests: what the driver produced is
// never the reference for what it should have produced.

#include "harness.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using namespace kniz_test;

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 64;
constexpr uint32_t kBytesPerPixel = 4;
constexpr size_t kImageBytes = static_cast<size_t>(kWidth) * kHeight * kBytesPerPixel;

// The compute dispatch writes one float per invocation, 64 invocations.
constexpr uint32_t kComputeCount = 64;
constexpr size_t kComputeBytes = kComputeCount * sizeof(float);

// All the buffers are host visible and give away nothing by being larger than
// they need to be; 256 bytes is one alignment unit on every driver Kniz has met.
constexpr VkDeviceSize kBufferSize = 256;

// ---------------------------------------------------------------- the values

// The vertex shader's block: a scale and an offset applied to a triangle built
// from gl_VertexIndex.  std140 puts the second vec4 at offset 16.
struct Transform {
  float offset[4];
  float scale[4];
};
static_assert(sizeof(Transform) == 32, "std140 vec4 pair is 32 bytes");

// The fragment shader's block, at the same binding number in the *next* set.
struct Tint {
  float base[4];
};
static_assert(sizeof(Tint) == 16, "");

// The fragment stage's only push constant.
struct Weight {
  float weight[4];
};
static_assert(sizeof(Weight) == 16, "");

// The compute shader's blocks.  FirstBlock is {vec4 offset; float scale;} in
// set 0 and SecondBlock is {vec4 tint;} in set 1 -- again the same binding
// number in two sets.
struct FirstBlock {
  float offset[4];
  float scale;
  float padding[3];
};
static_assert(sizeof(FirstBlock) == 32, "");

struct SecondBlock {
  float tint[4];
};
static_assert(sizeof(SecondBlock) == 16, "");

// The compute stage's push constant: a scalar followed by a vector.  Whether the
// vector lands at offset 4 or 16 is exactly the question that decides whether a
// Vulkan push constant block can be expressed as a GLSL uniform block, so the
// test exercises the answer rather than avoiding it.
struct ComputePush {
  float bias;
  float padding[3];
  float multiplier[4];
};
static_assert(sizeof(ComputePush) == 32, "");

struct ComputeInputs {
  FirstBlock first;
  SecondBlock second;
  ComputePush push;
};

// What the compute shader is asked to produce, worked out here from the values
// above rather than taken from what the driver wrote.  The arithmetic mirrors
// the shader line for line, and is deliberately built so that every input moves
// the answer: a block bound in the wrong place, a member read at the wrong
// offset, or a push constant that never arrived all produce a different number.
float expected_compute_value(const ComputeInputs& in) {
  const float v = (in.first.offset[0] + in.first.offset[1] +
                   in.first.offset[2] + in.first.offset[3]) * in.first.scale;
  const float w = in.second.tint[0] + 2.0f * in.second.tint[1] +
                  3.0f * in.second.tint[2] + 4.0f * in.second.tint[3];
  return (v + w + in.push.bias) * in.push.multiplier[0] + in.push.multiplier[1];
}

// The graphics colour: the fragment shader's uniform block multiplied by the
// push constant, quantised to the unorm8 attachment the way the hardware will.
void expected_graphics_color(const Tint& tint, const Weight& weight,
                             uint8_t out[4]) {
  for (int channel = 0; channel < 4; ++channel) {
    const double value =
        static_cast<double>(tint.base[channel]) * weight.weight[channel];
    out[channel] = static_cast<uint8_t>(std::lround(value * 255.0));
  }
}

// -------------------------------------------------------------- the geometry

// The vertex shader's input, in NDC: a triangle that covers the viewport and
// more, exactly as the draw test's gl_FragCoord phase uses.
const float kInputTriangle[3][2] = {{-1.0f, -1.0f}, {3.0f, -1.0f}, {-1.0f, 3.0f}};

// The same triangle after the transform block has been applied, in framebuffer
// coordinates counted downwards from the top-left.
void transformed_vertex(const Transform& transform, int index, double* x, double* y) {
  const double ndc_x = static_cast<double>(kInputTriangle[index][0]) *
                           transform.scale[0] +
                       transform.offset[0];
  const double ndc_y = static_cast<double>(kInputTriangle[index][1]) *
                           transform.scale[1] +
                       transform.offset[1];
  *x = (ndc_x + 1.0) * 0.5 * kWidth;
  *y = (ndc_y + 1.0) * 0.5 * kHeight;
}

// Barycentric weights of a point in framebuffer coordinates.  Whether the point
// is covered is decided here, from the transformed vertex positions, so the
// coverage the test demands is a computation and not a copy of the driver's.
bool covered(const Transform& transform, double px, double py) {
  double x[3] = {}, y[3] = {};
  for (int i = 0; i < 3; ++i)
    transformed_vertex(transform, i, &x[i], &y[i]);

  const double denominator =
      (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2]);
  const double w0 =
      ((y[1] - y[2]) * (px - x[2]) + (x[2] - x[1]) * (py - y[2])) / denominator;
  const double w1 =
      ((y[2] - y[0]) * (px - x[2]) + (x[0] - x[2]) * (py - y[2])) / denominator;
  const double w2 = 1.0 - w0 - w1;

  // The vertices are chosen so that no texel centre lands exactly on an edge, so
  // this slack never decides anything; it is here so that a future change of
  // transform produces a failure rather than a coin flip.
  constexpr double kSlack = 1e-9;
  return w0 >= -kSlack && w1 >= -kSlack && w2 >= -kSlack;
}

const uint8_t* texel(const std::vector<uint8_t>& bytes, uint32_t x, uint32_t y) {
  return bytes.data() + (static_cast<size_t>(y) * kWidth + x) * kBytesPerPixel;
}

std::string texel_text(const uint8_t* value) {
  char text[64];
  std::snprintf(text, sizeof text, "(%u, %u, %u, %u)", value[0], value[1],
                value[2], value[3]);
  return text;
}

VkClearValue clear_value(const uint8_t bytes[4]) {
  VkClearValue value{};
  for (int i = 0; i < 4; ++i)
    value.color.float32[i] = bytes[i] / 255.0f;
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
  std::printf("kniz uniform test (M2c)\n");

  const char* shader_dir = argc > 1 ? argv[1] : ".";

  if (!icd_selected())
    return 2;

  void* loader = open_loader();
  if (loader == nullptr)
    return 2;
  if (!resolve_get_instance_proc_addr(loader))
    return 2;

  Runtime runtime;
  if (!bring_up(&runtime, "kniz-uniform"))
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

  auto cmd_begin_render_pass =
      device_proc<PFN_vkCmdBeginRenderPass>(device, "vkCmdBeginRenderPass");
  auto cmd_end_render_pass =
      device_proc<PFN_vkCmdEndRenderPass>(device, "vkCmdEndRenderPass");
  auto cmd_bind_pipeline =
      device_proc<PFN_vkCmdBindPipeline>(device, "vkCmdBindPipeline");
  auto cmd_bind_descriptor_sets =
      device_proc<PFN_vkCmdBindDescriptorSets>(device, "vkCmdBindDescriptorSets");
  auto cmd_push_constants =
      device_proc<PFN_vkCmdPushConstants>(device, "vkCmdPushConstants");
  auto cmd_draw = device_proc<PFN_vkCmdDraw>(device, "vkCmdDraw");
  auto cmd_dispatch = device_proc<PFN_vkCmdDispatch>(device, "vkCmdDispatch");
  auto cmd_copy_image_to_buffer =
      device_proc<PFN_vkCmdCopyImageToBuffer>(device, "vkCmdCopyImageToBuffer");

  auto queue_submit = device_proc<PFN_vkQueueSubmit>(device, "vkQueueSubmit");
  auto create_fence = device_proc<PFN_vkCreateFence>(device, "vkCreateFence");
  auto reset_fences = device_proc<PFN_vkResetFences>(device, "vkResetFences");
  auto wait_for_fences = device_proc<PFN_vkWaitForFences>(device, "vkWaitForFences");

  require("vkCreateDescriptorSetLayout",
          reinterpret_cast<PFN_vkVoidFunction>(create_set_layout));
  require("vkCreatePipelineLayout",
          reinterpret_cast<PFN_vkVoidFunction>(create_pipeline_layout));
  require("vkCmdPushConstants", reinterpret_cast<PFN_vkVoidFunction>(cmd_push_constants));
  require("vkCmdDispatch", reinterpret_cast<PFN_vkVoidFunction>(cmd_dispatch));
  require("vkCreateGraphicsPipelines",
          reinterpret_cast<PFN_vkVoidFunction>(create_graphics_pipelines));
  require("vkCreateComputePipelines",
          reinterpret_cast<PFN_vkVoidFunction>(create_compute_pipelines));
  if (failures() > 0) {
    tear_down(&runtime);
    return 1;
  }

  // ---- buffers -------------------------------------------------------------
  //
  // One host visible block per buffer, always mapped while it is being written
  // and unmapped afterwards, which is the same pattern the compute test uses:
  // Kniz keeps the host block as the buffer's contents and moves them into GL at
  // submission.
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

  VkBuffer transform_buffer = VK_NULL_HANDLE;
  VkDeviceMemory transform_memory = VK_NULL_HANDLE;
  VkBuffer tint_buffer = VK_NULL_HANDLE;
  VkDeviceMemory tint_memory = VK_NULL_HANDLE;
  VkBuffer first_buffer = VK_NULL_HANDLE;
  VkDeviceMemory first_memory = VK_NULL_HANDLE;
  VkBuffer second_buffer = VK_NULL_HANDLE;
  VkDeviceMemory second_memory = VK_NULL_HANDLE;
  VkBuffer result_buffer = VK_NULL_HANDLE;
  VkDeviceMemory result_memory = VK_NULL_HANDLE;
  VkBuffer readback = VK_NULL_HANDLE;
  VkDeviceMemory readback_memory = VK_NULL_HANDLE;

  void* mapped = nullptr;
  {
    check(make_buffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, kBufferSize,
                      &transform_buffer, &transform_memory, &mapped),
          "the vertex stage's uniform buffer");
    std::memset(mapped, 0, kBufferSize);
    unmap_memory(device, transform_memory);
  }
  {
    check(make_buffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, kBufferSize,
                      &tint_buffer, &tint_memory, &mapped),
          "the fragment stage's uniform buffer");
    std::memset(mapped, 0, kBufferSize);
    unmap_memory(device, tint_memory);
  }
  {
    check(make_buffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, kBufferSize,
                      &first_buffer, &first_memory, &mapped),
          "the compute shader's first uniform buffer");
    std::memset(mapped, 0, kBufferSize);
    unmap_memory(device, first_memory);
  }
  {
    check(make_buffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, kBufferSize,
                      &second_buffer, &second_memory, &mapped),
          "the compute shader's second uniform buffer (a different set, the same "
          "binding number)");
    std::memset(mapped, 0, kBufferSize);
    unmap_memory(device, second_memory);
  }
  {
    check(make_buffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kBufferSize,
                      &result_buffer, &result_memory, &mapped),
          "the compute shader's result buffer");
    std::memset(mapped, 0, kBufferSize);
    unmap_memory(device, result_memory);
  }
  {
    // The one buffer here that is as large as the attachment: a copy of the
    // whole image has to land somewhere.
    check(make_buffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT, kImageBytes, &readback,
                      &readback_memory, &mapped),
          "the readback buffer for the render target");
    std::memset(mapped, 0, kImageBytes);
    unmap_memory(device, readback_memory);
  }

  // Writes a struct into a buffer by mapping it, so the test does not depend on
  // how Kniz keeps its host copy.
  auto write_buffer = [&](VkDeviceMemory memory, const void* data, size_t bytes) {
    void* target = nullptr;
    if (map_memory(device, memory, 0, VK_WHOLE_SIZE, 0, &target) != VK_SUCCESS)
      return;
    std::memcpy(target, data, bytes);
    unmap_memory(device, memory);
  };

  // ---- the render target ---------------------------------------------------
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory image_memory = VK_NULL_HANDLE;
  VkImageView image_view = VK_NULL_HANDLE;
  VkRenderPass render_pass = VK_NULL_HANDLE;
  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  {
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_R8G8B8A8_UNORM;
    info.extent = {kWidth, kHeight, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    check(create_image(device, &info, nullptr, &image) == VK_SUCCESS,
          "vkCreateImage (the render target)");

    VkMemoryRequirements requirements{};
    get_image_requirements(device, image, &requirements);
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = runtime.host_memory_type;
    check(allocate_memory(device, &allocation, nullptr, &image_memory) == VK_SUCCESS &&
              bind_image_memory(device, image, image_memory, 0) == VK_SUCCESS,
          "the render target has texture storage");

    VkImageViewCreateInfo view{};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = VK_FORMAT_R8G8B8A8_UNORM;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    check(create_image_view(device, &view, nullptr, &image_view) == VK_SUCCESS,
          "vkCreateImageView (the render target)");

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
    framebuffer_info.pAttachments = &image_view;
    framebuffer_info.width = kWidth;
    framebuffer_info.height = kHeight;
    framebuffer_info.layers = 1;
    check(create_framebuffer(device, &framebuffer_info, nullptr, &framebuffer) ==
              VK_SUCCESS,
          "vkCreateFramebuffer");
  }

  // ---- descriptor set layouts ---------------------------------------------
  //
  // Four set layouts, each with exactly one binding.  The two that the graphics
  // pipeline uses are separate layouts because they belong to separate sets, and
  // the same is true of the compute pipeline's.
  auto make_set_layout = [&](VkDescriptorType type, VkShaderStageFlags stages,
                             VkDescriptorSetLayout* out) -> VkResult {
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = type;
    binding.descriptorCount = 1;
    binding.stageFlags = stages;

    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = 1;
    info.pBindings = &binding;
    return create_set_layout(device, &info, nullptr, out);
  };

  VkDescriptorSetLayout vertex_set_layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout fragment_set_layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout compute_uniform_layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout compute_storage_layout = VK_NULL_HANDLE;
  check(make_set_layout(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        VK_SHADER_STAGE_VERTEX_BIT, &vertex_set_layout) == VK_SUCCESS,
        "a set layout with a uniform buffer for the vertex stage");
  check(make_set_layout(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        VK_SHADER_STAGE_FRAGMENT_BIT,
                        &fragment_set_layout) == VK_SUCCESS,
        "a second one for the fragment stage");
  check(make_set_layout(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        VK_SHADER_STAGE_COMPUTE_BIT,
                        &compute_uniform_layout) == VK_SUCCESS,
        "a set layout with a uniform buffer for compute");

  // The compute shader's set 0 also holds its storage buffer, at binding 1.
  {
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = 2;
    info.pBindings = bindings;
    check(create_set_layout(device, &info, nullptr, &compute_storage_layout) ==
              VK_SUCCESS,
          "a set layout with a uniform buffer and a storage buffer");
  }

  // ---- pipeline layouts ----------------------------------------------------
  //
  // This is where the flat GL binding points are decided, and the test is built
  // so that a wrong decision is visible: the push constant block takes uniform
  // binding point 0 and the two uniform blocks take the points after it, so the
  // vertex stage's block and the fragment stage's block end up at different
  // points even though both are binding 0 in their own sets.  Nothing here can
  // see those numbers directly; what the phases below see is whether each shader
  // reads the buffer the application bound.
  auto make_pipeline_layout = [&](const VkDescriptorSetLayout* sets, uint32_t set_count,
                                  const VkPushConstantRange* range,
                                  VkPipelineLayout* out) -> VkResult {
    VkPipelineLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info.setLayoutCount = set_count;
    info.pSetLayouts = sets;
    if (range != nullptr) {
      info.pushConstantRangeCount = 1;
      info.pPushConstantRanges = range;
    }
    return create_pipeline_layout(device, &info, nullptr, out);
  };

  VkPipelineLayout graphics_layout = VK_NULL_HANDLE;
  VkPipelineLayout compute_layout = VK_NULL_HANDLE;
  {
    const VkDescriptorSetLayout sets[2] = {vertex_set_layout, fragment_set_layout};
    // The push constant is written by the fragment stage only, so the range names
    // only that stage -- and the vertex stage's block and the push constant block
    // therefore live in the same GL uniform namespace at different points.
    const VkPushConstantRange range{VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                    static_cast<uint32_t>(sizeof(Weight))};
    check(make_pipeline_layout(sets, 2, &range, &graphics_layout) == VK_SUCCESS,
          "vkCreatePipelineLayout (two sets, one push constant range)");
  }
  {
    const VkDescriptorSetLayout sets[2] = {compute_storage_layout,
                                           compute_uniform_layout};
    const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                    static_cast<uint32_t>(sizeof(ComputePush))};
    check(make_pipeline_layout(sets, 2, &range, &compute_layout) == VK_SUCCESS,
          "vkCreatePipelineLayout (compute)");
  }

  if (failures() > 0) {
    tear_down(&runtime);
    return 1;
  }

  // ---- refusals that are properties of a pipeline layout -------------------
  //
  // Each of these is refused where the application reveals it, which for every
  // one of them is layout creation rather than the draw or dispatch that would
  // have been wrong.
  VkDescriptorSetLayout rejected_layout = VK_NULL_HANDLE;
  check(make_set_layout(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                        VK_SHADER_STAGE_VERTEX_BIT,
                        &rejected_layout) == VK_ERROR_FEATURE_NOT_PRESENT,
        "a dynamic uniform buffer is refused (vkCmdBindDescriptorSets offsets "
        "are not implemented)");
  check(make_set_layout(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        VK_SHADER_STAGE_FRAGMENT_BIT,
                        &rejected_layout) == VK_ERROR_FEATURE_NOT_PRESENT,
        "a combined image sampler is refused (that is M2d)");
  check(make_set_layout(VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
                        VK_SHADER_STAGE_COMPUTE_BIT,
                        &rejected_layout) == VK_ERROR_FEATURE_NOT_PRESENT,
        "a storage texel buffer is refused");
  {
    // The layout the refusals are aimed at must not be the one the test then
    // uses, so a throwaway handle is passed instead: a layout that is refused
    // leaves it untouched, and a layout that is mistakenly accepted leaves it
    // holding something this test never draws with.
    VkPipelineLayout rejected = VK_NULL_HANDLE;
    const VkDescriptorSetLayout sets[1] = {vertex_set_layout};
    VkPushConstantRange too_large{VK_SHADER_STAGE_VERTEX_BIT, 0, 132};
    check(make_pipeline_layout(sets, 1, &too_large, &rejected) ==
              VK_ERROR_INITIALIZATION_FAILED,
          "a push constant range past maxPushConstantsSize is refused");

    VkPushConstantRange misaligned{VK_SHADER_STAGE_VERTEX_BIT, 2, 4};
    check(make_pipeline_layout(sets, 1, &misaligned, &rejected) ==
              VK_ERROR_INITIALIZATION_FAILED,
          "a push constant range at an unaligned offset is refused");
  }

  // ---- descriptor pool and sets -------------------------------------------
  VkDescriptorPool pool = VK_NULL_HANDLE;
  {
    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[0].descriptorCount = 16;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[1].descriptorCount = 8;

    VkDescriptorPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.maxSets = 8;
    info.poolSizeCount = 2;
    info.pPoolSizes = sizes;
    check(create_descriptor_pool(device, &info, nullptr, &pool) == VK_SUCCESS,
          "vkCreateDescriptorPool");
  }

  auto allocate_set = [&](VkDescriptorSetLayout set_layout) -> VkDescriptorSet {
    VkDescriptorSetAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    info.descriptorPool = pool;
    info.descriptorSetCount = 1;
    info.pSetLayouts = &set_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (allocate_descriptor_sets(device, &info, &set) != VK_SUCCESS)
      return VK_NULL_HANDLE;
    return set;
  };

  auto write_uniform = [&](VkDescriptorSet set, VkBuffer buffer) {
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = buffer;
    buffer_info.offset = 0;
    buffer_info.range = kBufferSize;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = &buffer_info;
    update_descriptor_sets(device, 1, &write, 0, nullptr);
  };

  const VkDescriptorSet vertex_set = allocate_set(vertex_set_layout);
  const VkDescriptorSet fragment_set = allocate_set(fragment_set_layout);
  // A second vertex set, deliberately never written, for the refusal below.
  const VkDescriptorSet unwritten_vertex_set = allocate_set(vertex_set_layout);
  const VkDescriptorSet compute_set0 = allocate_set(compute_storage_layout);
  const VkDescriptorSet compute_set1 = allocate_set(compute_uniform_layout);
  check(vertex_set != VK_NULL_HANDLE && fragment_set != VK_NULL_HANDLE &&
            unwritten_vertex_set != VK_NULL_HANDLE &&
            compute_set0 != VK_NULL_HANDLE && compute_set1 != VK_NULL_HANDLE,
        "five descriptor sets are allocated");

  write_uniform(vertex_set, transform_buffer);
  write_uniform(fragment_set, tint_buffer);
  write_uniform(compute_set0, first_buffer);
  write_uniform(compute_set1, second_buffer);
  {
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = result_buffer;
    buffer_info.offset = 0;
    buffer_info.range = kBufferSize;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = compute_set0;
    write.dstBinding = 1;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buffer_info;
    update_descriptor_sets(device, 1, &write, 0, nullptr);
  }

  // ---- shader modules and pipelines ---------------------------------------
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

  VkShaderModule vertex_module = VK_NULL_HANDLE;
  VkShaderModule fragment_module = VK_NULL_HANDLE;
  VkShaderModule compute_module = VK_NULL_HANDLE;
  check(make_module("uniform.vert.spv", &vertex_module) &&
            make_module("uniform.frag.spv", &fragment_module) &&
            make_module("uniform.comp.spv", &compute_module),
        "the shader modules");

  auto build_graphics_pipeline = [&](VkPipelineLayout layout, VkPipeline* out) {
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertex_module;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragment_module;
    stages[1].pName = "main";

    // No vertex buffers at all: the shader builds its triangle out of
    // gl_VertexIndex, so nothing here can move the geometry except the uniform
    // block.
    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

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
  check(build_graphics_pipeline(graphics_layout, &graphics_pipeline) == VK_SUCCESS,
        "vkCreateGraphicsPipelines (a uniform block in two sets and a push "
        "constant)");

  VkPipeline compute_pipeline = VK_NULL_HANDLE;
  {
    VkComputePipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = compute_module;
    info.stage.pName = "main";
    info.layout = compute_layout;
    check(create_compute_pipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                                   &compute_pipeline) == VK_SUCCESS,
          "vkCreateComputePipelines (two uniform blocks in two sets, a storage "
          "buffer and push constants)");
  }

  // ---- a push constant block std140 cannot express -------------------------
  //
  // Vulkan packs an array of scalars in a push constant block 4 bytes apart;
  // a GLSL uniform block cannot be packed that tightly, since std140 puts those
  // elements 16 bytes apart and the scalar layout that would match is an
  // extension.  Lowering this block would leave the shader reading its weights
  // from the wrong offsets and producing a plausible wrong answer, so the
  // pipeline is refused at creation instead.  This is a real Vulkan construct --
  // it is what an application writes when it wants to push four weights -- so it
  // is a limitation of the emulation and not a malformed shader.
  {
    VkShaderModule packed_module = VK_NULL_HANDLE;
    check(make_module("uniform_packed.comp.spv", &packed_module),
          "the shader module holding a push constant block std140 cannot express");

    VkComputePipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = packed_module;
    info.stage.pName = "main";
    info.layout = compute_layout;
    VkPipeline packed_pipeline = VK_NULL_HANDLE;
    check(create_compute_pipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                                   &packed_pipeline) ==
              VK_ERROR_INITIALIZATION_FAILED,
          "a push constant block that needs tighter packing than std140 is "
          "refused rather than emitted with its members at the wrong offsets");
  }

  // ---- a shader that reads push constants with no ranges declared ----------
  //
  // Vulkan calls that invalid usage, and the useful failure is at pipeline
  // creation: Kniz lowers the push constant block onto the reserved uniform
  // binding point, and a layout with no ranges has a descriptor there instead,
  // so the shader would quietly read a uniform buffer's first bytes as its
  // constants.  The layout here declares both descriptor sets and no ranges, so
  // the only thing that can be refused is the push constant block.
  {
    const VkDescriptorSetLayout sets[2] = {vertex_set_layout, fragment_set_layout};
    VkPipelineLayout no_push = VK_NULL_HANDLE;
    check(make_pipeline_layout(sets, 2, nullptr, &no_push) == VK_SUCCESS,
          "a pipeline layout with both sets and no push constant ranges");

    VkPipeline refused = VK_NULL_HANDLE;
    check(build_graphics_pipeline(no_push, &refused) ==
              VK_ERROR_INITIALIZATION_FAILED,
          "a shader that reads push constants is refused when the layout "
          "declares no ranges for them");
  }

  if (failures() > 0) {
    tear_down(&runtime);
    return 1;
  }

  // ---- command buffer and fence -------------------------------------------
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkCommandBuffer commands = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  {
    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // The command buffer is re-recorded for every phase below.
    info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(create_command_pool(device, &info, nullptr, &command_pool) == VK_SUCCESS,
          "vkCreateCommandPool");

    VkCommandBufferAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocation.commandPool = command_pool;
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

  // ---- the compute phases --------------------------------------------------
  const ComputeInputs first_inputs = {
      {{1.5f, 2.0f, -0.5f, 3.0f}, 2.0f, {0.0f, 0.0f, 0.0f}},
      {{1.0f, 2.0f, 3.0f, 4.0f}},
      {0.5f, {0.0f, 0.0f, 0.0f}, {3.0f, -1.0f, 0.0f, 0.0f}},
  };
  // The second dispatch changes every one of the three inputs, so it is evidence
  // about the uniform blocks being re-read as well as about the push constant
  // being per-dispatch state rather than something set once.
  const ComputeInputs second_inputs = {
      {{2.0f, -1.0f, 0.5f, 1.5f}, 0.5f, {0.0f, 0.0f, 0.0f}},
      {{1.0f, 2.0f, 3.0f, 4.0f}},
      {-1.5f, {0.0f, 0.0f, 0.0f}, {2.0f, 0.25f, 0.0f, 0.0f}},
  };

  write_buffer(first_memory, &first_inputs.first, sizeof(FirstBlock));
  write_buffer(second_memory, &first_inputs.second, sizeof(SecondBlock));

  // A dispatch with everything bound, run twice with different push constants.
  // The two runs differ only in the push constant, so the difference between
  // their results is evidence that the push constant reached the shader and that
  // it is command buffer state rather than something set once per pipeline.
  auto dispatch_and_read = [&](const ComputePush& push,
                               std::vector<float>* out) -> bool {
    out->assign(kComputeCount, 0.0f);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (begin_command_buffer(commands, &begin) != VK_SUCCESS)
      return false;

    cmd_bind_pipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);
    const VkDescriptorSet sets[2] = {compute_set0, compute_set1};
    cmd_bind_descriptor_sets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, compute_layout,
                             0, 2, sets, 0, nullptr);
    cmd_push_constants(commands, compute_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       static_cast<uint32_t>(sizeof(ComputePush)), &push);
    cmd_dispatch(commands, 1, 1, 1);

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

    void* target = nullptr;
    if (map_memory(device, result_memory, 0, VK_WHOLE_SIZE, 0, &target) != VK_SUCCESS)
      return false;
    std::memcpy(out->data(), target, kComputeBytes);
    unmap_memory(device, result_memory);
    return true;
  };

  // The value the shader should produce, computed here from the same inputs.
  const float expected_first = expected_compute_value(first_inputs);
  const float expected_second = expected_compute_value(second_inputs);

  // The two failures this test exists to catch both produce a specific wrong
  // number, worked out here so a failure can be told apart from a rounding
  // difference:
  //
  //   * If the two uniform blocks had been given the same GL binding point, the
  //     first block's members would have been read out of the second block's
  //     buffer.  Its first vec4 is {1, 2, 3, 4} and the four bytes where `scale`
  //     would be are zero, so v would be 0 and the answer would be
  //     (0 + 30 + 0.5) * 3 - 1 = 90.5.
  //   * If the push constant had never arrived, both its members would read as
  //     zero and the answer would be (12 + 30 + 0) * 0 + 0 = 0.
  //
  // Neither is anywhere near 126.5, which is what the first dispatch asks for.
  const float collision_value = 90.5f;
  const float zero_push_value = 0.0f;

  {
    std::vector<float> values;
    check(dispatch_and_read(first_inputs.push, &values),
          "the compute dispatch with uniform buffers and push constants runs");

    int mismatches = 0;
    for (uint32_t i = 0; i < kComputeCount; ++i) {
      if (std::fabs(values[i] - expected_first) >
          std::fabs(expected_first) * 1e-4 + 1e-4) {
        if (mismatches == 0)
          std::printf("     first mismatch at %u: got %g, expected %g\n", i,
                      static_cast<double>(values[i]),
                      static_cast<double>(expected_first));
        ++mismatches;
      }
    }
    check(mismatches == 0,
          "all 64 values were computed from the uniform blocks and the push "
          "constant");

    // The two wrong answers above are the specific failures this test exists to
    // catch, and neither can be mistaken for the right one.
    check(std::fabs(expected_first - collision_value) > 1.0f &&
              std::fabs(expected_first - zero_push_value) > 1.0f,
          "the two blocks colliding, and the push constant never arriving, would "
          "both be visible here");
  }

  {
    write_buffer(first_memory, &second_inputs.first, sizeof(FirstBlock));
    std::vector<float> values;
    check(dispatch_and_read(second_inputs.push, &values),
          "a second dispatch with different push constants runs");

    int mismatches = 0;
    for (uint32_t i = 0; i < kComputeCount; ++i) {
      if (std::fabs(values[i] - expected_second) >
          std::fabs(expected_second) * 1e-4 + 1e-4)
        ++mismatches;
    }
    check(mismatches == 0,
          "the second dispatch used the new push constants (the buffer was "
          "refetched too)");
    check(std::fabs(expected_first - expected_second) > 1.0f,
          "the two dispatches genuinely ask for different answers");
  }

  // ---- the graphics phases -------------------------------------------------
  //
  // The attachment is cleared to nothing, so "the draw did not happen" and "the
  // draw wrote black" cannot be confused.
  const uint8_t kClearBytes[4] = {0, 0, 0, 0};
  const Tint kTint = {{1.0f, 0.75f, 0.25f, 0.875f}};
  const Weight kWeightA = {{1.0f, 0.5f, 0.25f, 0.25f}};
  const Weight kWeightB = {{1.0f, 1.0f, 1.0f, 1.0f}};

  uint8_t color_a[4] = {};
  uint8_t color_b[4] = {};
  expected_graphics_color(kTint, kWeightA, color_a);
  expected_graphics_color(kTint, kWeightB, color_b);

  write_buffer(tint_memory, &kTint, sizeof(Tint));

  struct GraphicsOptions {
    Transform transform;
    Weight weight;
    VkDescriptorSet vertex_descriptors = VK_NULL_HANDLE;
    bool draw = true;
  };

  auto render = [&](const GraphicsOptions& options,
                    std::vector<uint8_t>* out) -> bool {
    out->assign(kImageBytes, 0);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (begin_command_buffer(commands, &begin) != VK_SUCCESS)
      return false;

    const VkClearValue clear = clear_value(kClearBytes);
    VkRenderPassBeginInfo pass_begin{};
    pass_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    pass_begin.renderPass = render_pass;
    pass_begin.framebuffer = framebuffer;
    pass_begin.renderArea = {{0, 0}, {kWidth, kHeight}};
    pass_begin.clearValueCount = 1;
    pass_begin.pClearValues = &clear;
    cmd_begin_render_pass(commands, &pass_begin, VK_SUBPASS_CONTENTS_INLINE);

    if (options.draw) {
      cmd_bind_pipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        graphics_pipeline);
      const VkDescriptorSet sets[2] = {options.vertex_descriptors, fragment_set};
      cmd_bind_descriptor_sets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                               graphics_layout, 0, 2, sets, 0, nullptr);
      cmd_push_constants(commands, graphics_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         static_cast<uint32_t>(sizeof(Weight)), &options.weight);
      cmd_draw(commands, 3, 1, 0, 0);
    }

    cmd_end_render_pass(commands);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
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

    void* target = nullptr;
    if (map_memory(device, readback_memory, 0, VK_WHOLE_SIZE, 0, &target) !=
        VK_SUCCESS)
      return false;
    std::memcpy(out->data(), target, kImageBytes);
    unmap_memory(device, readback_memory);
    return true;
  };

  // The transform for each phase.  The identity covers the whole viewport; the
  // second moves the triangle's left edge to the middle and halves its width, so
  // the left quarter of the image is only drawn if the offset was ignored and the
  // left half only if both were.
  Transform identity{};
  identity.scale[0] = 1.0f;
  identity.scale[1] = 1.0f;

  Transform moved{};
  moved.offset[0] = 0.5f;
  moved.scale[0] = 0.5f;
  moved.scale[1] = 1.0f;

  // ---- phase 1: the uniform block reaches the vertex shader ----------------
  {
    GraphicsOptions options;
    options.transform = identity;
    options.weight = kWeightA;
    options.vertex_descriptors = vertex_set;

    write_buffer(transform_memory, &options.transform, sizeof(Transform));

    std::vector<uint8_t> pixels;
    check(render(options, &pixels), "phase 1 renders and reads back");

    // The identity transform leaves the shader's triangle exactly where the draw
    // test's gl_FragCoord phase has it: covering every texel centre.  So the
    // whole image is one flat colour, and a vertex shader that read nothing, or
    // read zeros, would leave it at the clear colour instead.
    int wrong = 0;
    uint32_t first_x = 0, first_y = 0;
    for (uint32_t y = 0; y < kHeight; ++y) {
      for (uint32_t x = 0; x < kWidth; ++x) {
        const uint8_t* actual = texel(pixels, x, y);
        if (std::abs(static_cast<int>(actual[0]) - color_a[0]) > 2 ||
            std::abs(static_cast<int>(actual[1]) - color_a[1]) > 2 ||
            std::abs(static_cast<int>(actual[2]) - color_a[2]) > 2 ||
            std::abs(static_cast<int>(actual[3]) - color_a[3]) > 2) {
          if (wrong == 0) {
            first_x = x;
            first_y = y;
          }
          ++wrong;
        }
      }
    }
    if (wrong != 0) {
      std::printf("     first wrong texel at (%u, %u): got %s, expected (%u, %u, "
                  "%u, %u)\n",
                  first_x, first_y, texel_text(texel(pixels, first_x, first_y)).c_str(),
                  color_a[0], color_a[1], color_a[2], color_a[3]);
    }
    check(wrong == 0,
          "all 4096 texels hold the colour the uniform block and the push "
          "constant ask for");
  }

  // ---- phase 2: the transform moves the geometry ---------------------------
  //
  // The full-image comparison is what makes this evidence: the coverage is
  // computed here from the transformed vertices, so a wrong uniform block shows
  // up as a texel that is drawn where the test says nothing should be, not as a
  // slightly odd triangle.
  {
    GraphicsOptions options;
    options.transform = moved;
    options.weight = kWeightA;
    options.vertex_descriptors = vertex_set;

    write_buffer(transform_memory, &options.transform, sizeof(Transform));

    std::vector<uint8_t> pixels;
    check(render(options, &pixels), "phase 2 renders and reads back");

    int wrong = 0;
    uint32_t first_x = 0, first_y = 0;
    for (uint32_t y = 0; y < kHeight; ++y) {
      for (uint32_t x = 0; x < kWidth; ++x) {
        const uint8_t* actual = texel(pixels, x, y);
        const uint8_t* expected =
            covered(moved, x + 0.5, y + 0.5) ? color_a : kClearBytes;
        for (int channel = 0; channel < 4; ++channel) {
          if (std::abs(static_cast<int>(actual[channel]) -
                       static_cast<int>(expected[channel])) > 2) {
            if (wrong == 0) {
              first_x = x;
              first_y = y;
            }
            ++wrong;
            break;
          }
        }
      }
    }
    if (wrong != 0) {
      std::printf("     first wrong texel at (%u, %u): got %s, expected %s\n",
                  first_x, first_y,
                  texel_text(texel(pixels, first_x, first_y)).c_str(),
                  texel_text(covered(moved, first_x + 0.5, first_y + 0.5) ? color_a
                                                                        : kClearBytes)
                      .c_str());
    }
    check(wrong == 0,
          "every texel agrees with the coverage computed here from the "
          "transformed vertices");

    // Three probes, named so that the output says which part of the transform is
    // missing if this fails: column 8 is outside either way, column 24 is inside
    // only if the offset was ignored, and column 56 is inside both.
    check(std::memcmp(texel(pixels, 8, 32), kClearBytes, 4) == 0,
          "phase 2: column 8 is outside the transformed triangle");
    check(std::memcmp(texel(pixels, 24, 32), kClearBytes, 4) == 0,
          "phase 2: column 24 is inside only if the offset was dropped, so it "
          "must be empty");
    check(std::abs(static_cast<int>(texel(pixels, 56, 32)[0]) -
                   static_cast<int>(color_a[0])) <= 2,
          "phase 2: column 56 is drawn");
  }

  // ---- phase 3: the push constant alone -----------------------------------
  //
  // The same pipeline and the same uniform block as phase 1, and only the push
  // constant differs.  A push constant that is uploaded once per pipeline, or
  // that never arrives at all, cannot pass both phases.
  {
    GraphicsOptions options;
    options.transform = identity;
    options.weight = kWeightB;
    options.vertex_descriptors = vertex_set;

    write_buffer(transform_memory, &options.transform, sizeof(Transform));

    std::vector<uint8_t> pixels;
    check(render(options, &pixels), "phase 3 renders and reads back");

    int wrong = 0;
    for (uint32_t y = 0; y < kHeight; ++y) {
      for (uint32_t x = 0; x < kWidth; ++x) {
        const uint8_t* actual = texel(pixels, x, y);
        for (int channel = 0; channel < 4; ++channel) {
          if (std::abs(static_cast<int>(actual[channel]) -
                       static_cast<int>(color_b[channel])) > 2) {
            ++wrong;
            break;
          }
        }
      }
    }
    check(wrong == 0,
          "all 4096 texels hold the colour the new push constant asks for");
    check(std::memcmp(color_a, color_b, 4) != 0,
          "phases 1 and 3 ask for different colours, so phase 3 is evidence "
          "rather than a repeat");
  }

  // ---- phase 4: a set whose binding was never written ----------------------
  //
  // Vulkan says a descriptor read before it is written is undefined, and the only
  // useful thing a driver can do about that is refuse rather than read whatever
  // the last command left behind.  The image must therefore still be the clear
  // colour: the alternative is a draw computed from the *previous* phase's
  // transform, which is exactly the kind of plausible-looking wrong answer this
  // project refuses to produce.
  {
    GraphicsOptions options;
    options.transform = identity;
    options.weight = kWeightA;
    options.vertex_descriptors = unwritten_vertex_set;

    std::vector<uint8_t> pixels;
    check(render(options, &pixels), "phase 4 renders and reads back");

    int wrong = 0;
    for (uint32_t y = 0; y < kHeight; ++y) {
      for (uint32_t x = 0; x < kWidth; ++x) {
        if (std::memcmp(texel(pixels, x, y), kClearBytes, 4) != 0)
          ++wrong;
      }
    }
    check(wrong == 0,
          "a draw whose uniform binding was never written is refused, leaving "
          "the clear colour");
  }

  std::printf("\n%s\n", failures() == 0 ? "all checks passed" : "FAILURES");
  tear_down(&runtime);
  return failures() == 0 ? 0 : 1;
}
