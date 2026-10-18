// M1 milestone test: offscreen compute, end to end, through the real loader.
//
// This is the gate for M1 because it is the first thing that exercises all three
// of the subsystems that could each have made the whole approach unworkable:
//
//   1. SPIR-V -> GLSL translation (would fail loudly at pipeline creation);
//   2. descriptor sets onto flat GL binding points (would fail silently, by
//      binding the wrong buffer to the wrong block);
//   3. the emulated VkDeviceMemory and its round trip through GL buffer objects
//      (would fail as stale or zeroed data).
//
// So the check is not "did it run" but "is the output bit for bit what a CPU
// reference computes".  A translation that silently mis-binds a descriptor
// produces plausible numbers rather than an error, and only a numeric
// comparison catches that.
//
// The shader is read from a file passed on the command line; CTest supplies the
// path.  Writing the words out by hand would have tied the test to a shader that
// no compiler produced, which is not what applications submit.

#include <vulkan/vulkan.h>

#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
  std::printf("  %-58s %s\n", what, ok ? "ok" : "FAILED");
  if (!ok)
    ++g_failures;
}

template <typename Fn, typename From>
Fn to_fn(From symbol) {
  static_assert(sizeof(Fn) == sizeof(From), "function pointer size mismatch");
  Fn fn = nullptr;
  std::memcpy(&fn, &symbol, sizeof(fn));
  return fn;
}

void require(const char* name, PFN_vkVoidFunction fn) {
  if (fn == nullptr) {
    std::printf("  missing entry point: %s\n", name);
    ++g_failures;
  }
}

PFN_vkVoidFunction (*g_get_instance_proc_addr)(VkInstance, const char*) = nullptr;
PFN_vkGetDeviceProcAddr g_get_device_proc_addr = nullptr;

template <typename Fn>
Fn instance_proc(VkInstance instance, const char* name) {
  return to_fn<Fn>(g_get_instance_proc_addr(instance, name));
}

template <typename Fn>
Fn device_proc(VkDevice device, const char* name) {
  return to_fn<Fn>(g_get_device_proc_addr(device, name));
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

// The reference implementation the GPU has to agree with.  Unsigned arithmetic
// wraps identically in GLSL, so this is an exact comparison, not a tolerance.
uint32_t reference(uint32_t value) {
  return value * 3u + 7u;
}

constexpr uint32_t kElementCount = 1024;
constexpr uint32_t kWorkgroupSize = 64;
constexpr uint32_t kBufferBytes = kElementCount * sizeof(uint32_t);

}  // namespace

int main(int argc, char** argv) {
  std::printf("kniz compute test (M1)\n");

  if (std::getenv("VK_ICD_FILENAMES") == nullptr) {
    std::fprintf(stderr,
                 "VK_ICD_FILENAMES is not set; the test would run against "
                 "whatever ICD is registered on this system\n");
    return 2;
  }

  const char* shader_path = argc > 1 ? argv[1] : "scale.comp.spv";
  const std::vector<uint32_t> spirv = read_spirv(shader_path);
  if (spirv.empty())
    return 2;

  void* loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  if (loader == nullptr) {
    std::fprintf(stderr, "cannot open libvulkan.so.1: %s\n", dlerror());
    return 2;
  }

  g_get_instance_proc_addr =
      to_fn<PFN_vkVoidFunction (*)(VkInstance, const char*)>(
          dlsym(loader, "vkGetInstanceProcAddr"));
  if (g_get_instance_proc_addr == nullptr) {
    std::fprintf(stderr, "loader has no vkGetInstanceProcAddr\n");
    return 2;
  }

  // ---- instance ------------------------------------------------------------
  auto create_instance = instance_proc<PFN_vkCreateInstance>(nullptr, "vkCreateInstance");
  if (create_instance == nullptr)
    return 2;

  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "kniz-compute";
  app.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo instance_info{};
  instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_info.pApplicationInfo = &app;

  VkInstance instance = VK_NULL_HANDLE;
  if (create_instance(&instance_info, nullptr, &instance) != VK_SUCCESS) {
    std::fprintf(stderr, "vkCreateInstance failed\n");
    return 1;
  }

  auto destroy_instance = instance_proc<PFN_vkDestroyInstance>(instance, "vkDestroyInstance");
  auto enumerate_physical =
      instance_proc<PFN_vkEnumeratePhysicalDevices>(instance, "vkEnumeratePhysicalDevices");
  auto get_properties =
      instance_proc<PFN_vkGetPhysicalDeviceProperties>(instance, "vkGetPhysicalDeviceProperties");
  auto get_memory_properties =
      instance_proc<PFN_vkGetPhysicalDeviceMemoryProperties>(instance, "vkGetPhysicalDeviceMemoryProperties");
  auto create_device = instance_proc<PFN_vkCreateDevice>(instance, "vkCreateDevice");

  // Device commands are reached through vkGetDeviceProcAddr, which is itself an
  // instance level command and has to be resolved before any device exists.
  g_get_device_proc_addr =
      instance_proc<PFN_vkGetDeviceProcAddr>(instance, "vkGetDeviceProcAddr");
  require("vkGetDeviceProcAddr",
          reinterpret_cast<PFN_vkVoidFunction>(g_get_device_proc_addr));

  uint32_t device_count = 0;
  enumerate_physical(instance, &device_count, nullptr);
  if (device_count == 0) {
    std::fprintf(stderr, "no physical devices\n");
    destroy_instance(instance, nullptr);
    return 1;
  }
  std::vector<VkPhysicalDevice> devices(device_count);
  enumerate_physical(instance, &device_count, devices.data());

  // Pick Kniz rather than assuming it is first: the loader may expose other
  // ICDs, and running this against the system's Intel driver would prove
  // nothing.
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  for (VkPhysicalDevice candidate : devices) {
    VkPhysicalDeviceProperties properties;
    get_properties(candidate, &properties);
    if (properties.vendorID == 0x4B4E) {
      physical = candidate;
      std::printf("     device     : %s\n", properties.deviceName);
      break;
    }
  }
  check(physical != VK_NULL_HANDLE, "a Kniz physical device is present");
  if (physical == VK_NULL_HANDLE) {
    destroy_instance(instance, nullptr);
    return 1;
  }

  float priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = 0;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;

  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;

  VkDevice device = VK_NULL_HANDLE;
  if (create_device(physical, &device_info, nullptr, &device) != VK_SUCCESS) {
    std::fprintf(stderr, "vkCreateDevice failed\n");
    destroy_instance(instance, nullptr);
    return 1;
  }

  // ---- device entry points -------------------------------------------------
  auto destroy_device = device_proc<PFN_vkDestroyDevice>(device, "vkDestroyDevice");
  auto get_queue = device_proc<PFN_vkGetDeviceQueue>(device, "vkGetDeviceQueue");
  auto create_buffer = device_proc<PFN_vkCreateBuffer>(device, "vkCreateBuffer");
  auto destroy_buffer = device_proc<PFN_vkDestroyBuffer>(device, "vkDestroyBuffer");
  auto get_buffer_requirements =
      device_proc<PFN_vkGetBufferMemoryRequirements>(device, "vkGetBufferMemoryRequirements");
  auto allocate_memory = device_proc<PFN_vkAllocateMemory>(device, "vkAllocateMemory");
  auto free_memory = device_proc<PFN_vkFreeMemory>(device, "vkFreeMemory");
  auto bind_buffer_memory = device_proc<PFN_vkBindBufferMemory>(device, "vkBindBufferMemory");
  auto map_memory = device_proc<PFN_vkMapMemory>(device, "vkMapMemory");
  auto unmap_memory = device_proc<PFN_vkUnmapMemory>(device, "vkUnmapMemory");
  auto create_shader_module = device_proc<PFN_vkCreateShaderModule>(device, "vkCreateShaderModule");
  auto destroy_shader_module = device_proc<PFN_vkDestroyShaderModule>(device, "vkDestroyShaderModule");
  auto create_set_layout = device_proc<PFN_vkCreateDescriptorSetLayout>(device, "vkCreateDescriptorSetLayout");
  auto destroy_set_layout = device_proc<PFN_vkDestroyDescriptorSetLayout>(device, "vkDestroyDescriptorSetLayout");
  auto create_pipeline_layout = device_proc<PFN_vkCreatePipelineLayout>(device, "vkCreatePipelineLayout");
  auto destroy_pipeline_layout = device_proc<PFN_vkDestroyPipelineLayout>(device, "vkDestroyPipelineLayout");
  auto create_compute_pipelines = device_proc<PFN_vkCreateComputePipelines>(device, "vkCreateComputePipelines");
  auto destroy_pipeline = device_proc<PFN_vkDestroyPipeline>(device, "vkDestroyPipeline");
  auto create_descriptor_pool = device_proc<PFN_vkCreateDescriptorPool>(device, "vkCreateDescriptorPool");
  auto destroy_descriptor_pool = device_proc<PFN_vkDestroyDescriptorPool>(device, "vkDestroyDescriptorPool");
  auto allocate_descriptor_sets = device_proc<PFN_vkAllocateDescriptorSets>(device, "vkAllocateDescriptorSets");
  auto update_descriptor_sets = device_proc<PFN_vkUpdateDescriptorSets>(device, "vkUpdateDescriptorSets");
  auto create_command_pool = device_proc<PFN_vkCreateCommandPool>(device, "vkCreateCommandPool");
  auto destroy_command_pool = device_proc<PFN_vkDestroyCommandPool>(device, "vkDestroyCommandPool");
  auto allocate_command_buffers = device_proc<PFN_vkAllocateCommandBuffers>(device, "vkAllocateCommandBuffers");
  auto begin_command_buffer = device_proc<PFN_vkBeginCommandBuffer>(device, "vkBeginCommandBuffer");
  auto end_command_buffer = device_proc<PFN_vkEndCommandBuffer>(device, "vkEndCommandBuffer");
  auto cmd_bind_pipeline = device_proc<PFN_vkCmdBindPipeline>(device, "vkCmdBindPipeline");
  auto cmd_bind_descriptor_sets = device_proc<PFN_vkCmdBindDescriptorSets>(device, "vkCmdBindDescriptorSets");
  auto cmd_dispatch = device_proc<PFN_vkCmdDispatch>(device, "vkCmdDispatch");
  auto cmd_fill_buffer = device_proc<PFN_vkCmdFillBuffer>(device, "vkCmdFillBuffer");
  auto cmd_update_buffer = device_proc<PFN_vkCmdUpdateBuffer>(device, "vkCmdUpdateBuffer");
  auto cmd_copy_buffer = device_proc<PFN_vkCmdCopyBuffer>(device, "vkCmdCopyBuffer");
  auto create_fence = device_proc<PFN_vkCreateFence>(device, "vkCreateFence");
  auto destroy_fence = device_proc<PFN_vkDestroyFence>(device, "vkDestroyFence");
  auto reset_fences = device_proc<PFN_vkResetFences>(device, "vkResetFences");
  auto wait_for_fences = device_proc<PFN_vkWaitForFences>(device, "vkWaitForFences");
  auto queue_submit = device_proc<PFN_vkQueueSubmit>(device, "vkQueueSubmit");
  auto device_wait_idle = device_proc<PFN_vkDeviceWaitIdle>(device, "vkDeviceWaitIdle");

  require("vkCreateCommandPool", reinterpret_cast<PFN_vkVoidFunction>(create_command_pool));
  require("vkCmdDispatch", reinterpret_cast<PFN_vkVoidFunction>(cmd_dispatch));
  require("vkCreateComputePipelines",
          reinterpret_cast<PFN_vkVoidFunction>(create_compute_pipelines));
  if (g_failures > 0) {
    destroy_device(device, nullptr);
    destroy_instance(instance, nullptr);
    return 1;
  }

  VkQueue queue = VK_NULL_HANDLE;
  get_queue(device, 0, 0, &queue);

  // ---- buffers and memory --------------------------------------------------
  // One buffer to read, one to write, plus a third to prove vkCmdCopyBuffer
  // moves data between GL buffer objects the same way.
  auto make_buffer = [&](VkBuffer* out) {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = kBufferBytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    return create_buffer(device, &info, nullptr, out);
  };

  VkBuffer input_buffer = VK_NULL_HANDLE;
  VkBuffer output_buffer = VK_NULL_HANDLE;
  VkBuffer copy_buffer = VK_NULL_HANDLE;
  check(make_buffer(&input_buffer) == VK_SUCCESS, "vkCreateBuffer (input)");
  check(make_buffer(&output_buffer) == VK_SUCCESS, "vkCreateBuffer (output)");
  check(make_buffer(&copy_buffer) == VK_SUCCESS, "vkCreateBuffer (copy target)");

  // The memory type the driver reports as mappable.  Asking through
  // vkGetPhysicalDeviceMemoryProperties rather than hardcoding the index keeps
  // the test honest about what the driver actually advertises.
  VkPhysicalDeviceMemoryProperties memory_properties;
  get_memory_properties(physical, &memory_properties);
  uint32_t host_type = 0xFFFFFFFFu;
  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
    if ((memory_properties.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
      host_type = i;
      break;
    }
  }
  check(host_type != 0xFFFFFFFFu, "driver reports a host visible memory type");

  auto allocate_for = [&](VkBuffer buffer, VkDeviceMemory* out) {
    VkMemoryRequirements requirements{};
    get_buffer_requirements(device, buffer, &requirements);
    VkMemoryAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.allocationSize = requirements.size;
    info.memoryTypeIndex = host_type;
    if (allocate_memory(device, &info, nullptr, out) != VK_SUCCESS)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    return bind_buffer_memory(device, buffer, *out, 0);
  };

  VkDeviceMemory input_memory = VK_NULL_HANDLE;
  VkDeviceMemory output_memory = VK_NULL_HANDLE;
  VkDeviceMemory copy_memory = VK_NULL_HANDLE;
  check(allocate_for(input_buffer, &input_memory) == VK_SUCCESS,
        "input buffer bound to memory");
  check(allocate_for(output_buffer, &output_memory) == VK_SUCCESS,
        "output buffer bound to memory");
  check(allocate_for(copy_buffer, &copy_memory) == VK_SUCCESS,
        "copy buffer bound to memory");

  // Application writes go through the mapped pointer, exactly as a real
  // application would.  Keeping the expected values on the host means the
  // reference can be recomputed after the buffer is unmapped.
  std::vector<uint32_t> expected(kElementCount);
  void* mapped = nullptr;
  if (map_memory(device, input_memory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
    auto* values = static_cast<uint32_t*>(mapped);
    for (uint32_t i = 0; i < kElementCount; ++i) {
      values[i] = i * 2654435761u + 12345u;
      expected[i] = reference(values[i]);
    }
    unmap_memory(device, input_memory);
    check(true, "input written through vkMapMemory");
  } else {
    check(false, "vkMapMemory");
  }

  // ---- shader, layouts and pipeline ---------------------------------------
  VkShaderModuleCreateInfo module_info{};
  module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  module_info.codeSize = spirv.size() * sizeof(uint32_t);
  module_info.pCode = spirv.data();

  VkShaderModule module = VK_NULL_HANDLE;
  check(create_shader_module(device, &module_info, nullptr, &module) == VK_SUCCESS,
        "vkCreateShaderModule");

  VkDescriptorSetLayoutBinding bindings[2]{};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo set_layout_info{};
  set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  set_layout_info.bindingCount = 2;
  set_layout_info.pBindings = bindings;

  VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
  check(create_set_layout(device, &set_layout_info, nullptr, &set_layout) == VK_SUCCESS,
        "vkCreateDescriptorSetLayout");

  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &set_layout;

  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  check(create_pipeline_layout(device, &pipeline_layout_info, nullptr, &pipeline_layout) ==
            VK_SUCCESS,
        "vkCreatePipelineLayout");

  VkPipelineShaderStageCreateInfo stage{};
  stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = module;
  stage.pName = "main";

  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage = stage;
  pipeline_info.layout = pipeline_layout;

  VkPipeline pipeline = VK_NULL_HANDLE;
  check(create_compute_pipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                 &pipeline) == VK_SUCCESS &&
            pipeline != VK_NULL_HANDLE,
        "vkCreateComputePipelines (SPIR-V translated and compiled)");

  // The M1 boundary, asserted rather than merely documented: a graphics
  // pipeline must be refused clearly instead of producing something that does
  // not draw.
  auto create_graphics_pipelines =
      device_proc<PFN_vkCreateGraphicsPipelines>(device, "vkCreateGraphicsPipelines");
  if (create_graphics_pipelines != nullptr) {
    VkGraphicsPipelineCreateInfo graphics{};
    graphics.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    VkPipeline rejected = VK_NULL_HANDLE;
    const VkResult result = create_graphics_pipelines(device, VK_NULL_HANDLE, 1, &graphics,
                                                      nullptr, &rejected);
    check(result == VK_ERROR_FEATURE_NOT_PRESENT && rejected == VK_NULL_HANDLE,
          "graphics pipelines are refused, not silently ignored");
  }

  if (pipeline == VK_NULL_HANDLE) {
    destroy_device(device, nullptr);
    destroy_instance(instance, nullptr);
    return 1;
  }

  // ---- descriptor set ------------------------------------------------------
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = 2;

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;

  VkDescriptorPool pool = VK_NULL_HANDLE;
  check(create_descriptor_pool(device, &pool_info, nullptr, &pool) == VK_SUCCESS,
        "vkCreateDescriptorPool");

  VkDescriptorSetAllocateInfo set_info{};
  set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  set_info.descriptorPool = pool;
  set_info.descriptorSetCount = 1;
  set_info.pSetLayouts = &set_layout;

  VkDescriptorSet set = VK_NULL_HANDLE;
  check(allocate_descriptor_sets(device, &set_info, &set) == VK_SUCCESS, "vkAllocateDescriptorSets");

  VkDescriptorBufferInfo input_info{};
  input_info.buffer = input_buffer;
  input_info.offset = 0;
  input_info.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo output_info{};
  output_info.buffer = output_buffer;
  output_info.offset = 0;
  output_info.range = VK_WHOLE_SIZE;

  VkWriteDescriptorSet writes[2]{};
  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = set;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &input_info;
  writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[1].dstSet = set;
  writes[1].dstBinding = 1;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[1].pBufferInfo = &output_info;
  update_descriptor_sets(device, 2, writes, 0, nullptr);
  check(true, "vkUpdateDescriptorSets");

  // ---- command pool and buffer --------------------------------------------
  VkCommandPoolCreateInfo command_pool_info{};
  command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  command_pool_info.queueFamilyIndex = 0;

  VkCommandPool command_pool = VK_NULL_HANDLE;
  check(create_command_pool(device, &command_pool_info, nullptr, &command_pool) == VK_SUCCESS,
        "vkCreateCommandPool");

  VkCommandBufferAllocateInfo command_info{};
  command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_info.commandPool = command_pool;
  command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_info.commandBufferCount = 1;

  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  check(allocate_command_buffers(device, &command_info, &command_buffer) == VK_SUCCESS,
        "vkAllocateCommandBuffers");

  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence = VK_NULL_HANDLE;
  create_fence(device, &fence_info, nullptr, &fence);

  const uint32_t group_count = (kElementCount + kWorkgroupSize - 1) / kWorkgroupSize;

  // Records bind + dispatch into cmd, submits it and waits.
  auto run = [&](VkCommandBuffer cmd) {
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    reset_fences(device, 1, &fence);
    if (queue_submit(queue, 1, &submit, fence) != VK_SUCCESS)
      return false;
    return wait_for_fences(device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
  };

  // ---- phase 1: dispatch and compare --------------------------------------
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_command_buffer(command_buffer, &begin);
  cmd_bind_pipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  cmd_bind_descriptor_sets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                           &set, 0, nullptr);
  cmd_dispatch(command_buffer, group_count, 1, 1);
  end_command_buffer(command_buffer);
  check(run(command_buffer), "vkQueueSubmit + vkWaitForFences");
  device_wait_idle(device);

  std::vector<uint32_t> results(kElementCount, 0);
  if (map_memory(device, output_memory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
    std::memcpy(results.data(), mapped, kBufferBytes);
    unmap_memory(device, output_memory);
  }

  uint32_t mismatches = 0;
  uint32_t first_bad = 0;
  for (uint32_t i = 0; i < kElementCount; ++i) {
    if (results[i] != expected[i]) {
      if (mismatches == 0)
        first_bad = i;
      ++mismatches;
    }
  }
  check(mismatches == 0, "compute output matches the CPU reference");
  if (mismatches != 0) {
    std::printf("     %u of %u elements differ; first at %u: got %u, expected %u\n",
                mismatches, kElementCount, first_bad, results[first_bad],
                expected[first_bad]);
  }

  // ---- phase 2: vkCmdCopyBuffer moves the result to another buffer ---------
  begin_command_buffer(command_buffer, &begin);
  VkBufferCopy region{};
  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = kBufferBytes;
  cmd_copy_buffer(command_buffer, output_buffer, copy_buffer, 1, &region);
  end_command_buffer(command_buffer);
  check(run(command_buffer), "vkCmdCopyBuffer submitted");

  std::vector<uint32_t> copied(kElementCount, 0);
  if (map_memory(device, copy_memory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
    std::memcpy(copied.data(), mapped, kBufferBytes);
    unmap_memory(device, copy_memory);
  }
  check(std::memcmp(copied.data(), expected.data(), kBufferBytes) == 0,
        "copied buffer matches the reference too");

  // ---- phase 3: vkCmdFillBuffer -------------------------------------------
  begin_command_buffer(command_buffer, &begin);
  cmd_fill_buffer(command_buffer, output_buffer, 0, kBufferBytes, 0xA5A5A5A5u);
  end_command_buffer(command_buffer);
  check(run(command_buffer), "vkCmdFillBuffer submitted");

  std::fill(results.begin(), results.end(), 0u);
  if (map_memory(device, output_memory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
    std::memcpy(results.data(), mapped, kBufferBytes);
    unmap_memory(device, output_memory);
  }
  bool filled = true;
  for (uint32_t value : results)
    filled = filled && value == 0xA5A5A5A5u;
  check(filled, "vkCmdFillBuffer pattern read back from the GL buffer");

  // ---- phase 4: vkCmdUpdateBuffer, then dispatch again ---------------------
  // Also checks that a second submission sees the first submission's GPU writes:
  // the input buffer is overwritten here and the output buffer still holds the
  // fill pattern, and neither of those facts comes from host memory.
  {
    std::vector<uint32_t> new_input(kElementCount);
    for (uint32_t i = 0; i < kElementCount; ++i) {
      new_input[i] = 0xDEADBEEFu - i;
      expected[i] = reference(new_input[i]);
    }

    begin_command_buffer(command_buffer, &begin);
    cmd_update_buffer(command_buffer, input_buffer, 0, kBufferBytes, new_input.data());
    cmd_bind_pipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    cmd_bind_descriptor_sets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                             &set, 0, nullptr);
    cmd_dispatch(command_buffer, group_count, 1, 1);
    end_command_buffer(command_buffer);
    check(run(command_buffer), "vkCmdUpdateBuffer + dispatch submitted");

    std::fill(results.begin(), results.end(), 0u);
    if (map_memory(device, output_memory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
      std::memcpy(results.data(), mapped, kBufferBytes);
      unmap_memory(device, output_memory);
    }
    check(std::memcmp(results.data(), expected.data(), kBufferBytes) == 0,
          "second dispatch matches the reference as well");
  }

  // ---- teardown ------------------------------------------------------------
  device_wait_idle(device);
  destroy_fence(device, fence, nullptr);
  destroy_command_pool(device, command_pool, nullptr);
  destroy_descriptor_pool(device, pool, nullptr);
  destroy_pipeline(device, pipeline, nullptr);
  destroy_pipeline_layout(device, pipeline_layout, nullptr);
  destroy_set_layout(device, set_layout, nullptr);
  destroy_shader_module(device, module, nullptr);
  destroy_buffer(device, copy_buffer, nullptr);
  destroy_buffer(device, output_buffer, nullptr);
  destroy_buffer(device, input_buffer, nullptr);
  free_memory(device, copy_memory, nullptr);
  free_memory(device, output_memory, nullptr);
  free_memory(device, input_memory, nullptr);
  destroy_device(device, nullptr);
  destroy_instance(instance, nullptr);
  check(true, "teardown");

  if (g_failures == 0) {
    std::printf("PASSED\n");
    return 0;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return 1;
}
