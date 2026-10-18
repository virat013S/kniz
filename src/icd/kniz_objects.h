#pragma once

// The Vulkan object model.
//
// Vulkan handles are opaque.  Kniz hands the application pointers to the
// structures below, wrapped in the real Vulkan handle types, and recovers them
// on the way back in.  Every object starts with a magic value so a stale handle
// -- or one belonging to another driver -- is rejected rather than
// dereferenced.
//
// On 64-bit platforms both dispatchable and non-dispatchable Vulkan handles are
// pointer sized, so pointers round-trip losslessly.  Kniz targets 64-bit only
// for now; a 32-bit build would need non-dispatchable handles to come from a
// table keyed by index instead.
//
// Deliberately absent: a dispatch table inside each object.  An ICD does not
// need one.  The loader assembles its own dispatch from the function pointers
// we hand back in vk_icdGetInstanceProcAddr, so there is nothing for us to
// store per object.

#include "kniz_common.h"
#include "kniz_gl.h"

#include <vulkan/vk_icd.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace kniz {

inline constexpr uint64_t kInstanceMagic = 0x4B4E495A494E5354ull;  // KNIZINST
inline constexpr uint64_t kPhysicalMagic = 0x4B4E495A50485944ull;  // KNIZPHYD
inline constexpr uint64_t kDeviceMagic = 0x4B4E495A44455649ull;    // KNIZDEVI
inline constexpr uint64_t kQueueMagic = 0x4B4E495A51554555ull;     // KNIZQUEU

inline constexpr uint64_t kShaderModuleMagic = 0x4B4E495A53484D4Full;  // KNIZSHMO
inline constexpr uint64_t kMemoryMagic = 0x4B4E495A4D454D4Full;        // KNIZMEMO
inline constexpr uint64_t kBufferMagic = 0x4B4E495A42554646ull;        // KNIZBUFF
inline constexpr uint64_t kSetLayoutMagic = 0x4B4E495A44534C59ull;     // KNIZDSLY
inline constexpr uint64_t kPipelineLayoutMagic = 0x4B4E495A504C4159ull;  // KNIZPLAY
inline constexpr uint64_t kDescriptorPoolMagic = 0x4B4E495A44504F4Full;  // KNIZDPOO
inline constexpr uint64_t kDescriptorSetMagic = 0x4B4E495A44534554ull;   // KNIZDSET
inline constexpr uint64_t kPipelineMagic = 0x4B4E495A50495045ull;        // KNIZPIPE
inline constexpr uint64_t kPipelineCacheMagic = 0x4B4E495A50434143ull;   // KNIZPCAC
inline constexpr uint64_t kCommandPoolMagic = 0x4B4E495A43504F4Full;     // KNIZCPOO
inline constexpr uint64_t kCommandBufferMagic = 0x4B4E495A434D4442ull;   // KNIZCMDB
inline constexpr uint64_t kFenceMagic = 0x4B4E495A46454E43ull;           // KNIZFENC

struct Instance;
struct PhysicalDevice;
struct Device;
struct Queue;
struct ShaderModule;
struct DeviceMemory;
struct Buffer;
struct DescriptorSetLayout;
struct PipelineLayout;
struct DescriptorPool;
struct DescriptorSet;
struct Pipeline;
struct CommandPool;
struct CommandBuffer;
struct Fence;
struct PipelineCache;

// ---------------------------------------------------------------------------
// Objects
//
// THE FIRST FIELD OF EVERY DISPATCHABLE OBJECT IS RESERVED.
//
// Vulkan puts a dispatch table pointer at offset 0 of every dispatchable
// handle.  For an ICD, the loader owns those bytes: after vkCreateDevice it
// executes
//
//     loader_init_dispatch(*pDevice, &dev->loader_dispatch);   // writes offset 0
//
// and it assumes the same layout when reading a handle back.  Meanwhile
// vulkan/vk_icd.h requires the ICD to initialise that first word with
// ICD_LOADER_MAGIC (via set_loader_magic_value), which the loader checks in
// debug builds.
//
// So the opening field must hold a VK_LOADER_DATA initialised by
// set_loader_magic_value(), must never be read by us, and nothing of ours may
// live there.  Our own bookkeeping -- including the magic used to validate a
// handle -- starts immediately after it.  Getting this wrong produces a driver
// whose handles appear valid until the loader overwrites them, at which point
// every device entry point silently rejects its own device.
//
// VkCommandBuffer is dispatchable too, and the loader writes into it during
// vkAllocateCommandBuffers, so it carries the same reservation.
// ---------------------------------------------------------------------------

struct Instance {
  VK_LOADER_DATA loader_data;  // reserved by the loader; must be first

  uint64_t magic = kInstanceMagic;

  // What the application asked for, and what we will actually honour.  Kniz
  // implements Vulkan 1.0 semantics until the corresponding work lands, and
  // reports that honestly rather than advertising a version whose entry points
  // would return VK_ERROR_FEATURE_NOT_PRESENT.
  uint32_t api_version = VK_API_VERSION_1_0;
  uint32_t effective_api_version = VK_API_VERSION_1_0;

  bool gl_ready = false;
  gl::ContextInfo gl{};

  // Exactly one physical device per instance.  It lives inline so that
  // VkPhysicalDevice stays stable across vkEnumeratePhysicalDevices calls, as
  // the specification requires.
  PhysicalDevice* physical = nullptr;
};

struct PhysicalDevice {
  VK_LOADER_DATA loader_data;  // reserved by the loader; must be first

  uint64_t magic = kPhysicalMagic;
  Instance* instance = nullptr;

  // Index within the instance, i.e. the value returned by
  // vkEnumeratePhysicalDevices.  vkCreateDevice must be handed the same index
  // back, so it is stored rather than recomputed.
  uint32_t index = 0;

  // Device identity, currently derived from the GL adapter we picked up.
  uint32_t vendor_id = 0;
  uint32_t device_id = 0;
};

struct Queue {
  // VkQueue is a non-dispatchable handle, but on 64-bit the loader still stores
  // a dispatch pointer at offset 0 of it (see loader_set_dispatch(*pQueue, ...)
  // in the loader's vkGetDeviceQueue trampoline), so the same reservation
  // applies.
  VK_LOADER_DATA loader_data;  // reserved by the loader; must be first

  uint64_t magic = kQueueMagic;
  Device* device = nullptr;
  uint32_t family_index = 0;
  uint32_t queue_index = 0;
};

struct Device {
  VK_LOADER_DATA loader_data;  // reserved by the loader; must be first

  uint64_t magic = kDeviceMagic;
  Instance* instance = nullptr;
  PhysicalDevice* physical = nullptr;

  // One queue family, one queue.
  Queue queue{};

  // Every live buffer.  Submission walks this to move data between the
  // application's host memory and the GL buffer objects.  See kniz_memory.cpp
  // for why that is necessary and why it is done the blunt way for now.
  std::vector<Buffer*> buffers;
};

// ---------------------------------------------------------------------------
// M1 objects
// ---------------------------------------------------------------------------

// SPIR-V as handed to vkCreateShaderModule.  Deliberately not translated yet:
// the GL binding points a shader uses depend on the pipeline layout, which is
// only known at pipeline creation.  See kniz_shader.cpp.
struct ShaderModule {
  uint64_t magic = kShaderModuleMagic;
  Device* device = nullptr;
  std::vector<uint32_t> spirv;
  std::string name;
};

// VkDeviceMemory has no OpenGL analogue.  Kniz therefore *is* the allocator:
// the allocation is a host block, and buffers are copied into and out of it
// around each submission.  That is slow and it is not what the Vulkan memory
// model promises -- see the fidelity budget in docs/DESIGN.md -- but with the
// single host-visible memory type this driver reports, it is correct.
struct DeviceMemory {
  uint64_t magic = kMemoryMagic;
  Device* device = nullptr;

  uint64_t size = 0;
  uint32_t type_index = 0;
  VkMemoryPropertyFlags properties = 0;

  uint8_t* host = nullptr;

  // Bumped on every unmap.  A buffer whose plane of the allocation may have
  // changed re-uploads when this no longer matches what it last uploaded.
  uint64_t version = 0;
  uint32_t map_count = 0;
};

struct Buffer {
  uint64_t magic = kBufferMagic;
  Device* device = nullptr;

  uint64_t size = 0;
  VkBufferUsageFlags usage = 0;
  gl::Uint gl_name = 0;

  // Bound memory, if any.  Vulkan allows a buffer to exist unbound, and
  // mapping or recording against one is an application error rather than
  // something to guard against here.
  DeviceMemory* memory = nullptr;
  uint64_t memory_offset = 0;

  uint64_t uploaded_version = 0;
};

struct DescriptorSetLayoutBinding {
  uint32_t binding = 0;
  VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  uint32_t count = 0;
  VkShaderStageFlags stages = 0;
};

struct DescriptorSetLayout {
  uint64_t magic = kSetLayoutMagic;
  Device* device = nullptr;
  std::vector<DescriptorSetLayoutBinding> bindings;

  // Largest binding number plus one, so a set can index its bindings directly.
  uint32_t binding_slots = 0;
};

struct PipelineLayout {
  uint64_t magic = kPipelineLayoutMagic;
  Device* device = nullptr;
  std::vector<DescriptorSetLayout*> sets;
};

// Where one shader resource lives in OpenGL.
//
// Vulkan addresses a descriptor by (set, binding); GLSL has one flat binding
// number per resource class and no notion of sets at all.  Because Kniz
// generates the GLSL itself, it can rewrite each resource to a dense GL binding
// point, and this record is the mapping it used -- the pipeline's shader and
// the dispatch path both read it, so they cannot disagree.
struct GlBinding {
  uint32_t set = 0;
  uint32_t binding = 0;
  uint32_t gl_index = 0;
};

struct DescriptorBinding {
  bool bound = false;
  Buffer* buffer = nullptr;
  uint64_t offset = 0;
  uint64_t range = 0;
};

struct DescriptorSet {
  uint64_t magic = kDescriptorSetMagic;
  Device* device = nullptr;
  DescriptorPool* pool = nullptr;
  DescriptorSetLayout* layout = nullptr;

  // Indexed by binding number; see DescriptorSetLayout::binding_slots.
  std::vector<DescriptorBinding> bindings;
};

struct DescriptorPool {
  uint64_t magic = kDescriptorPoolMagic;
  Device* device = nullptr;
  uint32_t max_sets = 0;
  std::vector<DescriptorSet*> allocated;
};

struct Pipeline {
  uint64_t magic = kPipelineMagic;
  Device* device = nullptr;
  VkPipelineBindPoint bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
  PipelineLayout* layout = nullptr;

  gl::Uint program = 0;
  std::vector<GlBinding> bindings;

  // Kept for diagnostics: when a shader fails to compile, this is the GLSL that
  // GL rejected, and reading it is usually faster than reading the SPIR-V.
  std::string glsl;
};

// ---------------------------------------------------------------------------
// The recorded command stream (ADR-1)
//
// A Vulkan command buffer is an object the application may record once and
// submit many times, from a different thread than the one that recorded it.
// OpenGL commands must instead be issued immediately, on a thread with the
// context current.  Kniz therefore records into this small IR and replays it
// inside vkQueueSubmit.  One struct covers the whole M1 command set, which is
// small enough that a tagged union would cost more in clarity than it saves.
// ---------------------------------------------------------------------------

struct RecordedCommand {
  enum class Kind {
    Dispatch,
    FillBuffer,
    UpdateBuffer,
    CopyBuffer,
    // A Vulkan pipeline barrier expresses ordering that GL reaches by executing
    // in order with an explicit memory barrier.  Dropping it would be wrong, so
    // it is recorded and replayed as glMemoryBarrier.
    Barrier,
  };

  Kind kind = Kind::Dispatch;

  // Dispatch.  Resolved when vkCmdDispatch is recorded rather than replayed
  // from state, because that is exactly when Vulkan says the bound resources
  // are chosen.
  Pipeline* pipeline = nullptr;
  std::vector<DescriptorSet*> sets;  // indexed by set number
  uint32_t group_count[3] = {1, 1, 1};

  // Buffer operations.  `offset` is the destination offset for every kind;
  // `src_offset` is the read offset for a copy.
  Buffer* dst = nullptr;
  Buffer* src = nullptr;
  uint64_t offset = 0;
  uint64_t src_offset = 0;
  uint64_t size = 0;
  uint32_t data = 0;                 // FillBuffer's 32-bit pattern
  std::vector<uint8_t> bytes;        // UpdateBuffer's payload
};

struct CommandPool {
  uint64_t magic = kCommandPoolMagic;
  Device* device = nullptr;
  uint32_t family_index = 0;
  bool resettable = true;
  std::vector<CommandBuffer*> buffers;
};

struct CommandBuffer {
  VK_LOADER_DATA loader_data;  // reserved by the loader; see the note above

  uint64_t magic = kCommandBufferMagic;
  Device* device = nullptr;
  CommandPool* pool = nullptr;

  bool recording = false;
  bool executable = false;

  std::vector<RecordedCommand> commands;

  // Recording-time state.  vkCmdDispatch captures it, so that binding and
  // dispatching stay independent of replay order.
  Pipeline* bound_pipeline = nullptr;
  PipelineLayout* bound_layout = nullptr;
  std::vector<DescriptorSet*> bound_sets;
};

struct Fence {
  uint64_t magic = kFenceMagic;
  Device* device = nullptr;

  // Kniz executes submissions synchronously, so a fence created unsignaled is
  // already signalled by the time vkQueueSubmit returns.  See kniz_command.cpp.
  bool signaled = false;
};

// Kniz does not cache anything across pipelines: it compiles GLSL on every
// vkCreateComputePipelines call and the driver's own program cache does the
// real work.  The handle exists because applications create one unconditionally
// and would otherwise be handed a null.
struct PipelineCache {
  uint64_t magic = kPipelineCacheMagic;
  Device* device = nullptr;
};

// ---------------------------------------------------------------------------
// Handle conversions
// ---------------------------------------------------------------------------

namespace detail {

template <typename T>
inline T* to_object(uint64_t magic, void* handle) {
  if (handle == nullptr)
    return nullptr;
  auto* object = static_cast<T*>(handle);
  if (object->magic != magic)
    return nullptr;
  return object;
}

}  // namespace detail

// Each accessor returns nullptr for a null handle, an uninitialised handle, or
// a handle that does not belong to Kniz.
inline Instance* from_handle(VkInstance handle) {
  return detail::to_object<Instance>(kInstanceMagic, handle);
}

inline PhysicalDevice* from_handle(VkPhysicalDevice handle) {
  return detail::to_object<PhysicalDevice>(kPhysicalMagic, handle);
}

inline Device* from_handle(VkDevice handle) {
  return detail::to_object<Device>(kDeviceMagic, handle);
}

inline Queue* from_handle(VkQueue handle) {
  return detail::to_object<Queue>(kQueueMagic, handle);
}

inline ShaderModule* from_handle(VkShaderModule handle) {
  return detail::to_object<ShaderModule>(kShaderModuleMagic, handle);
}

inline DeviceMemory* from_handle(VkDeviceMemory handle) {
  return detail::to_object<DeviceMemory>(kMemoryMagic, handle);
}

inline Buffer* from_handle(VkBuffer handle) {
  return detail::to_object<Buffer>(kBufferMagic, handle);
}

inline DescriptorSetLayout* from_handle(VkDescriptorSetLayout handle) {
  return detail::to_object<DescriptorSetLayout>(kSetLayoutMagic, handle);
}

inline PipelineLayout* from_handle(VkPipelineLayout handle) {
  return detail::to_object<PipelineLayout>(kPipelineLayoutMagic, handle);
}

inline DescriptorPool* from_handle(VkDescriptorPool handle) {
  return detail::to_object<DescriptorPool>(kDescriptorPoolMagic, handle);
}

inline DescriptorSet* from_handle(VkDescriptorSet handle) {
  return detail::to_object<DescriptorSet>(kDescriptorSetMagic, handle);
}

inline Pipeline* from_handle(VkPipeline handle) {
  return detail::to_object<Pipeline>(kPipelineMagic, handle);
}

inline CommandPool* from_handle(VkCommandPool handle) {
  return detail::to_object<CommandPool>(kCommandPoolMagic, handle);
}

inline CommandBuffer* from_handle(VkCommandBuffer handle) {
  return detail::to_object<CommandBuffer>(kCommandBufferMagic, handle);
}

inline Fence* from_handle(VkFence handle) {
  return detail::to_object<Fence>(kFenceMagic, handle);
}

inline PipelineCache* from_handle(VkPipelineCache handle) {
  return detail::to_object<PipelineCache>(kPipelineCacheMagic, handle);
}

// The inverse conversions are equally direct.
inline VkInstance to_handle(Instance* object) {
  return reinterpret_cast<VkInstance>(object);
}

inline VkPhysicalDevice to_handle(PhysicalDevice* object) {
  return reinterpret_cast<VkPhysicalDevice>(object);
}

inline VkDevice to_handle(Device* object) {
  return reinterpret_cast<VkDevice>(object);
}

inline VkQueue to_handle(Queue* object) {
  return reinterpret_cast<VkQueue>(object);
}

inline VkShaderModule to_handle(ShaderModule* object) {
  return reinterpret_cast<VkShaderModule>(object);
}

inline VkDeviceMemory to_handle(DeviceMemory* object) {
  return reinterpret_cast<VkDeviceMemory>(object);
}

inline VkBuffer to_handle(Buffer* object) {
  return reinterpret_cast<VkBuffer>(object);
}

inline VkDescriptorSetLayout to_handle(DescriptorSetLayout* object) {
  return reinterpret_cast<VkDescriptorSetLayout>(object);
}

inline VkPipelineLayout to_handle(PipelineLayout* object) {
  return reinterpret_cast<VkPipelineLayout>(object);
}

inline VkDescriptorPool to_handle(DescriptorPool* object) {
  return reinterpret_cast<VkDescriptorPool>(object);
}

inline VkDescriptorSet to_handle(DescriptorSet* object) {
  return reinterpret_cast<VkDescriptorSet>(object);
}

inline VkPipeline to_handle(Pipeline* object) {
  return reinterpret_cast<VkPipeline>(object);
}

inline VkCommandPool to_handle(CommandPool* object) {
  return reinterpret_cast<VkCommandPool>(object);
}

inline VkCommandBuffer to_handle(CommandBuffer* object) {
  return reinterpret_cast<VkCommandBuffer>(object);
}

inline VkFence to_handle(Fence* object) {
  return reinterpret_cast<VkFence>(object);
}

inline VkPipelineCache to_handle(PipelineCache* object) {
  return reinterpret_cast<VkPipelineCache>(object);
}

}  // namespace kniz
