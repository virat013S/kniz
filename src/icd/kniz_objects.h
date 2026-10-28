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
inline constexpr uint64_t kImageMagic = 0x4B4E495A494D4147ull;           // KNIZIMAG
inline constexpr uint64_t kImageViewMagic = 0x4B4E495A49565757ull;       // KNIZIVWW
inline constexpr uint64_t kSamplerMagic = 0x4B4E495A534D504Cull;         // KNIZSMPL

inline constexpr uint64_t kRenderPassMagic = 0x4B4E495A52504C5Aull;     // KNIZRPLZ
inline constexpr uint64_t kFramebufferMagic = 0x4B4E495A4652414Dull;     // KNIZFRAM

inline constexpr uint64_t kSurfaceMagic = 0x4B4E495A53555246ull;         // KNIZSURF
inline constexpr uint64_t kSwapchainMagic = 0x4B4E495A53574150ull;       // KNIZSWAP

// The largest push constant block Kniz accepts, and what it reports as
// maxPushConstantsSize.  Vulkan's own minimum for that limit is 128, so a
// driver cannot advertise less; this is the floor, adopted rather than an
// arbitrary choice.  GL has no push constants at all, so they are lowered onto
// a uniform buffer of exactly this size.
inline constexpr uint32_t kMaxPushConstantsSize = 128;

// A (set, binding) that does not exist.  Used where a field is optional, so that
// "no sampler" is distinguishable from "set 0, binding 0".
inline constexpr uint32_t kNoBinding = 0xFFFFFFFFu;

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
struct Image;
struct ImageView;
struct Sampler;
struct RenderPass;
struct Framebuffer;
struct Surface;
struct Swapchain;

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

  // The uniform buffer that backs push constants, which GL does not have.  One
  // is enough for the whole device because replay is synchronous: it is
  // uploaded to and bound at the reserved binding point as the recorded
  // vkCmdPushConstants commands replay, so its contents are only ever live
  // during the replay that wrote them.
  gl::Uint push_constant_buffer = 0;
  uint32_t push_constant_capacity = 0;
};

// True when the device is backed by a live GL context.  Every device entry point
// that is about to touch GL asks this first, so a missing context is reported
// once, in the same words, instead of surfacing later as an unattributable GL
// error or a crash.
inline bool device_can_run_gl(Device* device) {
  if (device == nullptr)
    return false;
  if (!device->instance->gl_ready) {
    KNIZ_ERR("no OpenGL context is available");
    return false;
  }
  return true;
}

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

  // Images bound to this allocation.  An image is not mirrored into `host`, so
  // mapping an allocation that holds one would hand the application a pointer to
  // bytes that are not the image.  vkMapMemory refuses when this is non-zero.
  uint32_t images_bound = 0;
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

// Where one shader resource lives in OpenGL.
//
// Vulkan addresses a descriptor by (set, binding); GLSL has one flat binding
// number per resource class and no notion of sets at all.  Because Kniz
// generates the GLSL itself, it can rewrite each resource to a dense GL binding
// point, and this record is the mapping it used -- the pipeline's shader and the
// replay path both read it, so they cannot disagree.
struct GlBinding {
  uint32_t set = 0;
  uint32_t binding = 0;
  uint32_t gl_index = 0;

  // Which GL binding namespace the index refers to.  GL keeps uniform buffers,
  // shader storage buffers and texture units in separate spaces that are indexed
  // independently, so each class is numbered from zero.
  VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

  // Set when this is a sampled image whose shader pairs it with a *separate*
  // sampler descriptor: the (set, binding) of that sampler.  GLSL has one
  // combined sampler for the pair, and it lives at the image's texture unit, so
  // the sampler object has to be bound to that same unit -- while the sampler
  // descriptor's own unit is one nothing reads.  kNoBinding when the image is
  // used through a combined image sampler, which carries its own.
  uint32_t sampler_set = kNoBinding;
  uint32_t sampler_binding = kNoBinding;
};

// GL's resource classes, each with its own flat binding namespace.  A uniform
// buffer and a storage buffer may both be at binding 0 in one layout, and a
// texture unit is a third namespace again.
enum class DescriptorClass {
  UniformBuffer,
  StorageBuffer,
  TextureUnit,
};

inline DescriptorClass descriptor_class(VkDescriptorType type) {
  switch (type) {
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      return DescriptorClass::UniformBuffer;
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
    case VK_DESCRIPTOR_TYPE_SAMPLER:
      return DescriptorClass::TextureUnit;
    default:
      return DescriptorClass::StorageBuffer;
  }
}

// Descriptors of this type are read-only uniform data rather than read-write
// storage, or an image read through a texture unit.  Used to decide which
// namespace a binding belongs in.
inline bool descriptor_is_uniform(VkDescriptorType type) {
  return type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
         type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
}

// Adds a binding a shader reported to the list a pipeline will bind, merging it
// with an entry for the same descriptor from another stage.
//
// A sampled image carries the sampler it is read through.  One image read
// through two *different* samplers cannot be expressed -- GL gives a combined
// sampler one texture unit, and the unit holds one sampler object -- so that is
// refused rather than silently choosing one.  A texelFetch pairs the image with
// the injected dummy sampler, which is not a descriptor at all, so it merges
// into whichever real sampler the image is also read through; a fetch ignores
// sampler state, so that is exact rather than a compromise.
inline bool merge_used_binding(std::vector<GlBinding>& list,
                               const GlBinding& used, std::string* error) {
  for (GlBinding& existing : list) {
    if (existing.set != used.set || existing.binding != used.binding ||
        existing.type != used.type)
      continue;

    const bool existing_has_sampler = existing.sampler_set != kNoBinding;
    const bool used_has_sampler = used.sampler_set != kNoBinding;
    if (existing_has_sampler && used_has_sampler &&
        (existing.sampler_set != used.sampler_set ||
         existing.sampler_binding != used.sampler_binding)) {
      if (error != nullptr) {
        *error = "the sampled image at set " + std::to_string(used.set) +
                 " binding " + std::to_string(used.binding) +
                 " is read through two different samplers, at set " +
                 std::to_string(existing.sampler_set) + " binding " +
                 std::to_string(existing.sampler_binding) + " and set " +
                 std::to_string(used.sampler_set) + " binding " +
                 std::to_string(used.sampler_binding) +
                 "; OpenGL gives a combined sampler one texture unit, so Kniz "
                 "cannot express that";
      }
      return false;
    }
    if (!existing_has_sampler && used_has_sampler) {
      existing.sampler_set = used.sampler_set;
      existing.sampler_binding = used.sampler_binding;
    }
    return true;
  }
  list.push_back(used);
  return true;
}

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

  // Where every descriptor the layout declares lives in OpenGL, computed once
  // here because it is a property of the layout alone.
  //
  // GLSL has one flat binding namespace per resource class and no descriptor
  // sets, while Vulkan addresses a descriptor by (set, binding): two resources
  // in different sets may legally share a binding number.  Compacting rather
  // than computing set * width + binding is deliberate -- GL bounds these
  // namespaces by GL_MAX_*_BUFFER_BINDINGS, which can be as low as 8, so the
  // numbers must stay dense.
  std::vector<GlBinding> bindings;

  // How many of those are uniform buffers, counting the push constant block if
  // there is one.  Push constants have no descriptor of their own and are given
  // the *first* uniform binding point, before the descriptors, because the index
  // must be the same in every layout an application may legally share a push
  // with; the reasoning is in kniz_descriptor.cpp.  The replay path and the
  // generated GLSL read the same number.
  uint32_t uniform_count = 0;
  uint32_t push_constant_gl_index = 0;

  // How many texture units the layout's image and sampler descriptors occupy.
  // A shader that does texelFetch on a separate image needs a sampler argument
  // that Vulkan does not require, so SPIRV-Cross injects one; it is given the
  // first unit past every descriptor, which is why the count is here and not
  // just the mapping.
  uint32_t texture_count = 0;

  // That first unit past every descriptor, i.e. where the injected sampler
  // lives.  Meaningful only when texture_count is non-zero.
  uint32_t dummy_sampler_gl_index() const { return texture_count; }

  // The pipeline layout's push constant ranges, and the end of the last one.
  // GL has no push constants, so they are lowered onto a uniform buffer that
  // Kniz owns and uploads to before each draw or dispatch.
  std::vector<VkPushConstantRange> push_constant_ranges;
  uint32_t push_constant_size = 0;
  bool has_push_constants = false;
};

struct DescriptorBinding {
  bool bound = false;

  // A buffer descriptor's contents.  Null for an image or sampler descriptor,
  // which is why `bound` exists rather than a null check on one pointer.
  Buffer* buffer = nullptr;
  uint64_t offset = 0;
  uint64_t range = 0;

  // An image descriptor's contents.  A sampled image needs the view; a combined
  // image sampler needs the sampler as well; a bare sampler needs only that.
  ImageView* view = nullptr;
  Sampler* sampler = nullptr;
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

// Where one vertex attribute reads from, resolved from a Vulkan
// VkVertexInputAttributeDescription against the pipeline's format table.  GL
// takes the type and component count at the attribute pointer, so the
// translation happens once, at pipeline creation, rather than at every draw.
struct VertexAttribute {
  uint32_t location = 0;
  uint32_t binding = 0;
  gl::Enum type = 0;   // GL_FLOAT, GL_UNSIGNED_BYTE, ...
  gl::Int components = 0;
  bool normalized = false;
  uint32_t offset = 0;
};

struct VertexBinding {
  uint32_t binding = 0;
  uint32_t stride = 0;
  VkVertexInputRate input_rate = VK_VERTEX_INPUT_RATE_VERTEX;
};

struct BoundVertexBuffer {
  Buffer* buffer = nullptr;
  uint64_t offset = 0;
};

struct Pipeline {
  uint64_t magic = kPipelineMagic;
  Device* device = nullptr;
  VkPipelineBindPoint bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
  PipelineLayout* layout = nullptr;

  gl::Uint program = 0;

  // The descriptors this pipeline's shaders actually read, which is a subset of
  // what its layout declares: a layout shared between pipelines commonly
  // declares bindings that a particular pair of stages never touches, and
  // requiring those to be bound would refuse programs Vulkan accepts.  The
  // mapping itself lives on the layout, since it is a property of the layout;
  // this is only the part of it that is used.
  std::vector<GlBinding> used_bindings;

  // Kept for diagnostics: when a shader fails to compile, this is the GLSL that
  // GL rejected, and reading it is usually faster than reading the SPIR-V.
  std::string glsl;

  // ---- graphics pipelines (M2b) -------------------------------------------
  //
  // A graphics pipeline is a GL program plus a great deal of state that GL keeps
  // in no single object.  The state is stored here and applied at replay, which
  // is the only place it can be applied, since GL has no pipeline object.
  RenderPass* render_pass = nullptr;

  // The vertex array object the attributes are captured in.  One per pipeline,
  // re-pointed at each draw because the buffers a draw names are only known then.
  gl::Uint vao = 0;

  std::vector<VertexBinding> vertex_bindings;
  std::vector<VertexAttribute> vertex_attributes;

  gl::Enum draw_mode = 0;  // GL_TRIANGLES, GL_LINE_STRIP, ...

  bool dynamic_viewport = false;
  bool dynamic_scissor = false;
  VkViewport viewport{};
  VkRect2D scissor{};

  // GL_CULL_FACE is enabled when there is something to cull; kNone means the
  // pipeline asked for no culling and the capability is turned off.
  gl::Enum cull_face = gl::kNone;
  gl::Enum front_face = gl::kCcw;
  VkColorComponentFlags color_write = 0xF;
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

    // M2.  Image work is recorded with its own fields rather than reusing the
    // buffer ones: a clear has no buffer at all, and a copy carries a list of
    // regions.
    ClearColorImage,
    CopyBufferToImage,
    CopyImageToBuffer,

    // M2b.  A render pass instance brackets the draws that belong to it, and a
    // draw captures everything it needs -- resources and state alike -- because
    // Vulkan says the state a draw uses is the state bound when it is recorded.
    BeginRenderPass,
    EndRenderPass,
    Draw,

    // M2c.  Push constants are command buffer state, and GL has no equivalent,
    // so they are recorded in order and uploaded to a uniform buffer as they
    // replay.  Recording them as their own command rather than folding them
    // into the draw is what makes the ordering fall out: a draw sees the values
    // written before it, exactly as Vulkan says.
    PushConstants,
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

  // Image operations.
  Image* image = nullptr;

  // vkCmdClearColorImage's value, copied byte for byte out of the caller's
  // VkClearColorValue.  Kept raw because which member of that union is meaningful
  // depends on the image format, which is known at replay but not necessarily at
  // record time.
  uint8_t clear_value[16] = {};
  VkImageSubresourceRange clear_range{};

  // vkCmdCopyBufferToImage / vkCmdCopyImageToBuffer.  The buffer is `dst` for the
  // first kind and `src` for the second, following the direction of the copy.
  std::vector<VkBufferImageCopy> regions;

  // ---- M2b: rendering ------------------------------------------------------
  //
  // BeginRenderPass.  The clear values are copied out of the caller's array
  // because Vulkan lets it be freed as soon as the call returns.
  Framebuffer* framebuffer = nullptr;
  VkRect2D render_area{};
  std::vector<uint8_t> clear_values;

  // Draw.  `pipeline` and `sets` are reused from the dispatch case; the rest is
  // the state a draw resolves: the vertex buffers bound at the time, the index
  // buffer if any, and the counts.
  std::vector<BoundVertexBuffer> vertex_buffers;  // indexed by binding number
  Buffer* index_buffer = nullptr;
  uint64_t index_offset = 0;
  VkIndexType index_type = VK_INDEX_TYPE_UINT16;
  uint32_t first_vertex = 0;
  uint32_t vertex_count = 0;
  uint32_t first_index = 0;
  uint32_t index_count = 0;
  int32_t vertex_offset = 0;
  VkViewport viewport{};
  VkRect2D scissor{};
  bool indexed = false;

  // ---- M2c: push constants ------------------------------------------------
  //
  // PushConstants.  The layout names the uniform binding point the values
  // belong at, which is a property of the layout and therefore stable across
  // the pipeline binds that may follow it.
  PipelineLayout* push_layout = nullptr;
  uint32_t push_offset = 0;
  // `bytes` holds the payload; the rest of the block keeps whatever the
  // application last wrote, exactly as Vulkan specifies.
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

  // Recording-time binding state.  vkCmdDispatch and vkCmdDraw capture it, so
  // that binding and recording stay independent of replay order.
  //
  // Compute and graphics have separate binding points in Vulkan -- a pipeline and
  // its descriptor sets bound for one are not bound for the other -- so the two
  // are tracked separately rather than sharing one slot.
  struct BindState {
    Pipeline* pipeline = nullptr;
    PipelineLayout* layout = nullptr;
    std::vector<DescriptorSet*> sets;
  };

  BindState compute;
  BindState graphics;

  // Dynamic state, as recorded by vkCmdSetViewport / vkCmdSetScissor.  A pipeline
  // that declares a piece of dynamic state ignores its own static copy of it, so
  // the draw path has to know whether either was ever set.
  bool viewport_set = false;
  VkViewport viewport{};
  bool scissor_set = false;
  VkRect2D scissor{};

  // Vertex buffers, indexed by Vulkan binding number.  Resized on demand because
  // a binding index is whatever the pipeline's vertex input declares.
  std::vector<BoundVertexBuffer> vertex_buffers;
  Buffer* index_buffer = nullptr;
  uint64_t index_offset = 0;
  VkIndexType index_type = VK_INDEX_TYPE_UINT16;

  // The render pass instance the recording is inside, if any.  Vulkan requires a
  // draw to be recorded inside one, and this is what makes that checkable.
  //
  // The render area is kept alongside it because it is not merely a description
  // of what the load operation clears: Vulkan discards every fragment outside it,
  // so it acts as a second scissor that the pipeline's own scissor state is
  // intersected with.
  Framebuffer* active_framebuffer = nullptr;
  VkRect2D active_render_area{};
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
// M2 objects
// ---------------------------------------------------------------------------

// A VkImage is a GL texture.  Unlike a buffer there is deliberately no host
// mirror: GL owns the texture's storage and its internal layout corresponds to
// no layout Vulkan describes, so every read and write goes through a command.
//
// Two consequences, both enforced rather than documented and forgotten:
// vkMapMemory refuses an allocation an image is bound to (see DeviceMemory), and
// linear tiling is refused at vkCreateImage, because a linear image's layout is
// something an application is entitled to observe and Kniz cannot provide it.
struct Image {
  uint64_t magic = kImageMagic;
  Device* device = nullptr;

  VkImageType type = VK_IMAGE_TYPE_2D;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkExtent3D extent{};
  uint32_t mip_levels = 1;
  uint32_t array_layers = 1;
  VkImageUsageFlags usage = 0;

  // Tracked for diagnostics only.  OpenGL has no image layouts, so a barrier
  // that transitions one is recorded and replayed as a memory barrier.
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;

  gl::Uint gl_texture = 0;

  DeviceMemory* memory = nullptr;
  uint64_t memory_offset = 0;

  // The GL texture's storage is created in vkBindImageMemory, which is the
  // point where Vulkan makes the format and extent immutable.
  bool storage_allocated = false;
};

// A view carries the format and subresource range a use of the image sees.  The
// GL side is the image's texture; a view exists so that format mismatches are
// caught where Vulkan says they are, and so that render passes and samplers have
// somewhere to record which range they mean.
struct ImageView {
  uint64_t magic = kImageViewMagic;
  Device* device = nullptr;
  Image* image = nullptr;

  VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkImageSubresourceRange range{};
};

// A Vulkan sampler is filter, wrap, LOD and border state with no image in it.
// That is exactly what a GL sampler object is, so the translation is direct --
// and it has to be an object rather than texture state, because two samplers may
// read one image with different settings and GL keeps the two kinds of state
// separate for precisely that reason.
struct Sampler {
  uint64_t magic = kSamplerMagic;
  Device* device = nullptr;

  gl::Uint gl_sampler = 0;
};

// ---------------------------------------------------------------------------
// M2b objects: drawing
// ---------------------------------------------------------------------------

// A render pass describes a set of attachments, how they are loaded and stored,
// and which of them a subpass writes.  GL has no render pass; the part of this
// that has a GL representation is the load operation, which becomes a clear of
// the corresponding framebuffer attachment, and the attachment list itself, which
// decides which GL attachments the framebuffer has.
//
// M2b accepts exactly one subpass.  Subpasses are the one construct GL cannot
// express at all -- they exist to keep data on-chip between passes, and GL would
// have to round-trip every one of them through a texture -- so they are refused
// rather than approximated.  See the fidelity budget in docs/DESIGN.md.
struct RenderPass {
  uint64_t magic = kRenderPassMagic;
  Device* device = nullptr;

  std::vector<VkAttachmentDescription> attachments;

  // The subpass's colour attachments, as attachment indices.  Kept ordered so
  // that subpass attachment i is GL_COLOR_ATTACHMENTi, which is the mapping
  // glDrawBuffers and glClearBufferfv both assume.
  std::vector<uint32_t> color_attachments;
};

// A framebuffer binds specific image views to a render pass's attachments.  On
// the GL side it is a framebuffer object, created once, whose attachments are the
// images' own textures.
struct Framebuffer {
  uint64_t magic = kFramebufferMagic;
  Device* device = nullptr;
  RenderPass* render_pass = nullptr;

  std::vector<ImageView*> attachments;

  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t layers = 0;

  gl::Uint gl_framebuffer = 0;
};

// ---------------------------------------------------------------------------
// M3 objects: presentation
// ---------------------------------------------------------------------------

// A surface is a window, not a drawing target.  It is created on the instance
// and outlives no device; the swapchain built on it is what owns GL objects.
//
// The only thing Kniz needs from the window is its native handle, because the
// drawable is an EGL window surface and EGL wants exactly that handle.  Nothing
// is ever asked of the window system directly -- no geometry query, no event
// loop -- which is why this file needs no platform headers and why a surface
// can be created from a window the application has not mapped yet.
struct Surface {
  uint64_t magic = kSurfaceMagic;
  Instance* instance = nullptr;

  // An X11 XID, kept as an integer rather than as an xcb_window_t so that no
  // platform header has to be included here.  Both VK_KHR_xcb_surface and
  // VK_KHR_xlib_surface hand us this same number.
  uint64_t native_window = 0;

  // The EGL drawable for that window, created the first time a swapchain needs
  // it and destroyed with the surface.
  //
  // It belongs here rather than to the swapchain because it is a property of the
  // window, not of the images: an application that resizes its window destroys
  // its swapchain and creates another on the same surface, and creating a second
  // EGL surface for one window is neither necessary nor always possible.  A
  // swapchain that replaced an old one therefore shares this drawable, which is
  // exactly what "the window did not change" means.
  gl::WindowSurface window = nullptr;
};

// A swapchain is a list of images, shown in the surface's window.
//
// The images are ordinary GL textures -- the same object M2a made for VkImage --
// so a render pass that draws into a swapchain image needs nothing new.  What is
// new is presentation: GL cannot make a texture *be* a window's front buffer, so
// the last step is explicit -- a draw from the image into the window, followed by
// eglSwapBuffers (see kniz_swapchain.cpp and DESIGN.md §3.13).
//
// The window itself is not here: it belongs to the Surface, because it is a
// property of the window rather than of the images, and an application that
// resizes destroys its swapchain and makes another on the same surface.
//
// The images are owned here rather than by the application.  vkGetSwapchainImagesKHR
// hands out handles that are valid until the swapchain is destroyed, and
// vkDestroyImage on one of them is invalid usage, so they are created and
// destroyed with the swapchain and never appear in any device list.
struct Swapchain {
  uint64_t magic = kSwapchainMagic;
  Device* device = nullptr;
  Surface* surface = nullptr;

  VkFormat format = VK_FORMAT_UNDEFINED;
  VkExtent2D extent{};
  VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;

  // Presentation is a draw: one triangle covering the window, sampling the
  // image through a sampler object, compiled once when the swapchain is made.
  // The alternative -- glBlitFramebuffer -- only works when the image and the
  // window happen to have the same format, which is not a constraint Vulkan
  // imposes and which a window manager is free to break.
  gl::Uint present_program = 0;
  gl::Uint present_array = 0;
  gl::Uint present_sampler = 0;

  std::vector<Image*> images;

  // Acquisition is round robin.  Kniz presents synchronously, so no image is
  // ever in flight: by the time vkQueuePresentKHR returns the frame has been
  // drawn and handed to the compositor, and the next acquire can hand out the
  // next image unconditionally.
  uint32_t next_image = 0;
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

inline Image* from_handle(VkImage handle) {
  return detail::to_object<Image>(kImageMagic, handle);
}

inline ImageView* from_handle(VkImageView handle) {
  return detail::to_object<ImageView>(kImageViewMagic, handle);
}

inline Sampler* from_handle(VkSampler handle) {
  return detail::to_object<Sampler>(kSamplerMagic, handle);
}

inline RenderPass* from_handle(VkRenderPass handle) {
  return detail::to_object<RenderPass>(kRenderPassMagic, handle);
}

inline Framebuffer* from_handle(VkFramebuffer handle) {
  return detail::to_object<Framebuffer>(kFramebufferMagic, handle);
}

inline Surface* from_handle(VkSurfaceKHR handle) {
  return detail::to_object<Surface>(kSurfaceMagic, handle);
}

inline Swapchain* from_handle(VkSwapchainKHR handle) {
  return detail::to_object<Swapchain>(kSwapchainMagic, handle);
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

inline VkImage to_handle(Image* object) {
  return reinterpret_cast<VkImage>(object);
}

inline VkImageView to_handle(ImageView* object) {
  return reinterpret_cast<VkImageView>(object);
}

inline VkSampler to_handle(Sampler* object) {
  return reinterpret_cast<VkSampler>(object);
}

inline VkRenderPass to_handle(RenderPass* object) {
  return reinterpret_cast<VkRenderPass>(object);
}

inline VkFramebuffer to_handle(Framebuffer* object) {
  return reinterpret_cast<VkFramebuffer>(object);
}

inline VkSurfaceKHR to_handle(Surface* object) {
  return reinterpret_cast<VkSurfaceKHR>(object);
}

inline VkSwapchainKHR to_handle(Swapchain* object) {
  return reinterpret_cast<VkSwapchainKHR>(object);
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
