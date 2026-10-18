// Buffers, device memory, descriptors.
//
// The shape of this file follows from one fact: OpenGL has nothing that behaves
// like VkDeviceMemory.  There is no allocator to call, no opaque allocation
// with properties, and no way for the driver to hand out a pointer that the GPU
// and the CPU both address.  Kniz therefore *is* the allocator: an allocation is
// a host block it owns, and buffers are copied between that block and a GL
// buffer object around each submission.
//
// With the single host-visible memory type this driver reports, that is
// correct.  It is also slow -- a submission that touches N buffers copies all N
// in and all N out -- and it is the honest cost of the emulation rather than a
// tuning problem.  docs/DESIGN.md records the day this will hurt.

#include "kniz_common.h"
#include "kniz_memory.h"
#include "kniz_objects.h"

#include <cstdlib>
#include <new>

namespace kniz {
namespace {

// Allocations are aligned to the offset alignment the driver reports, so that a
// host pointer plus any legal buffer offset is still suitably aligned.
constexpr uint64_t kAllocationAlignment = 256;

// Does the GPU ever write this buffer?  Only those need reading back.  A buffer
// that is merely a transfer source or a vertex input has nothing new to learn
// from the GPU after a submission, and copying it back would be pure waste.
bool gpu_writable(VkBufferUsageFlags usage) {
  return (usage & (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT)) != 0;
}

// Clamp a byte count to what a GLsizeiptr can hold.  A buffer this large is not
// reachable in practice (the heap is 1 GiB), but the conversion has to be
// defined rather than merely unreachable.
gl::SizePtr as_size_ptr(uint64_t bytes) {
  constexpr uint64_t kMax = static_cast<uint64_t>(PTRDIFF_MAX);
  return static_cast<gl::SizePtr>(bytes > kMax ? kMax : bytes);
}

bool device_can_run_gl(Device* device) {
  if (device == nullptr)
    return false;
  if (!device->instance->gl_ready) {
    KNIZ_ERR("no OpenGL context is available");
    return false;
  }
  return true;
}

Buffer* find_buffer(Device* device, VkBuffer handle) {
  Buffer* buffer = from_handle(handle);
  if (buffer == nullptr)
    KNIZ_ERR("%p is not a Kniz buffer", static_cast<void*>(handle));
  else if (buffer->device != device)
    KNIZ_ERR("buffer %p belongs to a different device", static_cast<void*>(handle));
  else
    return buffer;
  return nullptr;
}

void forget_buffer(Device* device, Buffer* buffer) {
  for (auto it = device->buffers.begin(); it != device->buffers.end(); ++it) {
    if (*it == buffer) {
      device->buffers.erase(it);
      return;
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateBuffer(
    VkDevice deviceHandle, const VkBufferCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkBuffer* pBuffer) {
  (void)pAllocator;
  if (pBuffer == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pBuffer = VK_NULL_HANDLE;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr || !device_can_run_gl(device))
    return VK_ERROR_INITIALIZATION_FAILED;
  if (pCreateInfo == nullptr || pCreateInfo->size == 0)
    return VK_ERROR_INITIALIZATION_FAILED;

  if ((pCreateInfo->flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT) != 0) {
    KNIZ_ERR("sparse buffers are not implemented");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  Buffer* buffer = new (std::nothrow) Buffer();
  if (buffer == nullptr)
    return VK_ERROR_OUT_OF_HOST_MEMORY;

  buffer->device = device;
  buffer->size = pCreateInfo->size;
  buffer->usage = pCreateInfo->usage;

  gl::f.GenBuffers(1, &buffer->gl_name);
  if (buffer->gl_name == 0) {
    KNIZ_ERR("glGenBuffers failed");
    delete buffer;
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  }

  device->buffers.push_back(buffer);
  *pBuffer = to_handle(buffer);
  KNIZ_TRACE("buffer %p: %llu bytes, usage 0x%x, GL name %u",
             static_cast<void*>(*pBuffer),
             static_cast<unsigned long long>(buffer->size), buffer->usage,
             buffer->gl_name);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyBuffer(VkDevice deviceHandle, VkBuffer bufferHandle,
                                         const VkAllocationCallbacks* pAllocator) {
  (void)pAllocator;
  Device* device = from_handle(deviceHandle);
  Buffer* buffer = from_handle(bufferHandle);
  if (device == nullptr || buffer == nullptr)
    return;

  if (buffer->gl_name != 0)
    gl::f.DeleteBuffers(1, &buffer->gl_name);

  forget_buffer(device, buffer);
  delete buffer;
}

VKAPI_ATTR void VKAPI_CALL GetBufferMemoryRequirements(
    VkDevice deviceHandle, VkBuffer bufferHandle,
    VkMemoryRequirements* pMemoryRequirements) {
  if (pMemoryRequirements == nullptr)
    return;
  std::memset(pMemoryRequirements, 0, sizeof(*pMemoryRequirements));

  Device* device = from_handle(deviceHandle);
  Buffer* buffer = find_buffer(device, bufferHandle);
  if (device == nullptr || buffer == nullptr)
    return;

  pMemoryRequirements->size = buffer->size;
  // Must match what vkBindBufferMemory is allowed to be handed, and it must not
  // be looser than what GL accepts for an indexed buffer range.  256 is both
  // Vulkan's largest legal value for this limit and a multiple of GL's.
  pMemoryRequirements->alignment = kAllocationAlignment;
  // Both reported memory types are acceptable: the emulation backs the
  // allocation with host memory either way, and an application that chooses the
  // non-mappable type simply uses a staging copy, which works.
  pMemoryRequirements->memoryTypeBits = 0x3;
}

// ---------------------------------------------------------------------------
// Device memory
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL AllocateMemory(
    VkDevice deviceHandle, const VkMemoryAllocateInfo* pAllocateInfo,
    const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory) {
  (void)pAllocator;
  if (pMemory == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pMemory = VK_NULL_HANDLE;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (pAllocateInfo == nullptr || pAllocateInfo->allocationSize == 0)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (pAllocateInfo->memoryTypeIndex > 1) {
    KNIZ_ERR("memory type %u does not exist; this driver reports 2",
             pAllocateInfo->memoryTypeIndex);
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  }

  DeviceMemory* memory = new (std::nothrow) DeviceMemory();
  if (memory == nullptr)
    return VK_ERROR_OUT_OF_HOST_MEMORY;

  memory->device = device;
  memory->size = pAllocateInfo->allocationSize;
  memory->type_index = pAllocateInfo->memoryTypeIndex;
  memory->properties =
      memory->type_index == 1
          ? (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
          : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  try {
    memory->host = static_cast<uint8_t*>(
        ::operator new(static_cast<size_t>(memory->size),
                       std::align_val_t(kAllocationAlignment)));
  } catch (const std::bad_alloc&) {
    delete memory;
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  }
  std::memset(memory->host, 0, static_cast<size_t>(memory->size));

  *pMemory = to_handle(memory);
  KNIZ_DBG("memory %p: %llu bytes, type %u", static_cast<void*>(*pMemory),
           static_cast<unsigned long long>(memory->size), memory->type_index);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FreeMemory(VkDevice deviceHandle, VkDeviceMemory memoryHandle,
                                      const VkAllocationCallbacks* pAllocator) {
  (void)pAllocator;
  Device* device = from_handle(deviceHandle);
  DeviceMemory* memory = from_handle(memoryHandle);
  if (device == nullptr || memory == nullptr)
    return;

  // Any buffer still pointing here would dangle.  Vulkan forbids that, so
  // rather than crash later, drop the associations now.
  for (Buffer* buffer : device->buffers) {
    if (buffer->memory == memory) {
      KNIZ_WARN("buffer %p is still bound to memory being freed", static_cast<void*>(buffer));
      buffer->memory = nullptr;
      buffer->memory_offset = 0;
    }
  }

  if (memory->host != nullptr)
    ::operator delete(memory->host, std::align_val_t(kAllocationAlignment));
  delete memory;
}

VKAPI_ATTR VkResult VKAPI_CALL BindBufferMemory(VkDevice deviceHandle,
                                                VkBuffer bufferHandle,
                                                VkDeviceMemory memoryHandle,
                                                VkDeviceSize memoryOffset) {
  Device* device = from_handle(deviceHandle);
  Buffer* buffer = find_buffer(device, bufferHandle);
  if (device == nullptr || buffer == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  DeviceMemory* memory = nullptr;
  if (memoryHandle != VK_NULL_HANDLE) {
    memory = from_handle(memoryHandle);
    if (memory == nullptr) {
      KNIZ_ERR("%p is not a Kniz memory allocation",
               static_cast<void*>(memoryHandle));
      return VK_ERROR_INITIALIZATION_FAILED;
    }
  }

  if (memory == nullptr &&
      (buffer->usage & VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT) == 0) {
    // Vulkan 1.1 added VK_MEMORY_UNBOUND semantics but this driver reports 1.0,
    // where a buffer must be bound before use.
    KNIZ_ERR("binding a buffer to VK_NULL_HANDLE requires Vulkan 1.1");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  if (memory != nullptr) {
    if (memoryOffset + buffer->size > memory->size) {
      KNIZ_ERR("buffer of %llu bytes at offset %llu does not fit in a %llu byte "
               "allocation",
               static_cast<unsigned long long>(buffer->size),
               static_cast<unsigned long long>(memoryOffset),
               static_cast<unsigned long long>(memory->size));
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
  }

  buffer->memory = memory;
  buffer->memory_offset = memoryOffset;
  buffer->uploaded_version = 0;

  // Give the GL buffer object its storage now.  Contents are whatever the host
  // block holds, which is uploaded on the next submission; emptying it here
  // keeps a stale previous binding from being read by mistake.
  if (buffer->gl_name != 0) {
    gl::f.BindBuffer(gl::kArrayBuffer, buffer->gl_name);
    gl::f.BufferData(gl::kArrayBuffer, as_size_ptr(buffer->size), nullptr,
                     gl::kDynamicDraw);
    gl::f.BindBuffer(gl::kArrayBuffer, 0);
  }

  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL MapMemory(VkDevice deviceHandle,
                                         VkDeviceMemory memoryHandle,
                                         VkDeviceSize offset, VkDeviceSize size,
                                         VkMemoryMapFlags flags, void** ppData) {
  (void)flags;
  if (ppData == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *ppData = nullptr;

  Device* device = from_handle(deviceHandle);
  DeviceMemory* memory = from_handle(memoryHandle);
  if (device == nullptr || memory == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  if ((memory->properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
    KNIZ_ERR("memory type %u is not host visible and cannot be mapped; use "
             "memory type 1",
             memory->type_index);
    return VK_ERROR_MEMORY_MAP_FAILED;
  }
  if (offset >= memory->size) {
    KNIZ_ERR("mapping offset %llu is past the end of a %llu byte allocation",
             static_cast<unsigned long long>(offset),
             static_cast<unsigned long long>(memory->size));
    return VK_ERROR_MEMORY_MAP_FAILED;
  }
  if (size != VK_WHOLE_SIZE && offset + size > memory->size) {
    KNIZ_ERR("mapping range [%llu, %llu) does not fit in a %llu byte allocation",
             static_cast<unsigned long long>(offset),
             static_cast<unsigned long long>(offset + size),
             static_cast<unsigned long long>(memory->size));
    return VK_ERROR_MEMORY_MAP_FAILED;
  }

  // Unlike Vulkan, a second map of an already-mapped range is accepted rather
  // than rejected: unmap releases every mapping of the allocation, so the
  // reference count is all that has to be right.
  ++memory->map_count;
  *ppData = memory->host + offset;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL UnmapMemory(VkDevice deviceHandle,
                                       VkDeviceMemory memoryHandle) {
  Device* device = from_handle(deviceHandle);
  DeviceMemory* memory = from_handle(memoryHandle);
  if (device == nullptr || memory == nullptr)
    return;

  memory->map_count = 0;

  // The application may have written anything through the pointer it held, so
  // every buffer sharing this allocation has to be re-uploaded before it is
  // read again.  Bumping the version is what makes that happen; buffers compare
  // against it at submission time.
  ++memory->version;
}

// Both of these are no-ops, and that is correct rather than lazy: the only
// mappable memory type this driver reports includes
// VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, so there is nothing to synchronise.  If
// a non-coherent type is ever added, this becomes the place it is handled.
VKAPI_ATTR VkResult VKAPI_CALL FlushMappedMemoryRanges(
    VkDevice deviceHandle, uint32_t memRangeCount,
    const VkMappedMemoryRange* pMemRanges) {
  (void)deviceHandle;
  (void)memRangeCount;
  (void)pMemRanges;
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL InvalidateMappedMemoryRanges(
    VkDevice deviceHandle, uint32_t memRangeCount,
    const VkMappedMemoryRange* pMemRanges) {
  (void)deviceHandle;
  (void)memRangeCount;
  (void)pMemRanges;
  return VK_SUCCESS;
}

// ---------------------------------------------------------------------------
// Submission-time data movement
// ---------------------------------------------------------------------------

void sync_buffers_to_gl(Device* device) {
  for (Buffer* buffer : device->buffers) {
    if (buffer->memory == nullptr || buffer->gl_name == 0)
      continue;
    if (buffer->uploaded_version == buffer->memory->version)
      continue;

    gl::f.BindBuffer(gl::kArrayBuffer, buffer->gl_name);
    gl::f.BufferSubData(gl::kArrayBuffer, 0, as_size_ptr(buffer->size),
                        buffer->memory->host + buffer->memory_offset);
    gl::f.BindBuffer(gl::kArrayBuffer, 0);
    buffer->uploaded_version = buffer->memory->version;
  }
}

void sync_buffers_from_gl(Device* device) {
  for (Buffer* buffer : device->buffers) {
    if (buffer->memory == nullptr || buffer->gl_name == 0)
      continue;
    if (!gpu_writable(buffer->usage))
      continue;

    gl::f.BindBuffer(gl::kArrayBuffer, buffer->gl_name);
    gl::f.GetBufferSubData(gl::kArrayBuffer, 0, as_size_ptr(buffer->size),
                           buffer->memory->host + buffer->memory_offset);
    gl::f.BindBuffer(gl::kArrayBuffer, 0);
  }
}

}  // namespace kniz
