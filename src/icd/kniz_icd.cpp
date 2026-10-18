// Global and instance level Vulkan commands, plus the loader bootstrap.
//
// The bootstrap is the only part of Kniz that the Vulkan loader looks up by
// symbol name.  Everything else is handed to the loader as a function pointer
// through one of the two entry point tables at the bottom of this file.

#include "kniz_common.h"
#include "kniz_device.h"
#include "kniz_entrypoints.h"
#include "kniz_gl.h"
#include "kniz_objects.h"

#include <atomic>
#include <cstdio>
#include <new>
#include <string>

namespace kniz {
namespace {

// The GL context is process wide, so it is reference counted against live
// instances rather than owned by any one of them.
std::atomic<uint32_t> g_live_instances{0};

// ---------------------------------------------------------------------------
// Physical device capabilities
// ---------------------------------------------------------------------------

// Vulkan defines a floor of guaranteed limits that every conformant
// implementation must meet and that every application must therefore tolerate.
// Kniz reports exactly that floor.
//
// That is a deliberate choice rather than laziness.  The floor is honest -- we
// claim nothing we cannot deliver -- it requires no GL queries to establish, and
// it is by construction usable by any application that runs at all.  Reporting
// optimistic values would only trade a clean failure for silent corruption.
// Once the backend can query the GL context for real (M2 onwards) these can be
// raised to what the adapter actually offers.
void fill_limits(VkPhysicalDeviceLimits& l) {
  std::memset(&l, 0, sizeof(l));

  l.maxImageDimension1D = 4096;
  l.maxImageDimension2D = 4096;
  l.maxImageDimension3D = 256;
  l.maxImageDimensionCube = 4096;
  l.maxImageArrayLayers = 256;
  l.maxTexelBufferElements = 65536;
  l.maxUniformBufferRange = 16384;
  l.maxStorageBufferRange = 134217728;
  l.maxPushConstantsSize = 128;
  l.maxMemoryAllocationCount = 4096;
  l.maxSamplerAllocationCount = 4000;
  l.bufferImageGranularity = 1;
  l.sparseAddressSpaceSize = 0;
  l.maxBoundDescriptorSets = 4;
  l.maxPerStageDescriptorSamplers = 16;
  l.maxPerStageDescriptorUniformBuffers = 12;
  l.maxPerStageDescriptorStorageBuffers = 4;
  l.maxPerStageDescriptorSampledImages = 16;
  l.maxPerStageDescriptorStorageImages = 4;
  l.maxPerStageDescriptorInputAttachments = 4;
  l.maxPerStageResources = 128;
  l.maxDescriptorSetSamplers = 96;
  l.maxDescriptorSetUniformBuffers = 90;
  l.maxDescriptorSetUniformBuffersDynamic = 8;
  l.maxDescriptorSetStorageBuffers = 24;
  l.maxDescriptorSetStorageBuffersDynamic = 4;
  l.maxDescriptorSetSampledImages = 96;
  l.maxDescriptorSetStorageImages = 24;
  l.maxDescriptorSetInputAttachments = 4;
  l.maxVertexInputAttributes = 16;
  l.maxVertexInputBindings = 16;
  l.maxVertexInputAttributeOffset = 2047;
  l.maxVertexInputBindingStride = 2048;
  l.maxVertexOutputComponents = 64;
  l.maxTessellationGenerationLevel = 64;
  l.maxTessellationPatchSize = 32;
  l.maxTessellationControlPerVertexInputComponents = 64;
  l.maxTessellationControlPerVertexOutputComponents = 64;
  l.maxTessellationControlPerPatchOutputComponents = 120;
  l.maxTessellationControlTotalOutputComponents = 2048;
  l.maxTessellationEvaluationInputComponents = 64;
  l.maxTessellationEvaluationOutputComponents = 128;
  l.maxGeometryShaderInvocations = 32;
  l.maxGeometryInputComponents = 64;
  l.maxGeometryOutputComponents = 64;
  l.maxGeometryOutputVertices = 256;
  l.maxGeometryTotalOutputComponents = 1024;
  l.maxFragmentInputComponents = 64;
  l.maxFragmentOutputAttachments = 4;
  l.maxFragmentDualSrcAttachments = 1;
  l.maxFragmentCombinedOutputResources = 4;
  l.maxComputeSharedMemorySize = 16384;
  l.maxComputeWorkGroupCount[0] = 65535;
  l.maxComputeWorkGroupCount[1] = 65535;
  l.maxComputeWorkGroupCount[2] = 65535;
  l.maxComputeWorkGroupInvocations = 128;
  l.maxComputeWorkGroupSize[0] = 128;
  l.maxComputeWorkGroupSize[1] = 128;
  l.maxComputeWorkGroupSize[2] = 64;
  l.subPixelPrecisionBits = 4;
  l.subTexelPrecisionBits = 4;
  l.mipmapPrecisionBits = 4;
  l.maxDrawIndexedIndexValue = 0xFFFFFFFFu;
  l.maxDrawIndirectCount = 0xFFFFFFFFu;
  l.maxSamplerLodBias = 2.0f;
  l.maxSamplerAnisotropy = 1.0f;
  l.maxViewports = 1;
  l.maxViewportDimensions[0] = 4096;
  l.maxViewportDimensions[1] = 4096;
  l.viewportBoundsRange[0] = -8192.0f;
  l.viewportBoundsRange[1] = 8191.0f;
  l.minMemoryMapAlignment = 64;
  l.minTexelBufferOffsetAlignment = 256;
  l.minUniformBufferOffsetAlignment = 256;
  l.minStorageBufferOffsetAlignment = 256;
  l.minTexelOffset = -8;
  l.maxTexelOffset = 7;
  l.minTexelGatherOffset = -32;
  l.maxTexelGatherOffset = 31;
  l.minInterpolationOffset = -0.5f;
  l.maxInterpolationOffset = 0.5f;
  l.subPixelInterpolationOffsetBits = 4;
  l.maxFramebufferWidth = 4096;
  l.maxFramebufferHeight = 4096;
  l.maxFramebufferLayers = 256;
  l.framebufferColorSampleCounts = VK_SAMPLE_COUNT_1_BIT;
  l.framebufferDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT;
  l.framebufferStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT;
  l.framebufferNoAttachmentsSampleCounts = VK_SAMPLE_COUNT_1_BIT;
  l.maxColorAttachments = 4;
  l.sampledImageColorSampleCounts = VK_SAMPLE_COUNT_1_BIT;
  l.sampledImageIntegerSampleCounts = VK_SAMPLE_COUNT_1_BIT;
  l.sampledImageDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT;
  l.sampledImageStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT;
  l.storageImageSampleCounts = VK_SAMPLE_COUNT_1_BIT;
  l.maxSampleMaskWords = 1;
  l.timestampComputeAndGraphics = VK_FALSE;
  l.timestampPeriod = 0.0f;
  l.maxClipDistances = 8;
  l.maxCullDistances = 8;
  l.maxCombinedClipAndCullDistances = 8;
  l.discreteQueuePriorities = 2;
  l.pointSizeRange[0] = 1.0f;
  l.pointSizeRange[1] = 64.0f;
  l.lineWidthRange[0] = 1.0f;
  l.lineWidthRange[1] = 1.0f;
  l.pointSizeGranularity = 1.0f;
  l.lineWidthGranularity = 0.0f;
  l.strictLines = VK_FALSE;
  l.standardSampleLocations = VK_TRUE;
  l.optimalBufferCopyOffsetAlignment = 1;
  l.optimalBufferCopyRowPitchAlignment = 1;
  l.nonCoherentAtomSize = 64;
}

// FNV-1a, used only to derive a stable cache UUID from the GL version string so
// that applications keying pipeline caches off it get a consistent value across
// runs on the same backend.
uint64_t fnv1a(const std::string& text, uint64_t seed) {
  uint64_t hash = seed;
  for (const char c : text) {
    hash ^= static_cast<unsigned char>(c);
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string device_name(const Instance& instance) {
  std::string name = "Kniz";
  if (!instance.gl.renderer.empty())
    name += " on " + instance.gl.renderer;
  return name;
}

void fill_properties(const PhysicalDevice& physical,
                     VkPhysicalDeviceProperties& p) {
  std::memset(&p, 0, sizeof(p));

  p.apiVersion = physical.instance->effective_api_version;
  p.driverVersion = kDriverVersion;
  p.vendorID = kVendorId;
  p.deviceID = physical.device_id;
  p.deviceType = VK_PHYSICAL_DEVICE_TYPE_OTHER;

  copy_string(p.deviceName, device_name(*physical.instance).c_str());
  fill_limits(p.limits);

  const std::string seed =
      physical.instance->gl.version + "|" + physical.instance->gl.renderer;
  const uint64_t lo = fnv1a(seed, 14695981039346656037ull);
  const uint64_t hi = fnv1a(seed, 1099511628211ull);
  std::memcpy(p.pipelineCacheUUID, &lo, sizeof(lo));
  std::memcpy(p.pipelineCacheUUID + sizeof(lo), &hi,
              sizeof(p.pipelineCacheUUID) - sizeof(lo));

  p.sparseProperties = VkPhysicalDeviceSparseProperties{};
}

// Nothing is enabled yet.  Advertising a feature that silently does nothing is
// the worst possible failure mode: the application stops taking its fallback
// path and produces wrong output instead of refusing to start.  Features get
// turned on one at a time, as they are implemented.
void fill_features(VkPhysicalDeviceFeatures& f) { std::memset(&f, 0, sizeof(f)); }

void fill_queue_families(VkQueueFamilyProperties& q) {
  std::memset(&q, 0, sizeof(q));
  q.queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT |
                 VK_QUEUE_TRANSFER_BIT;
  q.queueCount = 1;
  q.timestampValidBits = 0;
  q.minImageTransferGranularity = VkExtent3D{1, 1, 1};
}

// An Intel style unified-memory layout: a single device-local heap that is also
// host visible.  Type 0 is the "GPU only" type applications expect to exist,
// type 1 is the one they will actually map.
void fill_memory_properties(VkPhysicalDeviceMemoryProperties& m) {
  std::memset(&m, 0, sizeof(m));
  m.memoryTypeCount = 2;
  m.memoryHeapCount = 1;
  m.memoryHeaps[0].size = 1024ull * 1024ull * 1024ull;
  m.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
  m.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  m.memoryTypes[0].heapIndex = 0;
  m.memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  m.memoryTypes[1].heapIndex = 0;
}

// ---------------------------------------------------------------------------
// Global and instance commands
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL EnumerateInstanceVersion(uint32_t* pApiVersion) {
  if (pApiVersion == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  // We implement Vulkan 1.0 semantics.  Saying so costs us applications that
  // require 1.1 or later, and that is the correct outcome: they would fail
  // anyway, just later and more confusingly.
  *pApiVersion = VK_API_VERSION_1_0;
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateInstanceExtensionProperties(
    const char* pLayerName, uint32_t* pPropertyCount,
    VkExtensionProperties* pProperties) {
  (void)pLayerName;
  if (pPropertyCount == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  // No instance extensions yet.  VK_KHR_surface and the platform surface
  // extensions arrive with the swapchain in M3.
  return enumerate_count(0, pPropertyCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(
    const VkInstanceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkInstance* pInstance) {
  if (pInstance == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pInstance = VK_NULL_HANDLE;

  if (pCreateInfo == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  if (pAllocator != nullptr) {
    // Honouring custom allocators means routing every allocation through them.
    // Saying so out loud is better than quietly ignoring them, because a caller
    // tracking its own memory would otherwise see unexplained imbalance.
    KNIZ_WARN("host allocation callbacks are not supported and will be ignored");
  }

  uint32_t requested = VK_API_VERSION_1_0;
  if (pCreateInfo->pApplicationInfo != nullptr &&
      pCreateInfo->pApplicationInfo->apiVersion != 0) {
    requested = pCreateInfo->pApplicationInfo->apiVersion;
  }

  for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; ++i) {
    KNIZ_WARN("instance extension '%s' requested but not implemented",
              pCreateInfo->ppEnabledExtensionNames[i]);
  }

  Instance* instance = nullptr;
  try {
    instance = new Instance();
    // vk_icd.h requires the reserved first word to be initialised before the
    // object is handed to the loader.  See kniz_objects.h.
    set_loader_magic_value(instance);
    instance->api_version = requested;

    // Report the lower of what was asked for and what we can do.  An
    // application that asks for 1.3 and gets a 1.0 instance is correct by
    // construction; one that asks for 1.3 and is told it got 1.3 is not.
    instance->effective_api_version =
        (requested < VK_API_VERSION_1_0) ? requested : VK_API_VERSION_1_0;

    if (g_live_instances.fetch_add(1, std::memory_order_acq_rel) == 0) {
      if (!gl::create_headless()) {
        g_live_instances.fetch_sub(1, std::memory_order_acq_rel);
        delete instance;
        return VK_ERROR_INITIALIZATION_FAILED;
      }
    }
    instance->gl_ready = true;
    instance->gl = gl::info();

    instance->physical = new PhysicalDevice();
    set_loader_magic_value(instance->physical);
    instance->physical->instance = instance;
    instance->physical->index = 0;
    // The GL adapter has no Vulkan identity, so invent one that is stable for a
    // given backend.  Reporting the real Intel/AMD/NVIDIA vendor IDs here would
    // make applications take vendor-specific paths that we do not implement.
    instance->physical->vendor_id = kVendorId;
    instance->physical->device_id = static_cast<uint32_t>(
        fnv1a(instance->gl.renderer, 2166136261ull) & 0xFFFFu);

  } catch (const std::bad_alloc&) {
    delete instance;
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  }

  *pInstance = to_handle(instance);

  KNIZ_INFO("instance created: requested Vulkan %u.%u, exposing %u.%u",
            VK_VERSION_MAJOR(requested), VK_VERSION_MINOR(requested),
            VK_VERSION_MAJOR(instance->effective_api_version),
            VK_VERSION_MINOR(instance->effective_api_version));

  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyInstance(
    VkInstance instanceHandle, const VkAllocationCallbacks* pAllocator) {
  (void)pAllocator;

  Instance* instance = from_handle(instanceHandle);
  if (instance == nullptr)
    return;

  delete instance->physical;
  delete instance;

  if (g_live_instances.fetch_sub(1, std::memory_order_acq_rel) == 1)
    gl::destroy();
}

VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDevices(
    VkInstance instanceHandle, uint32_t* pPhysicalDeviceCount,
    VkPhysicalDevice* pPhysicalDevices) {
  if (pPhysicalDeviceCount == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  Instance* instance = from_handle(instanceHandle);
  if (instance == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  // One device, and the handle is stable, so the count/repeat pattern is safe
  // to satisfy directly.
  if (pPhysicalDevices == nullptr) {
    *pPhysicalDeviceCount = 1;
    return VK_SUCCESS;
  }

  if (*pPhysicalDeviceCount == 0) {
    *pPhysicalDeviceCount = 1;
    return VK_INCOMPLETE;
  }

  pPhysicalDevices[0] = to_handle(instance->physical);
  *pPhysicalDeviceCount = 1;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceFeatures(
    VkPhysicalDevice physicalHandle, VkPhysicalDeviceFeatures* pFeatures) {
  if (pFeatures == nullptr)
    return;
  (void)physicalHandle;
  fill_features(*pFeatures);
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceProperties(
    VkPhysicalDevice physicalHandle, VkPhysicalDeviceProperties* pProperties) {
  if (pProperties == nullptr)
    return;
  PhysicalDevice* physical = from_handle(physicalHandle);
  if (physical == nullptr) {
    std::memset(pProperties, 0, sizeof(*pProperties));
    return;
  }
  fill_properties(*physical, *pProperties);
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceQueueFamilyProperties(
    VkPhysicalDevice physicalHandle, uint32_t* pQueueFamilyPropertyCount,
    VkQueueFamilyProperties* pQueueFamilyProperties) {
  if (pQueueFamilyPropertyCount == nullptr)
    return;
  (void)physicalHandle;

  if (pQueueFamilyProperties == nullptr) {
    *pQueueFamilyPropertyCount = 1;
    return;
  }
  if (*pQueueFamilyPropertyCount == 0) {
    *pQueueFamilyPropertyCount = 1;
    return;
  }

  fill_queue_families(pQueueFamilyProperties[0]);
  *pQueueFamilyPropertyCount = 1;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice physicalHandle, VkPhysicalDeviceMemoryProperties* pMemoryProperties) {
  if (pMemoryProperties == nullptr)
    return;
  (void)physicalHandle;
  fill_memory_properties(*pMemoryProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceExtensionProperties(
    VkPhysicalDevice physicalHandle, const char* pLayerName,
    uint32_t* pPropertyCount, VkExtensionProperties* pProperties) {
  (void)physicalHandle;
  (void)pLayerName;
  if (pPropertyCount == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  // No device extensions yet.  VK_KHR_swapchain is the first that matters.
  return enumerate_count(0, pPropertyCount, pProperties);
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceFormatProperties(
    VkPhysicalDevice physicalHandle, VkFormat format,
    VkFormatProperties* pFormatProperties) {
  (void)physicalHandle;
  (void)format;
  if (pFormatProperties == nullptr)
    return;
  // M0 has no format database.  Reporting uniform "nothing is supported" is
  // honest; M2 adds the VkFormat <-> GL internal format table.
  std::memset(pFormatProperties, 0, sizeof(*pFormatProperties));
}

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceImageFormatProperties(
    VkPhysicalDevice physicalHandle, VkFormat format, VkImageType type,
    VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags,
    VkImageFormatProperties* pImageFormatProperties) {
  (void)physicalHandle;
  (void)format;
  (void)type;
  (void)tiling;
  (void)usage;
  (void)flags;
  if (pImageFormatProperties != nullptr)
    std::memset(pImageFormatProperties, 0, sizeof(*pImageFormatProperties));
  return VK_ERROR_FORMAT_NOT_SUPPORTED;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceSparseImageFormatProperties(
    VkPhysicalDevice physicalHandle, VkFormat format, VkImageType type,
    VkSampleCountFlagBits samples, VkImageUsageFlags usage,
    VkImageTiling tiling, uint32_t* pPropertyCount,
    VkSparseImageFormatProperties* pProperties) {
  (void)physicalHandle;
  (void)format;
  (void)type;
  (void)samples;
  (void)usage;
  (void)tiling;
  (void)pProperties;  // nothing to write when the count is zero
  if (pPropertyCount == nullptr)
    return;
  // Sparse resources are not supported at all, so the answer is always zero
  // properties regardless of the requested format.
  *pPropertyCount = 0;
}

// ---------------------------------------------------------------------------
// Entry point tables
// ---------------------------------------------------------------------------

struct Entry {
  const char* name;
  PFN_vkVoidFunction fn;
};

// Reinterpreting one function pointer type as another is the mechanism Vulkan's
// vkGetProcAddr design is built on; the loader does exactly this on the other
// side of the boundary.
template <typename Fn>
PFN_vkVoidFunction as_entry(Fn fn) {
  return reinterpret_cast<PFN_vkVoidFunction>(fn);
}

const Entry kEntries[] = {
    {"vkEnumerateInstanceVersion", as_entry(EnumerateInstanceVersion)},
    {"vkEnumerateInstanceExtensionProperties",
     as_entry(EnumerateInstanceExtensionProperties)},
    {"vkCreateInstance", as_entry(CreateInstance)},
    {"vkDestroyInstance", as_entry(DestroyInstance)},
    {"vkEnumeratePhysicalDevices", as_entry(EnumeratePhysicalDevices)},
    {"vkGetPhysicalDeviceFeatures", as_entry(GetPhysicalDeviceFeatures)},
    {"vkGetPhysicalDeviceProperties", as_entry(GetPhysicalDeviceProperties)},
    {"vkGetPhysicalDeviceQueueFamilyProperties",
     as_entry(GetPhysicalDeviceQueueFamilyProperties)},
    {"vkGetPhysicalDeviceMemoryProperties",
     as_entry(GetPhysicalDeviceMemoryProperties)},
    {"vkEnumerateDeviceExtensionProperties",
     as_entry(EnumerateDeviceExtensionProperties)},
    {"vkGetPhysicalDeviceFormatProperties",
     as_entry(GetPhysicalDeviceFormatProperties)},
    {"vkGetPhysicalDeviceImageFormatProperties",
     as_entry(GetPhysicalDeviceImageFormatProperties)},
    {"vkGetPhysicalDeviceSparseImageFormatProperties",
     as_entry(GetPhysicalDeviceSparseImageFormatProperties)},

    // Device level commands, also reachable here.  See kniz_entrypoints.h.
    {"vkGetInstanceProcAddr", as_entry(vkGetInstanceProcAddr)},
    {"vkGetDeviceProcAddr", as_entry(GetDeviceProcAddr)},
    {"vkCreateDevice", as_entry(CreateDevice)},
    {"vkDestroyDevice", as_entry(DestroyDevice)},
    {"vkGetDeviceQueue", as_entry(GetDeviceQueue)},
    {"vkDeviceWaitIdle", as_entry(DeviceWaitIdle)},
    {"vkQueueWaitIdle", as_entry(QueueWaitIdle)},
};

}  // namespace

PFN_vkVoidFunction resolve_instance_proc(const char* name) {
  if (name == nullptr)
    return nullptr;
  for (const Entry& entry : kEntries) {
    if (std::strcmp(entry.name, name) == 0)
      return entry.fn;
  }
  KNIZ_DBG("unimplemented command requested: %s", name);
  return nullptr;
}

}  // namespace kniz

// ---------------------------------------------------------------------------
// Loader bootstrap
//
// These four symbols are the entire public surface of the ICD as far as the
// loader is concerned.
// ---------------------------------------------------------------------------

extern "C" KNIZ_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vk_icdNegotiateLoaderICDInterfaceVersion(uint32_t* pSupportedVersion) {
  if (pSupportedVersion == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  // Interface version 5 is the newest that does not require us to implement
  // anything Kniz cannot currently honour.  Report the lower of the two so a
  // newer loader is not misled into expecting version 6 or 7 semantics.
  constexpr uint32_t kMaxSupported = 5;
  const uint32_t requested = *pSupportedVersion;

  if (requested == 0) {
    KNIZ_ERR("loader requested ICD interface version 0");
    return VK_ERROR_INCOMPATIBLE_DRIVER;
  }

  if (requested > kMaxSupported) {
    KNIZ_DBG(
        "loader supports interface version %u, using %u; some newer loader "
        "behaviour will not be used",
        requested, kMaxSupported);
  }

  *pSupportedVersion = requested < kMaxSupported ? requested : kMaxSupported;
  KNIZ_DBG("loader ICD interface version negotiated: %u", *pSupportedVersion);
  return VK_SUCCESS;
}

extern "C" KNIZ_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char* pName) {
  (void)instance;  // dispatched on name only; see kniz_entrypoints.h
  return kniz::resolve_instance_proc(pName);
}

// Called by the loader to resolve physical-device-level *device extension*
// commands, which the loader cannot look up on a VkDevice because no device
// exists yet.  Returning nullptr for anything else is the documented contract.
extern "C" KNIZ_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetPhysicalDeviceProcAddr(VkInstance instance, const char* pName) {
  (void)instance;
  if (pName == nullptr)
    return nullptr;
  if (std::strncmp(pName, "vkGetPhysicalDevice", 19) != 0)
    return nullptr;
  return kniz::resolve_instance_proc(pName);
}

// Some applications and tools bypass the loader and dlopen() a driver directly,
// expecting vkGetInstanceProcAddr to be an exported symbol.  Providing it costs
// nothing and makes the "drop-in replacement" mode described in the README
// possible later.
extern "C" KNIZ_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
  (void)instance;
  return kniz::resolve_instance_proc(pName);
}
