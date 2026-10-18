#pragma once

// Device level entry points.
//
// These live in their own translation unit because it keeps the instance and
// device tables from tangling, and because M2 onwards will add a lot here.

#include <vulkan/vulkan.h>

namespace kniz {

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(
    VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDevice* pDevice);

VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice device,
                                         const VkAllocationCallbacks* pAllocator);

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device,
                                                           const char* pName);

VKAPI_ATTR void VKAPI_CALL GetDeviceQueue(VkDevice device,
                                          uint32_t queueFamilyIndex,
                                          uint32_t queueIndex, VkQueue* pQueue);

VKAPI_ATTR VkResult VKAPI_CALL DeviceWaitIdle(VkDevice device);

VKAPI_ATTR VkResult VKAPI_CALL QueueWaitIdle(VkQueue queue);

// ---------------------------------------------------------------------------
// M1: buffers, memory and descriptors
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateBuffer(
    VkDevice device, const VkBufferCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkBuffer* pBuffer);

VKAPI_ATTR void VKAPI_CALL DestroyBuffer(VkDevice device, VkBuffer buffer,
                                         const VkAllocationCallbacks* pAllocator);

VKAPI_ATTR void VKAPI_CALL GetBufferMemoryRequirements(
    VkDevice device, VkBuffer buffer, VkMemoryRequirements* pMemoryRequirements);

VKAPI_ATTR VkResult VKAPI_CALL AllocateMemory(
    VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo,
    const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory);

VKAPI_ATTR void VKAPI_CALL FreeMemory(VkDevice device, VkDeviceMemory memory,
                                      const VkAllocationCallbacks* pAllocator);

VKAPI_ATTR VkResult VKAPI_CALL BindBufferMemory(VkDevice device, VkBuffer buffer,
                                                VkDeviceMemory memory,
                                                VkDeviceSize memoryOffset);

VKAPI_ATTR VkResult VKAPI_CALL MapMemory(VkDevice device, VkDeviceMemory memory,
                                         VkDeviceSize offset, VkDeviceSize size,
                                         VkMemoryMapFlags flags, void** ppData);

VKAPI_ATTR void VKAPI_CALL UnmapMemory(VkDevice device, VkDeviceMemory memory);

VKAPI_ATTR VkResult VKAPI_CALL FlushMappedMemoryRanges(
    VkDevice device, uint32_t memRangeCount, const VkMappedMemoryRange* pMemRanges);

VKAPI_ATTR VkResult VKAPI_CALL InvalidateMappedMemoryRanges(
    VkDevice device, uint32_t memRangeCount, const VkMappedMemoryRange* pMemRanges);

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorSetLayout(
    VkDevice device, const VkDescriptorSetLayoutCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDescriptorSetLayout* pSetLayout);

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorSetLayout(
    VkDevice device, VkDescriptorSetLayout descriptorSetLayout,
    const VkAllocationCallbacks* pAllocator);

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineLayout(
    VkDevice device, const VkPipelineLayoutCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkPipelineLayout* pPipelineLayout);

VKAPI_ATTR void VKAPI_CALL DestroyPipelineLayout(
    VkDevice device, VkPipelineLayout pipelineLayout,
    const VkAllocationCallbacks* pAllocator);

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorPool(
    VkDevice device, const VkDescriptorPoolCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDescriptorPool* pDescriptorPool);

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorPool(
    VkDevice device, VkDescriptorPool descriptorPool,
    const VkAllocationCallbacks* pAllocator);

VKAPI_ATTR VkResult VKAPI_CALL ResetDescriptorPool(
    VkDevice device, VkDescriptorPool descriptorPool,
    VkDescriptorPoolResetFlags flags);

VKAPI_ATTR VkResult VKAPI_CALL AllocateDescriptorSets(
    VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo,
    VkDescriptorSet* pDescriptorSets);

VKAPI_ATTR VkResult VKAPI_CALL FreeDescriptorSets(
    VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount,
    const VkDescriptorSet* pDescriptorSets);

VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSets(
    VkDevice device, uint32_t descriptorWriteCount,
    const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount,
    const VkCopyDescriptorSet* pDescriptorCopies);

// ---------------------------------------------------------------------------
// M1: shader modules and pipelines
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateShaderModule(
    VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule);

VKAPI_ATTR void VKAPI_CALL DestroyShaderModule(
    VkDevice device, VkShaderModule shaderModule,
    const VkAllocationCallbacks* pAllocator);

VKAPI_ATTR VkResult VKAPI_CALL CreateComputePipelines(
    VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
    const VkComputePipelineCreateInfo* pCreateInfos,
    const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines);

VKAPI_ATTR VkResult VKAPI_CALL CreateGraphicsPipelines(
    VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
    const VkGraphicsPipelineCreateInfo* pCreateInfos,
    const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines);

VKAPI_ATTR void VKAPI_CALL DestroyPipeline(VkDevice device, VkPipeline pipeline,
                                           const VkAllocationCallbacks* pAllocator);

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineCache(
    VkDevice device, const VkPipelineCacheCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkPipelineCache* pPipelineCache);

VKAPI_ATTR void VKAPI_CALL DestroyPipelineCache(
    VkDevice device, VkPipelineCache pipelineCache,
    const VkAllocationCallbacks* pAllocator);

VKAPI_ATTR VkResult VKAPI_CALL GetPipelineCacheData(
    VkDevice device, VkPipelineCache pipelineCache, size_t* pDataSize,
    void* pData);

VKAPI_ATTR VkResult VKAPI_CALL MergePipelineCaches(
    VkDevice device, VkPipelineCache dstCache, uint32_t srcCacheCount,
    const VkPipelineCache* pSrcCaches);

// ---------------------------------------------------------------------------
// M1: command buffers and submission
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateCommandPool(
    VkDevice device, const VkCommandPoolCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkCommandPool* pCommandPool);

VKAPI_ATTR void VKAPI_CALL DestroyCommandPool(
    VkDevice device, VkCommandPool commandPool,
    const VkAllocationCallbacks* pAllocator);

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandPool(
    VkDevice device, VkCommandPool commandPool, VkCommandPoolResetFlags flags);

VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(
    VkDevice device, const VkCommandBufferAllocateInfo* pAllocateInfo,
    VkCommandBuffer* pCommandBuffers);

VKAPI_ATTR void VKAPI_CALL FreeCommandBuffers(
    VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount,
    const VkCommandBuffer* pCommandBuffers);

VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(
    VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo* pBeginInfo);

VKAPI_ATTR VkResult VKAPI_CALL EndCommandBuffer(VkCommandBuffer commandBuffer);

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandBuffer(
    VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags);

VKAPI_ATTR void VKAPI_CALL CmdBindPipeline(VkCommandBuffer commandBuffer,
                                           VkPipelineBindPoint pipelineBindPoint,
                                           VkPipeline pipeline);

VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSets(
    VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
    VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount,
    const VkDescriptorSet* pDescriptorSets, uint32_t dynamicOffsetCount,
    const uint32_t* pDynamicOffsets);

VKAPI_ATTR void VKAPI_CALL CmdPushConstants(VkCommandBuffer commandBuffer,
                                            VkPipelineLayout layout,
                                            VkShaderStageFlags stageFlags,
                                            uint32_t offset, uint32_t size,
                                            const void* pValues);

VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier(
    VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask,
    VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags,
    uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers,
    uint32_t bufferMemoryBarrierCount,
    const VkBufferMemoryBarrier* pBufferMemoryBarriers,
    uint32_t imageMemoryBarrierCount,
    const VkImageMemoryBarrier* pImageMemoryBarriers);

VKAPI_ATTR void VKAPI_CALL CmdDispatch(VkCommandBuffer commandBuffer,
                                       uint32_t groupCountX, uint32_t groupCountY,
                                       uint32_t groupCountZ);

VKAPI_ATTR void VKAPI_CALL CmdFillBuffer(VkCommandBuffer commandBuffer,
                                         VkBuffer dstBuffer,
                                         VkDeviceSize dstOffset,
                                         VkDeviceSize size, uint32_t data);

VKAPI_ATTR void VKAPI_CALL CmdUpdateBuffer(VkCommandBuffer commandBuffer,
                                           VkBuffer dstBuffer,
                                           VkDeviceSize dstOffset,
                                           VkDeviceSize dataSize,
                                           const void* pData);

VKAPI_ATTR void VKAPI_CALL CmdCopyBuffer(VkCommandBuffer commandBuffer,
                                         VkBuffer srcBuffer, VkBuffer dstBuffer,
                                         uint32_t regionCount,
                                         const VkBufferCopy* pRegions);

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(VkQueue queue, uint32_t submitCount,
                                           const VkSubmitInfo* pSubmits,
                                           VkFence fence);

VKAPI_ATTR VkResult VKAPI_CALL CreateFence(VkDevice device,
                                           const VkFenceCreateInfo* pCreateInfo,
                                           const VkAllocationCallbacks* pAllocator,
                                           VkFence* pFence);

VKAPI_ATTR void VKAPI_CALL DestroyFence(VkDevice device, VkFence fence,
                                        const VkAllocationCallbacks* pAllocator);

VKAPI_ATTR VkResult VKAPI_CALL ResetFences(VkDevice device, uint32_t fenceCount,
                                           const VkFence* pFences);

VKAPI_ATTR VkResult VKAPI_CALL GetFenceStatus(VkDevice device, VkFence fence);

VKAPI_ATTR VkResult VKAPI_CALL WaitForFences(VkDevice device, uint32_t fenceCount,
                                             const VkFence* pFences,
                                             VkBool32 waitAll, uint64_t timeout);

}  // namespace kniz
