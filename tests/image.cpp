// M2a test: images, end to end, through the real loader.
//
// An image is the first thing Kniz backs with something other than a host block.
// A buffer's contents have one representation -- bytes -- so the backend can keep
// a copy and shuffle it in and out.  An image has two: the layout Vulkan
// describes, and whatever OpenGL stores, which is opaque.  So an image lives only
// in a GL texture and every read and write goes through a command, which is what
// makes this a milestone rather than a detail.
//
// The checks are numeric rather than structural, for the same reason the compute
// test's are: swap the byte order between the two layouts, or reverse a row
// order, and the result is still an image that looks like an image.
//
//   1. a buffer -> image -> buffer round trip returns the bytes it was given, with
//      a pattern that varies along both axes, so neither a transpose nor a flip
//      survives it;
//   2. vkCmdClearColorImage produces exactly the texel values its float clear
//      value denotes;
//   3. a sub-rectangle copy lands in the right part of the image, with a padded
//      buffer row length treated as a stride, and leaves the rest alone;
//   4. the refusals that keep the rest honest really do refuse.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "harness.h"

using namespace kniz_test;

namespace {

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 32;
constexpr uint32_t kBytesPerPixel = 4;
constexpr uint32_t kImageBytes = kWidth * kHeight * kBytesPerPixel;

// The colour phase 2 clears to, as k/255 so that the float -> unorm conversion is
// exact and the expected bytes are the numerators.
constexpr uint8_t kClearRed = 64;
constexpr uint8_t kClearGreen = 128;
constexpr uint8_t kClearBlue = 192;
constexpr uint8_t kClearAlpha = 255;

// A pattern that varies with x and with y, in every component.  A translation
// that flips the image, or confuses the row stride with the column stride,
// changes nearly every texel -- which is the point of writing it this way rather
// than with something symmetric that would survive the mistake.
std::vector<uint8_t> pattern(uint32_t width, uint32_t height, uint32_t x0, uint32_t y0) {
  std::vector<uint8_t> bytes(static_cast<size_t>(width) * height * kBytesPerPixel);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      const uint32_t gx = x0 + x;
      const uint32_t gy = y0 + y;
      uint8_t* texel =
          bytes.data() + (static_cast<size_t>(y) * width + x) * kBytesPerPixel;
      texel[0] = static_cast<uint8_t>(gx * 3u);
      texel[1] = static_cast<uint8_t>(gy * 5u);
      texel[2] = static_cast<uint8_t>(gx ^ gy);
      texel[3] = 255;
    }
  }
  return bytes;
}

}  // namespace

int main() {
  std::printf("kniz image test (M2a)\n");

  if (!icd_selected())
    return 2;

  void* loader = open_loader();
  if (loader == nullptr)
    return 2;
  if (!resolve_get_instance_proc_addr(loader))
    return 2;

  Runtime runtime;
  if (!bring_up(&runtime, "kniz-image"))
    return 1;

  VkDevice device = runtime.device;
  VkQueue queue = runtime.queue;

  auto create_buffer = device_proc<PFN_vkCreateBuffer>(device, "vkCreateBuffer");
  auto destroy_buffer = device_proc<PFN_vkDestroyBuffer>(device, "vkDestroyBuffer");
  auto get_buffer_requirements =
      device_proc<PFN_vkGetBufferMemoryRequirements>(device, "vkGetBufferMemoryRequirements");
  auto create_image = device_proc<PFN_vkCreateImage>(device, "vkCreateImage");
  auto destroy_image = device_proc<PFN_vkDestroyImage>(device, "vkDestroyImage");
  auto get_image_requirements =
      device_proc<PFN_vkGetImageMemoryRequirements>(device, "vkGetImageMemoryRequirements");
  auto bind_image_memory = device_proc<PFN_vkBindImageMemory>(device, "vkBindImageMemory");
  auto create_image_view = device_proc<PFN_vkCreateImageView>(device, "vkCreateImageView");
  auto destroy_image_view = device_proc<PFN_vkDestroyImageView>(device, "vkDestroyImageView");
  auto allocate_memory = device_proc<PFN_vkAllocateMemory>(device, "vkAllocateMemory");
  auto free_memory = device_proc<PFN_vkFreeMemory>(device, "vkFreeMemory");
  auto bind_buffer_memory = device_proc<PFN_vkBindBufferMemory>(device, "vkBindBufferMemory");
  auto map_memory = device_proc<PFN_vkMapMemory>(device, "vkMapMemory");
  auto unmap_memory = device_proc<PFN_vkUnmapMemory>(device, "vkUnmapMemory");
  auto create_command_pool = device_proc<PFN_vkCreateCommandPool>(device, "vkCreateCommandPool");
  auto destroy_command_pool = device_proc<PFN_vkDestroyCommandPool>(device, "vkDestroyCommandPool");
  auto allocate_command_buffers =
      device_proc<PFN_vkAllocateCommandBuffers>(device, "vkAllocateCommandBuffers");
  auto begin_command_buffer = device_proc<PFN_vkBeginCommandBuffer>(device, "vkBeginCommandBuffer");
  auto end_command_buffer = device_proc<PFN_vkEndCommandBuffer>(device, "vkEndCommandBuffer");
  auto cmd_pipeline_barrier = device_proc<PFN_vkCmdPipelineBarrier>(device, "vkCmdPipelineBarrier");
  auto cmd_clear_color_image =
      device_proc<PFN_vkCmdClearColorImage>(device, "vkCmdClearColorImage");
  auto cmd_copy_buffer_to_image =
      device_proc<PFN_vkCmdCopyBufferToImage>(device, "vkCmdCopyBufferToImage");
  auto cmd_copy_image_to_buffer =
      device_proc<PFN_vkCmdCopyImageToBuffer>(device, "vkCmdCopyImageToBuffer");
  auto queue_submit = device_proc<PFN_vkQueueSubmit>(device, "vkQueueSubmit");
  auto create_fence = device_proc<PFN_vkCreateFence>(device, "vkCreateFence");
  auto destroy_fence = device_proc<PFN_vkDestroyFence>(device, "vkDestroyFence");
  auto reset_fences = device_proc<PFN_vkResetFences>(device, "vkResetFences");
  auto wait_for_fences = device_proc<PFN_vkWaitForFences>(device, "vkWaitForFences");
  auto get_image_format_properties = instance_proc<PFN_vkGetPhysicalDeviceImageFormatProperties>(
      runtime.instance, "vkGetPhysicalDeviceImageFormatProperties");

  require("vkCreateImage", reinterpret_cast<PFN_vkVoidFunction>(create_image));
  require("vkCmdCopyBufferToImage",
          reinterpret_cast<PFN_vkVoidFunction>(cmd_copy_buffer_to_image));
  require("vkCmdClearColorImage",
          reinterpret_cast<PFN_vkVoidFunction>(cmd_clear_color_image));
  if (failures() > 0) {
    tear_down(&runtime);
    return 1;
  }

  const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

  // ---- the driver answers before it is asked to act -----------------------
  // An application picks its resource strategy from this call, so a lie here
  // surfaces much later and much further from its cause.
  {
    VkImageFormatProperties properties{};
    const VkResult supported = get_image_format_properties(
        runtime.physical, format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0, &properties);
    check(supported == VK_SUCCESS, "optimal tiling is reported as supported");
    check(properties.maxExtent.width >= kWidth && properties.maxMipLevels > 0,
          "a supported format comes back with usable limits");

    const VkResult linear = get_image_format_properties(
        runtime.physical, format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_LINEAR,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0, &properties);
    check(linear == VK_ERROR_FORMAT_NOT_SUPPORTED,
          "linear tiling is reported as unsupported");

    const VkResult storage = get_image_format_properties(
        runtime.physical, format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT, 0, &properties);
    check(storage == VK_ERROR_FORMAT_NOT_SUPPORTED,
          "storage images are reported as unsupported");
  }

  // ---- objects ------------------------------------------------------------
  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = format;
  image_info.extent.width = kWidth;
  image_info.extent.height = kHeight;
  image_info.extent.depth = 1;
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VkImage image = VK_NULL_HANDLE;
  check(create_image(device, &image_info, nullptr, &image) == VK_SUCCESS &&
            image != VK_NULL_HANDLE,
        "vkCreateImage");
  if (image == VK_NULL_HANDLE) {
    tear_down(&runtime);
    return 1;
  }

  VkMemoryRequirements image_requirements{};
  get_image_requirements(device, image, &image_requirements);
  check(image_requirements.size >= kImageBytes,
        "vkGetImageMemoryRequirements covers the image's texels");

  VkDeviceMemory image_memory = VK_NULL_HANDLE;
  {
    VkMemoryAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    info.allocationSize = image_requirements.size;
    info.memoryTypeIndex = runtime.host_memory_type;
    check(allocate_memory(device, &info, nullptr, &image_memory) == VK_SUCCESS,
          "vkAllocateMemory (image)");
    check(bind_image_memory(device, image, image_memory, 0) == VK_SUCCESS,
          "vkBindImageMemory creates the GL texture storage");

    // The emulation's central limitation, asserted rather than merely described:
    // an image is not in the host block, so a mapping of its allocation cannot be
    // honoured, and handing back a pointer anyway would show the application
    // bytes that are not the image.
    void* mapped = nullptr;
    const VkResult result = map_memory(device, image_memory, 0, VK_WHOLE_SIZE, 0, &mapped);
    check(result == VK_ERROR_MEMORY_MAP_FAILED && mapped == nullptr,
          "mapping an allocation an image is bound to is refused");
  }

  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = format;
  view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  view_info.subresourceRange.levelCount = 1;
  view_info.subresourceRange.layerCount = 1;

  VkImageView view = VK_NULL_HANDLE;
  check(create_image_view(device, &view_info, nullptr, &view) == VK_SUCCESS,
        "vkCreateImageView");

  // A view that reinterprets the format needs glTextureView, so it must be
  // refused rather than sampled with the transfer function missing.
  {
    VkImageViewCreateInfo srgb = view_info;
    srgb.format = VK_FORMAT_R8G8B8A8_SRGB;
    VkImageView rejected = VK_NULL_HANDLE;
    check(create_image_view(device, &srgb, nullptr, &rejected) ==
                  VK_ERROR_FORMAT_NOT_SUPPORTED &&
              rejected == VK_NULL_HANDLE,
          "a view with a different format is refused");
  }

  auto make_buffer = [&](VkDeviceSize size, VkBuffer* out, VkDeviceMemory* memory) {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (create_buffer(device, &info, nullptr, out) != VK_SUCCESS)
      return false;

    VkMemoryRequirements requirements{};
    get_buffer_requirements(device, *out, &requirements);
    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = runtime.host_memory_type;
    if (allocate_memory(device, &allocate, nullptr, memory) != VK_SUCCESS)
      return false;
    return bind_buffer_memory(device, *out, *memory, 0) == VK_SUCCESS;
  };

  VkBuffer source_buffer = VK_NULL_HANDLE;
  VkBuffer result_buffer = VK_NULL_HANDLE;
  VkDeviceMemory source_memory = VK_NULL_HANDLE;
  VkDeviceMemory result_memory = VK_NULL_HANDLE;
  check(make_buffer(kImageBytes, &source_buffer, &source_memory),
        "staging buffer for the upload");
  check(make_buffer(kImageBytes, &result_buffer, &result_memory),
        "staging buffer for the readback");

  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = 0;
  VkCommandPool pool = VK_NULL_HANDLE;
  create_command_pool(device, &pool_info, nullptr, &pool);

  VkCommandBufferAllocateInfo command_allocate{};
  command_allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_allocate.commandPool = pool;
  command_allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_allocate.commandBufferCount = 1;
  VkCommandBuffer command = VK_NULL_HANDLE;
  allocate_command_buffers(device, &command_allocate, &command);

  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence = VK_NULL_HANDLE;
  create_fence(device, &fence_info, nullptr, &fence);

  auto submit = [&](VkCommandBuffer buffer) {
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &buffer;
    reset_fences(device, 1, &fence);
    if (queue_submit(queue, 1, &submit_info, fence) != VK_SUCCESS)
      return false;
    return wait_for_fences(device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
  };

  auto begin = [&]() {
    VkCommandBufferBeginInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_command_buffer(command, &info);
  };

  auto write_staging = [&](const std::vector<uint8_t>& bytes, const char* what) {
    void* mapped = nullptr;
    if (map_memory(device, source_memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
      check(false, what);
      return;
    }
    std::memcpy(mapped, bytes.data(), bytes.size());
    unmap_memory(device, source_memory);
  };

  auto read_staging = [&]() {
    std::vector<uint8_t> bytes(kImageBytes, 0);
    void* mapped = nullptr;
    if (map_memory(device, result_memory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
      std::memcpy(bytes.data(), mapped, kImageBytes);
      unmap_memory(device, result_memory);
    }
    return bytes;
  };

  auto whole_image_region = [&]() {
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {kWidth, kHeight, 1};
    return region;
  };

  // The layout transitions an application records around its transfers.  GL has
  // no image layouts, so these are here to prove the barrier path records and
  // replays without disturbing the data.
  auto transition = [&](VkImageLayout from, VkImageLayout to) {
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
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    cmd_pipeline_barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &barrier);
  };

  // ---- phase 1: buffer -> image -> buffer ---------------------------------
  const std::vector<uint8_t> original = pattern(kWidth, kHeight, 0, 0);
  write_staging(original, "staging buffer mapped for the round trip");

  {
    const VkBufferImageCopy region = whole_image_region();
    begin();
    transition(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    cmd_copy_buffer_to_image(command, source_buffer, image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    transition(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    cmd_copy_image_to_buffer(command, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             result_buffer, 1, &region);
    end_command_buffer(command);
    check(submit(command), "upload and readback submitted");
  }

  {
    const std::vector<uint8_t> readback = read_staging();
    const bool same = std::memcmp(readback.data(), original.data(), kImageBytes) == 0;
    check(same, "an image round trip returns the bytes it was given");
    if (!same) {
      for (uint32_t i = 0; i < kImageBytes; ++i) {
        if (readback[i] != original[i]) {
          std::printf("     first difference at byte %u (row %u, column %u, "
                      "component %u): got %u, expected %u\n",
                      i, i / (kWidth * kBytesPerPixel),
                      (i / kBytesPerPixel) % kWidth, i % kBytesPerPixel, readback[i],
                      original[i]);
          break;
        }
      }
    }
  }

  // ---- phase 2: a clear value arrives as exact texels ---------------------
  {
    VkClearColorValue clear{};
    clear.float32[0] = static_cast<float>(kClearRed) / 255.0f;
    clear.float32[1] = static_cast<float>(kClearGreen) / 255.0f;
    clear.float32[2] = static_cast<float>(kClearBlue) / 255.0f;
    clear.float32[3] = 1.0f;

    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel = 0;
    range.levelCount = 1;
    range.baseArrayLayer = 0;
    range.layerCount = 1;

    const VkBufferImageCopy region = whole_image_region();
    begin();
    cmd_clear_color_image(command, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1,
                          &range);
    cmd_copy_image_to_buffer(command, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             result_buffer, 1, &region);
    end_command_buffer(command);
    check(submit(command), "clear and readback submitted");
  }

  {
    const std::vector<uint8_t> readback = read_staging();
    bool exact = true;
    for (uint32_t i = 0; i < kImageBytes; i += kBytesPerPixel) {
      exact = exact && readback[i + 0] == kClearRed && readback[i + 1] == kClearGreen &&
              readback[i + 2] == kClearBlue && readback[i + 3] == kClearAlpha;
    }
    check(exact, "vkCmdClearColorImage produced the exact texel values");
    if (!exact) {
      std::printf("     first texel is %u %u %u %u, expected %u %u %u %u\n", readback[0],
                  readback[1], readback[2], readback[3], kClearRed, kClearGreen,
                  kClearBlue, kClearAlpha);
    }
  }

  // ---- phase 3: a sub-rectangle goes to the right place -------------------
  // The upload names rows 8..24 and columns 16..48 of the image, out of a buffer
  // whose rows are padded to be wider than the region.  Both of those are things a
  // translation gets wrong quietly: a padded row length is a stride and not a
  // skip, and the region's offset decides which texels are overwritten.
  constexpr uint32_t kRegionX = 16;
  constexpr uint32_t kRegionY = 8;
  constexpr uint32_t kRegionWidth = 32;
  constexpr uint32_t kRegionHeight = 16;
  constexpr uint32_t kPaddedRowPixels = kRegionWidth + 7;  // deliberately not tight

  const std::vector<uint8_t> rectangle = pattern(kRegionWidth, kRegionHeight, 100, 40);
  {
    // Only the first kRegionWidth pixels of each row are the region.  The padding
    // is filled with a value the pattern never produces, so a translation that
    // read it would show up as a wrong texel rather than as a plausible one.
    std::vector<uint8_t> padded(
        static_cast<size_t>(kPaddedRowPixels) * kRegionHeight * kBytesPerPixel, 0xEE);
    for (uint32_t y = 0; y < kRegionHeight; ++y) {
      std::memcpy(padded.data() +
                      static_cast<size_t>(y) * kPaddedRowPixels * kBytesPerPixel,
                  rectangle.data() +
                      static_cast<size_t>(y) * kRegionWidth * kBytesPerPixel,
                  kRegionWidth * kBytesPerPixel);
    }
    write_staging(padded, "staging buffer mapped for the sub-rectangle copy");
  }

  {
    VkBufferImageCopy upload{};
    upload.bufferOffset = 0;
    upload.bufferRowLength = kPaddedRowPixels;
    upload.bufferImageHeight = kRegionHeight;
    upload.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    upload.imageSubresource.layerCount = 1;
    upload.imageOffset = {kRegionX, kRegionY, 0};
    upload.imageExtent = {kRegionWidth, kRegionHeight, 1};

    // The readback covers the whole image, tightly packed, so the texels the copy
    // did not name are checked as well as the ones it did.
    const VkBufferImageCopy whole = whole_image_region();
    begin();
    cmd_copy_buffer_to_image(command, source_buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1,
                             &upload);
    cmd_copy_image_to_buffer(command, image, VK_IMAGE_LAYOUT_GENERAL, result_buffer, 1,
                             &whole);
    end_command_buffer(command);
    check(submit(command), "sub-rectangle copy submitted");
  }

  {
    const std::vector<uint8_t> readback = read_staging();
    uint32_t wrong_inside = 0;
    uint32_t wrong_outside = 0;
    for (uint32_t y = 0; y < kHeight; ++y) {
      for (uint32_t x = 0; x < kWidth; ++x) {
        const uint8_t* got =
            readback.data() + (static_cast<size_t>(y) * kWidth + x) * kBytesPerPixel;
        const bool inside = x >= kRegionX && x < kRegionX + kRegionWidth &&
                            y >= kRegionY && y < kRegionY + kRegionHeight;
        if (inside) {
          const uint8_t* want =
              rectangle.data() +
              (static_cast<size_t>(y - kRegionY) * kRegionWidth + (x - kRegionX)) *
                  kBytesPerPixel;
          if (std::memcmp(got, want, kBytesPerPixel) != 0)
            ++wrong_inside;
        } else if (got[0] != kClearRed || got[1] != kClearGreen ||
                   got[2] != kClearBlue || got[3] != kClearAlpha) {
          ++wrong_outside;
        }
      }
    }
    check(wrong_inside == 0, "the copied rectangle holds the source texels");
    check(wrong_outside == 0, "texels outside the rectangle are left alone");
    if (wrong_inside != 0 || wrong_outside != 0) {
      std::printf("     %u texel(s) wrong inside the rectangle, %u outside\n",
                  wrong_inside, wrong_outside);
    }
  }

  // ---- refusals -----------------------------------------------------------
  {
    VkImage rejected = VK_NULL_HANDLE;

    VkImageCreateInfo linear = image_info;
    linear.tiling = VK_IMAGE_TILING_LINEAR;
    check(create_image(device, &linear, nullptr, &rejected) ==
                  VK_ERROR_FEATURE_NOT_PRESENT &&
              rejected == VK_NULL_HANDLE,
          "linear tiling is refused at vkCreateImage");

    VkImageCreateInfo multisampled = image_info;
    multisampled.samples = VK_SAMPLE_COUNT_4_BIT;
    check(create_image(device, &multisampled, nullptr, &rejected) ==
                  VK_ERROR_FEATURE_NOT_PRESENT &&
              rejected == VK_NULL_HANDLE,
          "multisampled images are refused at vkCreateImage");

    VkImageCreateInfo storage = image_info;
    storage.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    check(create_image(device, &storage, nullptr, &rejected) ==
                  VK_ERROR_FEATURE_NOT_PRESENT &&
              rejected == VK_NULL_HANDLE,
          "storage images are refused at vkCreateImage");

    VkImageCreateInfo unknown_format = image_info;
    unknown_format.format = VK_FORMAT_R8G8B8_UNORM;
    check(create_image(device, &unknown_format, nullptr, &rejected) ==
                  VK_ERROR_FORMAT_NOT_SUPPORTED &&
              rejected == VK_NULL_HANDLE,
          "a format with no OpenGL counterpart is refused");
  }

  // ---- teardown -----------------------------------------------------------
  destroy_fence(device, fence, nullptr);
  destroy_command_pool(device, pool, nullptr);
  destroy_image_view(device, view, nullptr);
  destroy_buffer(device, result_buffer, nullptr);
  destroy_buffer(device, source_buffer, nullptr);
  free_memory(device, result_memory, nullptr);
  free_memory(device, source_memory, nullptr);
  destroy_image(device, image, nullptr);
  free_memory(device, image_memory, nullptr);
  tear_down(&runtime);
  check(true, "teardown");

  if (failures() == 0) {
    std::printf("PASSED\n");
    return 0;
  }
  std::printf("FAILED: %d check(s)\n", failures());
  return 1;
}
