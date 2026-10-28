// Images: Vulkan images on OpenGL textures.
//
// kniz_image.h says why an image is not mirrored into host memory.  What is here
// is the format table that turns a Vulkan format into GL state, the object entry
// points, and the three things a recorded command can do to an image: clear it,
// fill it from a buffer, and read it back into one.
//
// One thing is deliberately absent: a y flip.  Vulkan's framebuffer origin is the
// upper left and OpenGL's is the lower left, which sounds like a flip must be
// needed somewhere.  In the other direction -- translating OpenGL to Vulkan, which
// is what Zink does -- one is, and it is why Zink wants negative-height viewports.
// Here it is not, and the reason is worth writing down, because "fixing" it later
// would be a silent bug everywhere at once:
//
//     OpenGL maps clip space to window coordinates as y_w = (y_d + 1) * h / 2,
//     counted up from the lower left.  Vulkan maps it as y_f = (y_f + 1) * h / 2,
//     counted down from the upper left.  Both therefore put y_d = -1 in row 0 of
//     their image.  GL's row 0 is its bottom row and Vulkan's is its top row, but
//     that is only a difference in which word describes it.
//
// So Vulkan image row r and GL texture row r hold the same content.  Everything
// that consumes these images -- sampling, readback, and in time the swapchain --
// does so through Vulkan's convention, so GL's convention never applies and the
// row orders already agree.  Below, "top" means Vulkan's top.

#include "kniz_common.h"
#include "kniz_image.h"
#include "kniz_memory.h"
#include "kniz_objects.h"

#include <cstring>
#include <new>
#include <vector>

namespace kniz {
namespace {

// | Vulkan format                     | sized internal         | format            | type                 | Bpp | depth |
constexpr ImageFormat kFormats[] = {
    {VK_FORMAT_R8_UNORM, gl::kR8, gl::kRed, gl::kUnsignedByte, 1, false},
    {VK_FORMAT_R8G8_UNORM, gl::kRg8, gl::kRg, gl::kUnsignedByte, 2, false},
    {VK_FORMAT_R8G8B8A8_UNORM, gl::kRgba8, gl::kRgba, gl::kUnsignedByte, 4, false},
    {VK_FORMAT_R8G8B8A8_SRGB, gl::kSrgb8Alpha8, gl::kRgba, gl::kUnsignedByte, 4, false},
    {VK_FORMAT_B8G8R8A8_UNORM, gl::kRgba8, gl::kBgra, gl::kUnsignedByte, 4, false},
    {VK_FORMAT_B8G8R8A8_SRGB, gl::kSrgb8Alpha8, gl::kBgra, gl::kUnsignedByte, 4, false},
    {VK_FORMAT_R32_SFLOAT, gl::kR32f, gl::kRed, gl::kFloat, 4, false},
    {VK_FORMAT_R16G16B16A16_SFLOAT, gl::kRgba16f, gl::kRgba, gl::kHalfFloat, 8, false},
    {VK_FORMAT_R32G32B32A32_SFLOAT, gl::kRgba32f, gl::kRgba, gl::kFloat, 16, false},
    {VK_FORMAT_D16_UNORM, gl::kDepthComponent16, gl::kDepthComponent, gl::kUnsignedShort, 2, true},
    {VK_FORMAT_X8_D24_UNORM_PACK32, gl::kDepthComponent24, gl::kDepthComponent, gl::kUnsignedInt, 4, true},
    {VK_FORMAT_D32_SFLOAT, gl::kDepthComponent32f, gl::kDepthComponent, gl::kFloat, 4, true},
    {VK_FORMAT_D24_UNORM_S8_UINT, gl::kDepth24Stencil8, gl::kDepthStencil, gl::kUnsignedInt248, 4, true},
};

// Image creation flags that would be a lie rather than a simplification live in
// kniz_image.h, shared with vkGetPhysicalDeviceImageFormatProperties.

bool image_gl_ready(Device* device) {
  if (!device_can_run_gl(device))
    return false;
  if (!gl::make_current()) {
    KNIZ_ERR("could not make the OpenGL context current on this thread");
    return false;
  }
  return true;
}

// The address of a buffer's bytes.  Computed from the binding rather than stored
// on the buffer, so that rebinding memory cannot leave a stale pointer behind.
// The caller has already established `offset + length <= buffer->size`, and the
// binding was validated against the allocation, so the pointer is in range.
uint8_t* buffer_bytes(Buffer* buffer, uint64_t offset, uint64_t length,
                      const char* what) {
  if (buffer->memory == nullptr) {
    KNIZ_ERR("%s needs buffer %p to have memory bound, and it does not",
             what, static_cast<void*>(to_handle(buffer)));
    return nullptr;
  }
  if (offset + length > buffer->size) {
    KNIZ_ERR("%s touches bytes [%llu, %llu) of a %llu byte buffer", what,
             static_cast<unsigned long long>(offset),
             static_cast<unsigned long long>(offset + length),
             static_cast<unsigned long long>(buffer->size));
    return nullptr;
  }
  return buffer->memory->host + buffer->memory_offset + offset;
}

// The buffer-side geometry of a copy.
//
// Vulkan's bufferRowLength and bufferImageHeight are in texels and texel rows,
// with zero meaning "tightly packed".  bufferOffset points at the region's first
// texel, so a padded row length is a stride and never a skip: the region's column
// x sits at the start of each row.
struct RegionGeometry {
  uint32_t row_length_texels = 0;
  uint64_t span_bytes = 0;  // from the region's first byte to its last
};

bool region_geometry(const Image* image, const ImageFormat* format,
                     const VkBufferImageCopy& region, RegionGeometry* out,
                     const char* what) {
  if (region.imageSubresource.mipLevel != 0) {
    KNIZ_ERR("%s only supports mip level 0, and level %u was asked for", what,
             region.imageSubresource.mipLevel);
    return false;
  }
  if (region.imageSubresource.baseArrayLayer != 0 ||
      region.imageSubresource.layerCount != 1 || region.imageExtent.depth != 1) {
    KNIZ_ERR("%s only supports one array layer, and %u were asked for", what,
             region.imageSubresource.layerCount);
    return false;
  }
  if (region.imageExtent.width == 0 || region.imageExtent.height == 0) {
    KNIZ_ERR("%s was given an empty region", what);
    return false;
  }
  if (region.imageOffset.x + region.imageExtent.width > image->extent.width ||
      region.imageOffset.y + region.imageExtent.height > image->extent.height) {
    KNIZ_ERR("%s region [%u, %u) x [%u, %u) does not fit in a %ux%u image", what,
             region.imageOffset.x,
             region.imageOffset.x + region.imageExtent.width, region.imageOffset.y,
             region.imageOffset.y + region.imageExtent.height, image->extent.width,
             image->extent.height);
    return false;
  }

  out->row_length_texels = region.bufferRowLength != 0 ? region.bufferRowLength
                                                      : region.imageExtent.width;
  const uint64_t row_pitch =
      static_cast<uint64_t>(out->row_length_texels) * format->bytes_per_pixel;
  out->span_bytes = static_cast<uint64_t>(region.imageExtent.height - 1) * row_pitch +
                    static_cast<uint64_t>(region.imageExtent.width) *
                        format->bytes_per_pixel;
  return true;
}

// Depth-stencil images carry two aspects and Kniz treats them as one unit, so a
// region has to name an aspect the format actually has.
bool aspect_matches(const ImageFormat* format, VkImageAspectFlags aspect,
                    const char* what) {
  if (format->depth) {
    // A depth-only region of a combined depth-stencil format is legal and means
    // the depth half, which is the half this backend transfers.
    if ((aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) == 0) {
      KNIZ_ERR("%s needs a depth or stencil aspect and was given 0x%x", what,
               static_cast<unsigned>(aspect));
      return false;
    }
    return true;
  }
  if (aspect != VK_IMAGE_ASPECT_COLOR_BIT) {
    KNIZ_ERR("%s needs VK_IMAGE_ASPECT_COLOR_BIT and was given 0x%x", what,
             static_cast<unsigned>(aspect));
    return false;
  }
  return true;
}

// A view's format must be the image's own.
//
// Vulkan also allows an sRGB and its UNORM counterpart to be used this way, and
// that pair is genuinely tempting because OpenGL exposes sRGB-ness as a property
// of the texture object.  It is not a property that can be *changed* per view
// without glTextureView, though: an sRGB view of a UNORM texture would sample
// with the transfer function missing and produce plausible, wrong colour.  So the
// pair is refused along with everything else, and the refusal names the feature
// that would make it possible.
bool format_matches(VkFormat view_format, VkFormat image_format) {
  if (view_format == image_format)
    return true;
  KNIZ_ERR("image view format %d must equal the image's format %d: a view that "
           "reinterprets the format needs glTextureView, which is not implemented",
           static_cast<int>(view_format), static_cast<int>(image_format));
  return false;
}

uint32_t mip_dimension(uint32_t base, uint32_t level) {
  const uint32_t shifted = base >> level;
  return shifted > 0 ? shifted : 1;
}

}  // namespace

const ImageFormat* image_format(VkFormat format) {
  for (const ImageFormat& candidate : kFormats) {
    if (candidate.vk == format)
      return &candidate;
  }
  return nullptr;
}

uint64_t image_byte_size(VkFormat format, const VkExtent3D& extent,
                         uint32_t mip_levels, uint32_t array_layers) {
  const ImageFormat* info = image_format(format);
  if (info == nullptr)
    return 0;

  uint64_t total = 0;
  uint64_t width = extent.width;
  uint64_t height = extent.height;
  for (uint32_t level = 0; level < mip_levels; ++level) {
    total += width * height * info->bytes_per_pixel;
    width = width > 1 ? width / 2 : 1;
    height = height > 1 ? height / 2 : 1;
  }
  return total * array_layers;
}

bool image_create_storage(Image* image) {
  const ImageFormat* format = image_format(image->format);
  if (format == nullptr) {
    KNIZ_ERR("image format %d has no OpenGL counterpart",
             static_cast<int>(image->format));
    return false;
  }

  gl::f.BindTexture(gl::kTexture2D, image->gl_texture);

  // Immutable storage.  Its value here is not speed but that the level count is
  // fixed at creation: Vulkan promises the same thing, and glTexImage2D would
  // quietly reallocate the storage if a later operation disagreed about it.
  gl::f.TexStorage2D(gl::kTexture2D, static_cast<gl::Size>(image->mip_levels),
                     format->internal, static_cast<gl::Size>(image->extent.width),
                     static_cast<gl::Size>(image->extent.height));
  if (gl::report_errors("after glTexStorage2D"))
    return false;

  // A texture without a complete filter and mip range is *incomplete*, and an
  // incomplete texture samples as black instead of failing.  So this is not
  // cosmetic: it is the difference between "sampled nothing" and "sampled the
  // image".  Vulkan's filter and wrap state belongs to a sampler, which does not
  // exist yet; when it does, these become the defaults that a sampler overrides.
  gl::f.TexParameteri(gl::kTexture2D, gl::kTextureMinFilter, gl::kNearest);
  gl::f.TexParameteri(gl::kTexture2D, gl::kTextureMagFilter, gl::kNearest);
  gl::f.TexParameteri(gl::kTexture2D, gl::kTextureWrapS, gl::kClampToEdge);
  gl::f.TexParameteri(gl::kTexture2D, gl::kTextureWrapT, gl::kClampToEdge);
  gl::f.TexParameteri(gl::kTexture2D, gl::kTextureBaseLevel, 0);
  gl::f.TexParameteri(gl::kTexture2D, gl::kTextureMaxLevel,
                      static_cast<gl::Int>(image->mip_levels) - 1);
  if (gl::report_errors("after setting texture parameters"))
    return false;

  image->storage_allocated = true;
  return true;
}

void image_clear(Image* image, const VkImageSubresourceRange& range,
                 const uint8_t* clear_value) {
  const ImageFormat* format = image_format(image->format);
  if (format == nullptr || format->depth) {
    KNIZ_ERR("vkCmdClearColorImage was recorded against an image whose format %d "
             "is not a colour format",
             static_cast<int>(image->format));
    return;
  }
  if (range.baseArrayLayer != 0 || range.layerCount != 1) {
    KNIZ_ERR("vkCmdClearColorImage only supports one array layer for now");
    return;
  }

  gl::f.BindTexture(gl::kTexture2D, image->gl_texture);

  // The value arrives as the format's own component class, which for every
  // colour format Kniz supports is floating point, and GL converts floats into
  // whatever the internal format is.  Passing the bytes-through external format
  // instead would misread them -- an RGBA8 clear would be read as four bytes.
  for (uint32_t i = 0; i < range.levelCount; ++i) {
    const uint32_t level = range.baseMipLevel + i;
    if (level >= image->mip_levels) {
      KNIZ_ERR("vkCmdClearColorImage named mip level %u of a %u level image", level,
               image->mip_levels);
      return;
    }
    gl::f.ClearTexImage(image->gl_texture, static_cast<gl::Int>(level),
                        format->external, gl::kFloat, clear_value);
  }
  gl::report_errors("after glClearTexImage");
}

bool image_write(Buffer* buffer, Image* image, const VkBufferImageCopy& region) {
  const ImageFormat* format = image_format(image->format);
  if (format == nullptr) {
    KNIZ_ERR("vkCmdCopyBufferToImage targets an image whose format %d has no "
             "OpenGL counterpart",
             static_cast<int>(image->format));
    return false;
  }
  if (!aspect_matches(format, region.imageSubresource.aspectMask,
                      "vkCmdCopyBufferToImage")) {
    return false;
  }

  RegionGeometry geometry{};
  if (!region_geometry(image, format, region, &geometry, "vkCmdCopyBufferToImage"))
    return false;

  uint8_t* source =
      buffer_bytes(buffer, region.bufferOffset, geometry.span_bytes,
                   "vkCmdCopyBufferToImage");
  if (source == nullptr)
    return false;

  gl::f.BindTexture(gl::kTexture2D, image->gl_texture);

  // Rows in the buffer may be padded, which GL calls a row length.  Alignment
  // must be one byte or GL would insert padding of its own on top of Vulkan's
  // and read the wrong rows, and it matters for any format narrower than four
  // bytes per texel.
  gl::f.PixelStorei(gl::kUnpackAlignment, 1);
  gl::f.PixelStorei(gl::kUnpackRowLength,
                    static_cast<gl::Int>(geometry.row_length_texels));

  gl::f.TexSubImage2D(gl::kTexture2D, 0, static_cast<gl::Int>(region.imageOffset.x),
                      static_cast<gl::Int>(region.imageOffset.y),
                      static_cast<gl::Size>(region.imageExtent.width),
                      static_cast<gl::Size>(region.imageExtent.height),
                      format->external, format->type, source);

  // Leaving the row length set would silently reinterpret the next upload.
  gl::f.PixelStorei(gl::kUnpackRowLength, 0);

  return !gl::report_errors("after glTexSubImage2D");
}

bool image_read(Buffer* buffer, Image* image, const VkBufferImageCopy& region) {
  const ImageFormat* format = image_format(image->format);
  if (format == nullptr) {
    KNIZ_ERR("vkCmdCopyImageToBuffer reads an image whose format %d has no "
             "OpenGL counterpart",
             static_cast<int>(image->format));
    return false;
  }
  if (!aspect_matches(format, region.imageSubresource.aspectMask,
                      "vkCmdCopyImageToBuffer")) {
    return false;
  }

  RegionGeometry geometry{};
  if (!region_geometry(image, format, region, &geometry, "vkCmdCopyImageToBuffer"))
    return false;

  uint8_t* destination =
      buffer_bytes(buffer, region.bufferOffset, geometry.span_bytes,
                   "vkCmdCopyImageToBuffer");
  if (destination == nullptr)
    return false;

  gl::f.BindTexture(gl::kTexture2D, image->gl_texture);

  // glGetTexImage reads a whole level and has no notion of a sub-rectangle, so a
  // region less than the whole level goes through a temporary and is picked out
  // of it.  That costs a full level of host memory per copy and is the price of
  // not needing a framebuffer to read an image; docs/DESIGN.md records it.
  const uint64_t level_width = image->extent.width;
  const uint64_t level_bytes =
      level_width * image->extent.height * format->bytes_per_pixel;

  std::vector<uint8_t> level(static_cast<size_t>(level_bytes));
  gl::f.PixelStorei(gl::kPackAlignment, 1);
  gl::f.GetTexImage(gl::kTexture2D, 0, format->external, format->type, level.data());

  // report_errors drains the queue, so a failure has to be acted on here rather
  // than looked for again below: the data in `level` cannot be trusted.
  if (gl::report_errors("after glGetTexImage"))
    return false;

  const uint64_t row_pitch =
      static_cast<uint64_t>(geometry.row_length_texels) * format->bytes_per_pixel;
  const uint64_t copy_bytes =
      static_cast<uint64_t>(region.imageExtent.width) * format->bytes_per_pixel;

  for (uint32_t row = 0; row < region.imageExtent.height; ++row) {
    const uint64_t source_offset =
        ((static_cast<uint64_t>(region.imageOffset.y) + row) * level_width +
         region.imageOffset.x) *
        format->bytes_per_pixel;
    std::memcpy(destination + row * row_pitch, level.data() + source_offset,
                static_cast<size_t>(copy_bytes));
  }

  // The bytes now exist in two places that must not disagree: the host mirror,
  // which is what the application maps, and the GL buffer object, which a later
  // submission may read.  Submission copies the object back into the mirror for
  // every buffer the GPU may have written, so writing only the mirror here would
  // be undone before vkQueueSubmit returned.  Uploading the same span keeps the
  // two in step; the span's padding bytes come from the mirror's original
  // contents, so copying them changes nothing.
  if (buffer->gl_name != 0) {
    gl::f.BindBuffer(gl::kArrayBuffer, buffer->gl_name);
    gl::f.BufferSubData(gl::kArrayBuffer,
                        static_cast<gl::IntPtr>(region.bufferOffset),
                        gl::as_size_ptr(geometry.span_bytes), destination);
    gl::f.BindBuffer(gl::kArrayBuffer, 0);

    // uploaded_version is deliberately left alone.  If it already matched the
    // allocation's version, the whole buffer was current and still is; if it did
    // not, the next submission re-uploads the whole buffer, which is harmless and
    // keeps the invariant "the object is at least as new as the version" true.
  }
  return true;
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateImage(
    VkDevice deviceHandle, const VkImageCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkImage* pImage) {
  (void)pAllocator;
  if (pImage == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pImage = VK_NULL_HANDLE;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr || pCreateInfo == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (!image_gl_ready(device))
    return VK_ERROR_INITIALIZATION_FAILED;

  if (!gl::info().texture_capable) {
    KNIZ_ERR("images need an OpenGL 4.5 context and this one is %d.%d",
             gl::info().major, gl::info().minor);
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  // Linear tiling is refused rather than approximated, and the refusal is the
  // honest one rather than an omission.  A linear image's memory layout is
  // observable -- vkGetImageSubresourceLayout exists precisely to report it --
  // and an application may rely on it.  GL's texture storage has its own layout
  // and cannot be told to use another.  Optimal tiling promises nothing about
  // layout and is what applications actually use.
  if (pCreateInfo->tiling != VK_IMAGE_TILING_OPTIMAL) {
    KNIZ_ERR("only VK_IMAGE_TILING_OPTIMAL is implemented: a linear image's "
             "layout is visible to the application and OpenGL cannot provide it");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if (pCreateInfo->imageType != VK_IMAGE_TYPE_2D) {
    KNIZ_ERR("only VK_IMAGE_TYPE_2D images are implemented");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if (pCreateInfo->samples != VK_SAMPLE_COUNT_1_BIT) {
    KNIZ_ERR("multisampled images are not implemented (%u samples requested)",
             static_cast<unsigned>(pCreateInfo->samples));
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if ((pCreateInfo->flags & kUnimplementedImageFlags) != 0) {
    KNIZ_ERR("image creation flags 0x%x are not implemented",
             static_cast<unsigned>(pCreateInfo->flags & kUnimplementedImageFlags));
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if ((pCreateInfo->usage & kUnimplementedImageUsage) != 0) {
    KNIZ_ERR("the requested image usage 0x%x includes a feature that is not "
             "implemented (storage images and input attachments)",
             static_cast<unsigned>(pCreateInfo->usage & kUnimplementedImageUsage));
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if (pCreateInfo->extent.width == 0 || pCreateInfo->extent.height == 0 ||
      pCreateInfo->extent.depth != 1 || pCreateInfo->mipLevels == 0 ||
      pCreateInfo->arrayLayers == 0) {
    KNIZ_ERR("a 2D image needs non-zero width, height, mip levels and array "
             "layers, and a depth of one");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  const ImageFormat* format = image_format(pCreateInfo->format);
  if (format == nullptr) {
    KNIZ_ERR("image format %d has no OpenGL counterpart",
             static_cast<int>(pCreateInfo->format));
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  }

  // The GL size limit is the binding one, and asking for it is more accurate than
  // assuming anything: Vulkan's required minimum is 4096, which is lower than
  // every driver that provides the 4.5 context this path needs.
  gl::Int max_size = 0;
  gl::f.GetIntegerv(gl::kMaxTextureSize, &max_size);
  if (max_size > 0 && (pCreateInfo->extent.width > static_cast<uint32_t>(max_size) ||
                       pCreateInfo->extent.height > static_cast<uint32_t>(max_size))) {
    KNIZ_ERR("a %ux%u image exceeds the OpenGL limit of %d",
             pCreateInfo->extent.width, pCreateInfo->extent.height, max_size);
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  }

  Image* image = new (std::nothrow) Image();
  if (image == nullptr)
    return VK_ERROR_OUT_OF_HOST_MEMORY;

  image->device = device;
  image->type = pCreateInfo->imageType;
  image->format = pCreateInfo->format;
  image->extent = pCreateInfo->extent;
  image->mip_levels = pCreateInfo->mipLevels;
  image->array_layers = pCreateInfo->arrayLayers;
  image->usage = pCreateInfo->usage;
  image->layout = pCreateInfo->initialLayout;

  gl::f.GenTextures(1, &image->gl_texture);
  if (image->gl_texture == 0) {
    KNIZ_ERR("glGenTextures failed");
    delete image;
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  }

  *pImage = to_handle(image);
  KNIZ_TRACE("image %p: %ux%u format %d, %u level(s), %u layer(s), GL texture %u",
             static_cast<void*>(*pImage), image->extent.width, image->extent.height,
             static_cast<int>(image->format), image->mip_levels, image->array_layers,
             image->gl_texture);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyImage(VkDevice deviceHandle, VkImage imageHandle,
                                        const VkAllocationCallbacks* pAllocator) {
  (void)pAllocator;
  Device* device = from_handle(deviceHandle);
  Image* image = from_handle(imageHandle);
  if (device == nullptr || image == nullptr)
    return;
  if (!image_gl_ready(device))
    return;

  if (image->gl_texture != 0)
    gl::f.DeleteTextures(1, &image->gl_texture);

  if (image->memory != nullptr && image->memory->images_bound > 0)
    image->memory->images_bound -= 1;

  delete image;
}

VKAPI_ATTR void VKAPI_CALL GetImageMemoryRequirements(
    VkDevice deviceHandle, VkImage imageHandle,
    VkMemoryRequirements* pMemoryRequirements) {
  if (pMemoryRequirements == nullptr)
    return;
  *pMemoryRequirements = VkMemoryRequirements{};

  Device* device = from_handle(deviceHandle);
  Image* image = from_handle(imageHandle);
  if (device == nullptr || image == nullptr || image->device != device) {
    KNIZ_ERR("%p is not an image of this device", static_cast<void*>(imageHandle));
    return;
  }

  pMemoryRequirements->size =
      image_byte_size(image->format, image->extent, image->mip_levels,
                      image->array_layers);
  pMemoryRequirements->alignment = kAllocationAlignment;

  // Both memory types, as for buffers: the allocation does not hold the image's
  // pixels at all, so the distinction the application is making does not arise.
  pMemoryRequirements->memoryTypeBits = 0x3;
}

VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory(VkDevice deviceHandle,
                                               VkImage imageHandle,
                                               VkDeviceMemory memoryHandle,
                                               VkDeviceSize memoryOffset) {
  Device* device = from_handle(deviceHandle);
  Image* image = from_handle(imageHandle);
  DeviceMemory* memory = from_handle(memoryHandle);
  if (device == nullptr || !image_gl_ready(device))
    return VK_ERROR_INITIALIZATION_FAILED;
  if (image == nullptr || image->device != device) {
    KNIZ_ERR("%p is not an image of this device", static_cast<void*>(imageHandle));
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (memory == nullptr || memory->device != device) {
    KNIZ_ERR("%p is not an allocation of this device",
             static_cast<void*>(memoryHandle));
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (image->memory != nullptr) {
    KNIZ_ERR("image %p already has memory bound", static_cast<void*>(imageHandle));
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  const uint64_t size = image_byte_size(image->format, image->extent,
                                        image->mip_levels, image->array_layers);
  if (memoryOffset % kAllocationAlignment != 0) {
    KNIZ_ERR("image offset %llu is not a multiple of the reported alignment %llu",
             static_cast<unsigned long long>(memoryOffset),
             static_cast<unsigned long long>(kAllocationAlignment));
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (memoryOffset + size > memory->size) {
    KNIZ_ERR("a %llu byte image at offset %llu does not fit in a %llu byte "
             "allocation",
             static_cast<unsigned long long>(size),
             static_cast<unsigned long long>(memoryOffset),
             static_cast<unsigned long long>(memory->size));
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  }

  if (!image_create_storage(image))
    return VK_ERROR_INITIALIZATION_FAILED;

  image->memory = memory;
  image->memory_offset = memoryOffset;
  memory->images_bound += 1;

  KNIZ_TRACE("image %p bound at offset %llu of a %llu byte allocation",
             static_cast<void*>(imageHandle),
             static_cast<unsigned long long>(memoryOffset),
             static_cast<unsigned long long>(memory->size));
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL GetImageSubresourceLayout(
    VkDevice deviceHandle, VkImage imageHandle,
    const VkImageSubresource* pSubresource, VkSubresourceLayout* pLayout) {
  if (pLayout == nullptr)
    return;
  *pLayout = VkSubresourceLayout{};

  Device* device = from_handle(deviceHandle);
  Image* image = from_handle(imageHandle);
  if (device == nullptr || image == nullptr || pSubresource == nullptr)
    return;

  const ImageFormat* format = image_format(image->format);
  if (format == nullptr)
    return;

  // Unreachable in practice, because vkCreateImage already refused linear tiling,
  // and the layout is only defined for a linear image.  Answering with a tightly
  // packed one keeps a caller that asks anyway from reading uninitialised memory;
  // the warning is there so that nobody mistakes the number for the truth.
  const uint32_t level = pSubresource->mipLevel;
  const uint64_t width = mip_dimension(image->extent.width, level);
  const uint64_t height = mip_dimension(image->extent.height, level);

  pLayout->offset = 0;
  pLayout->rowPitch = width * format->bytes_per_pixel;
  pLayout->arrayPitch = pLayout->rowPitch * height;
  pLayout->depthPitch = pLayout->arrayPitch;
  pLayout->size = pLayout->arrayPitch * image->array_layers;

  KNIZ_WARN("vkGetImageSubresourceLayout is answering with a tightly packed "
            "layout, which is not how OpenGL stores the image; a linear image is "
            "the only kind whose layout Vulkan defines, and Kniz does not create "
            "them");
}

VKAPI_ATTR VkResult VKAPI_CALL CreateImageView(
    VkDevice deviceHandle, const VkImageViewCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkImageView* pImageView) {
  (void)pAllocator;
  if (pImageView == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pImageView = VK_NULL_HANDLE;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr || pCreateInfo == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  Image* image = from_handle(pCreateInfo->image);
  if (image == nullptr || image->device != device) {
    KNIZ_ERR("%p is not an image of this device",
             static_cast<void*>(pCreateInfo->image));
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  if (pCreateInfo->viewType != VK_IMAGE_VIEW_TYPE_2D &&
      pCreateInfo->viewType != VK_IMAGE_VIEW_TYPE_2D_ARRAY) {
    KNIZ_ERR("only 2D image views are implemented");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  const ImageFormat* view_format = image_format(pCreateInfo->format);
  if (view_format == nullptr) {
    KNIZ_ERR("image view format %d has no OpenGL counterpart",
             static_cast<int>(pCreateInfo->format));
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  }
  if (!format_matches(pCreateInfo->format, image->format))
    return VK_ERROR_FORMAT_NOT_SUPPORTED;

  if (pCreateInfo->components.r != VK_COMPONENT_SWIZZLE_IDENTITY ||
      pCreateInfo->components.g != VK_COMPONENT_SWIZZLE_IDENTITY ||
      pCreateInfo->components.b != VK_COMPONENT_SWIZZLE_IDENTITY ||
      pCreateInfo->components.a != VK_COMPONENT_SWIZZLE_IDENTITY) {
    KNIZ_ERR("image view component swizzles are not implemented");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  const VkImageSubresourceRange& range = pCreateInfo->subresourceRange;
  if (range.levelCount == 0 || range.layerCount == 0 ||
      range.baseMipLevel + range.levelCount > image->mip_levels ||
      range.baseArrayLayer + range.layerCount > image->array_layers) {
    KNIZ_ERR("image view subresource range [%u, +%u) x [%u, +%u) does not fit in "
             "%u level(s) and %u layer(s)",
             range.baseMipLevel, range.levelCount, range.baseArrayLayer,
             range.layerCount, image->mip_levels, image->array_layers);
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (!aspect_matches(image_format(image->format), range.aspectMask,
                      "vkCreateImageView"))
    return VK_ERROR_INITIALIZATION_FAILED;

  ImageView* view = new (std::nothrow) ImageView();
  if (view == nullptr)
    return VK_ERROR_OUT_OF_HOST_MEMORY;

  view->device = device;
  view->image = image;
  view->view_type = pCreateInfo->viewType;
  view->format = pCreateInfo->format;
  view->range = range;

  *pImageView = to_handle(view);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyImageView(
    VkDevice deviceHandle, VkImageView imageViewHandle,
    const VkAllocationCallbacks* pAllocator) {
  (void)pAllocator;
  Device* device = from_handle(deviceHandle);
  ImageView* view = from_handle(imageViewHandle);
  if (device == nullptr || view == nullptr)
    return;
  delete view;
}

// ---------------------------------------------------------------------------
// Samplers
// ---------------------------------------------------------------------------

namespace {

// Vulkan's address modes on GL's wrap parameters.  VK_SAMPLER_ADDRESS_MODE_*
// and GL's wrap modes are the same four behaviours; the fifth, mirror clamp to
// edge, is a Vulkan extension this driver does not report.
bool gl_wrap_mode(VkSamplerAddressMode mode, gl::Int* out) {
  switch (mode) {
    case VK_SAMPLER_ADDRESS_MODE_REPEAT:
      *out = gl::kRepeat;
      return true;
    case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:
      *out = gl::kMirroredRepeat;
      return true;
    case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:
      *out = gl::kClampToEdge;
      return true;
    case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:
      *out = gl::kClampToBorder;
      return true;
    default:
      return false;
  }
}

// GL has one minification filter that names the mipmap mode and one that does
// not, while Vulkan splits the same information across minFilter and mipmapMode.
// All four combinations are expressible; this is the table.
gl::Int gl_min_filter(VkFilter filter, VkSamplerMipmapMode mipmap) {
  if (filter == VK_FILTER_NEAREST)
    return mipmap == VK_SAMPLER_MIPMAP_MODE_NEAREST ? gl::kNearestMipmapNearest
                                                   : gl::kNearestMipmapLinear;
  return mipmap == VK_SAMPLER_MIPMAP_MODE_NEAREST ? gl::kLinearMipmapNearest
                                                 : gl::kLinearMipmapLinear;
}

void border_colour(VkBorderColor colour, gl::Float out[4]) {
  out[0] = 0.0f;
  out[1] = 0.0f;
  out[2] = 0.0f;
  out[3] = 0.0f;
  switch (colour) {
    case VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK:
    case VK_BORDER_COLOR_INT_OPAQUE_BLACK:
      out[3] = 1.0f;
      break;
    case VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE:
    case VK_BORDER_COLOR_INT_OPAQUE_WHITE:
      out[0] = out[1] = out[2] = out[3] = 1.0f;
      break;
    default:
      // The two transparent blacks are already in `out`.
      break;
  }
}

}  // namespace

VKAPI_ATTR VkResult VKAPI_CALL CreateSampler(
    VkDevice deviceHandle, const VkSamplerCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkSampler* pSampler) {
  (void)pAllocator;
  if (pSampler == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pSampler = VK_NULL_HANDLE;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr || pCreateInfo == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  if (!image_gl_ready(device))
    return VK_ERROR_INITIALIZATION_FAILED;

  if (!device->instance->gl.sampler_capable) {
    KNIZ_ERR("samplers need GL sampler objects, which this OpenGL context does "
             "not have");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  if (pCreateInfo->flags != 0) {
    KNIZ_ERR("sampler flags 0x%x are not implemented",
             static_cast<unsigned>(pCreateInfo->flags));
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if (pCreateInfo->unnormalizedCoordinates != VK_FALSE) {
    KNIZ_ERR("unnormalized sampler coordinates are not implemented");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if (pCreateInfo->compareEnable != VK_FALSE) {
    // Comparison samplers need GL_TEXTURE_COMPARE_MODE and a shadow sampler in
    // the shader, which is a different translation path than plain sampling.
    KNIZ_ERR("comparison samplers are not implemented; Kniz samples depth "
             "images as plain values");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  if (pCreateInfo->anisotropyEnable != VK_FALSE) {
    // The driver reports samplerAnisotropy = VK_FALSE, so an application that
    // asks for it is going beyond what was offered; the extension is not
    // resolved either, so there is nothing to enable.
    KNIZ_ERR("anisotropic filtering is not implemented");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  gl::Int wrap_u = 0;
  gl::Int wrap_v = 0;
  if (!gl_wrap_mode(pCreateInfo->addressModeU, &wrap_u) ||
      !gl_wrap_mode(pCreateInfo->addressModeV, &wrap_v)) {
    KNIZ_ERR("sampler address mode %d/%d is not one OpenGL has",
             static_cast<int>(pCreateInfo->addressModeU),
             static_cast<int>(pCreateInfo->addressModeV));
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  // addressModeW is not checked: Kniz samples 2D images, where it has no
  // effect, and refusing a value nothing reads would refuse legal applications.

  Sampler* sampler = new (std::nothrow) Sampler();
  if (sampler == nullptr)
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  sampler->device = device;

  gl::f.GenSamplers(1, &sampler->gl_sampler);
  if (sampler->gl_sampler == 0) {
    KNIZ_ERR("glGenSamplers failed");
    delete sampler;
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  }

  gl::f.SamplerParameteri(sampler->gl_sampler, gl::kTextureMagFilter,
                          pCreateInfo->magFilter == VK_FILTER_NEAREST
                              ? gl::kNearest
                              : gl::kLinear);
  gl::f.SamplerParameteri(sampler->gl_sampler, gl::kTextureMinFilter,
                          gl_min_filter(pCreateInfo->minFilter,
                                        pCreateInfo->mipmapMode));
  gl::f.SamplerParameteri(sampler->gl_sampler, gl::kTextureWrapS, wrap_u);
  gl::f.SamplerParameteri(sampler->gl_sampler, gl::kTextureWrapT, wrap_v);
  // A 2D image has no W coordinate, so the value is only set to keep the
  // sampler object from carrying a driver default that a 3D-capable path would
  // later pick up.
  gl::f.SamplerParameteri(sampler->gl_sampler, gl::kTextureWrapR, gl::kClampToEdge);

  gl::Float border[4];
  border_colour(pCreateInfo->borderColor, border);
  gl::f.SamplerParameterfv(sampler->gl_sampler, gl::kTextureBorderColor, border);

  gl::f.SamplerParameterf(sampler->gl_sampler, gl::kTextureMinLod,
                          pCreateInfo->minLod);
  gl::f.SamplerParameterf(sampler->gl_sampler, gl::kTextureMaxLod,
                          pCreateInfo->maxLod);
  gl::f.SamplerParameterf(sampler->gl_sampler, gl::kTextureLodBias,
                          pCreateInfo->mipLodBias);

  if (gl::report_errors("after creating a sampler")) {
    gl::f.DeleteSamplers(1, &sampler->gl_sampler);
    delete sampler;
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  *pSampler = to_handle(sampler);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroySampler(
    VkDevice deviceHandle, VkSampler samplerHandle,
    const VkAllocationCallbacks* pAllocator) {
  (void)pAllocator;
  Device* device = from_handle(deviceHandle);
  Sampler* sampler = from_handle(samplerHandle);
  if (device == nullptr || sampler == nullptr)
    return;
  if (sampler->gl_sampler != 0 && device->instance->gl_ready &&
      gl::make_current())
    gl::f.DeleteSamplers(1, &sampler->gl_sampler);
  delete sampler;
}

}  // namespace kniz
