#pragma once

// Images: Vulkan images on OpenGL textures.
//
// The file exists because of one asymmetry with buffers.  A buffer's contents
// have a single representation -- a byte array -- so Kniz can keep a host copy
// and shuffle it in and out of GL.  An image has two: the layout Vulkan
// describes, and whatever GL actually stores, which is opaque and
// implementation-defined.  There is no correspondence between them, so an image
// is backed only by its GL texture and every read and write goes through a
// command.  See docs/DESIGN.md for what that costs.

#include "kniz_objects.h"

namespace kniz {

// Image creation flags and usage bits that would be a lie rather than a
// simplification.  Everything else -- cube compatibility, mutable format, and so
// on -- costs nothing yet because nothing acts on it.
//
// Shared with the instance-level vkGetPhysicalDeviceImageFormatProperties, so
// that "can I do this" and "do it" cannot drift apart; the two used to be
// answered in different files, which is exactly how that goes wrong.
inline constexpr VkImageCreateFlags kUnimplementedImageFlags =
    VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
    VK_IMAGE_CREATE_SPARSE_ALIASED_BIT |
    VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;

inline constexpr VkImageUsageFlags kUnimplementedImageUsage =
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;

// What a depth or depth/stencil image may be used for.  Sampling a depth texture
// is legal Vulkan and would work in GL, but nothing in this driver has been
// tested through that path -- the sampler lowering, the shader's declared type and
// the comparison are all different for depth -- so it is refused rather than
// offered and left to produce a plausible, wrong value.
inline constexpr VkImageUsageFlags kDepthImageUsage =
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

// The GL representation of one Vulkan format.
struct ImageFormat {
  VkFormat vk = VK_FORMAT_UNDEFINED;

  // Sized internal format, for glTexStorage2D.
  gl::Enum internal = 0;

  // Pixel format and component type for transfers.  Used on both sides of a copy
  // so that bytes pass through unchanged rather than being converted twice.
  gl::Enum external = 0;
  gl::Enum type = 0;

  uint32_t bytes_per_pixel = 0;

  // Depth or depth/stencil.  Such a format cannot be the target of
  // vkCmdClearColorImage, and is not a colour attachment.
  bool depth = false;

  // A depth/stencil format has both aspects, and GL attaches it at
  // GL_DEPTH_STENCIL_ATTACHMENT rather than GL_DEPTH_ATTACHMENT.  The two
  // attachment points are not interchangeable: a packed format at the depth-only
  // point makes the framebuffer incomplete.
  bool stencil = false;
};

// Null when the format is not one Kniz can present as a texture.  Callers report
// the numeric value, since no name is available.
const ImageFormat* image_format(VkFormat format);

// What vkGetPhysicalDeviceFormatProperties reports for a format: the same
// features vkCreateImage and the render pass actually accept, derived from the
// table above so that the answer and the behaviour cannot drift apart.
//
// This matters more than it looks.  Applications ask before they act, and a
// format reported as having no features makes them take a fallback path -- vkcube
// asks whether R8G8B8A8_SRGB can be sampled, and when told no, it builds its
// texture as a linear image instead and never creates the optimal one at all.
// Answering "nothing is supported" is not conservative, it is wrong.
VkFormatFeatureFlags format_features(VkFormat format);

// Bytes a tightly packed image of this description would occupy.  Not a GL
// concept: GL allocates whatever its own layout needs, and this number exists
// only to answer vkGetImageMemoryRequirements, whose contract Vulkan defines in
// terms of a layout Kniz does not use.
uint64_t image_byte_size(VkFormat format, const VkExtent3D& extent,
                         uint32_t mip_levels, uint32_t array_layers);

// Creates the GL texture storage.  Called from vkBindImageMemory, which is where
// Vulkan makes an image's format and extent immutable.  Reports its own errors.
bool image_create_storage(Image* image);

// The operations replay performs on images, all with a current GL context.
//
// image_clear takes the caller's VkClearColorValue copied to bytes, because
// which member of that union is meaningful depends on the image's format.
void image_clear(Image* image, const VkImageSubresourceRange& range,
                 const uint8_t* clear_value);

// The two directions of vkCmdCopyBufferToImage / vkCmdCopyImageToBuffer.  The
// buffer is reached through its host mirror, which submission has already made
// current.
bool image_write(Buffer* buffer, Image* image, const VkBufferImageCopy& region);
bool image_read(Buffer* buffer, Image* image, const VkBufferImageCopy& region);

}  // namespace kniz
