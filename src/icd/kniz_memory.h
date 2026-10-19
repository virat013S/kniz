#pragma once

// Memory, buffers and descriptors.
//
// The part of this that is not a translation is the data movement.  Vulkan
// applications hand Kniz a pointer into a VkDeviceMemory and expect the GPU to
// see what they wrote; GL has no such object, so Kniz owns the allocation and
// copies in and out of it around every submission.  See docs/DESIGN.md.

#include "kniz_objects.h"

namespace kniz {

// The alignment of every allocation's host block, and the value reported as a
// buffer's and an image's `alignment`.
//
// Sharing one number is what makes memory offsets safe to trust: the block is
// aligned to this, so an offset that is a multiple of it -- which is all the
// entry points accept -- leaves a pointer with the same alignment.  Vulkan
// requires the reported value to be a power of two, and a host allocator
// requires it not to exceed the ABI's maximum, which 256 does not.
inline constexpr uint64_t kAllocationAlignment = 256;

// Move application writes into the GL buffer objects.  Called by submission
// before replaying, with the GL context current.
void sync_buffers_to_gl(Device* device);

// Move GPU results back into the application's host memory.  Called after
// replay, once the GL commands have completed.
void sync_buffers_from_gl(Device* device);

}  // namespace kniz
