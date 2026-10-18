#pragma once

// Memory, buffers and descriptors.
//
// The part of this that is not a translation is the data movement.  Vulkan
// applications hand Kniz a pointer into a VkDeviceMemory and expect the GPU to
// see what they wrote; GL has no such object, so Kniz owns the allocation and
// copies in and out of it around every submission.  See docs/DESIGN.md.

#include "kniz_objects.h"

namespace kniz {

// Move application writes into the GL buffer objects.  Called by submission
// before replaying, with the GL context current.
void sync_buffers_to_gl(Device* device);

// Move GPU results back into the application's host memory.  Called after
// replay, once the GL commands have completed.
void sync_buffers_from_gl(Device* device);

}  // namespace kniz
