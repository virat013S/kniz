#pragma once

// Descriptors, and the two things a recorded command needs from them.
//
// Creating descriptor set layouts and pipeline layouts, and computing the flat
// OpenGL binding point every descriptor they declare ends up at, all live in one
// translation unit because they are one decision: the mapping is a property of
// the pipeline layout and of nothing else, so the pipeline that generates the
// GLSL and the replay path that binds the buffers read the same table.
//
// Binding a descriptor to GL is identical whether the command that needs it is a
// dispatch or a draw, so that code is here too rather than duplicated.  It used
// to live in the dispatch path, which meant graphics pipelines with descriptors
// silently bound nothing at all.

#include "kniz_objects.h"

namespace kniz {

// Bind every descriptor the pipeline's layout declares, taking the values from
// `sets`, which is indexed by set number.  `what` names the caller so the error
// messages say which command is refusing.  Returns false if a descriptor is
// missing or cannot be represented in GL, in which case the caller must not
// issue the draw or dispatch: doing so would read whatever the last command
// left bound, which is exactly the kind of silent wrongness the rest of the
// driver refuses rather than produces.
bool bind_descriptor_sets(const Pipeline* pipeline,
                          const std::vector<DescriptorSet*>& sets,
                          const char* what);

// Upload `size` bytes of push constants to the device's push constant buffer and
// bind it at the binding point the layout reserved for them.
bool apply_push_constants(Device* device, const PipelineLayout* layout,
                          uint32_t offset, const void* bytes, uint32_t size);

// Fill the device's push constant buffer with zeroes.  Called once per
// submission: Vulkan leaves push constants undefined at the start of a command
// buffer, and leaving whatever the previous submission wrote in GL's binding
// would make a bug in an application look like a bug in Kniz.
bool reset_push_constants(Device* device);

// Create and release the device's push constant buffer.
bool create_push_constant_buffer(Device* device);
void destroy_push_constant_buffer(Device* device);

}  // namespace kniz
