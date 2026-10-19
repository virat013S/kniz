#pragma once

// Drawing: render passes, framebuffers, vertex buffers and the draw itself.
//
// This is where the translation stops being a translation.  Everything before
// M2b had a Vulkan construct that mapped onto one GL construct: a buffer is a
// byte array on both sides, an image is a texture on both sides.  Drawing does
// not work that way.  A Vulkan pipeline is an immutable object holding dozens of
// pieces of state that GL keeps in no single place -- some in the program, some
// in the vertex array object, some in the context itself -- and a render pass is
// a description of work GL has no concept of.
//
// So the recorded command stream stores the resolved state and the replay path
// applies it.  That is the same trade ADR-1 already made, one level down.
//
// Two pieces of this are worth stating before the code:
//
//   * No y flip is applied anywhere -- not to the vertex data, not to the
//     viewport, not to the scissor, not to the winding order.  GL and Vulkan both
//     put a vertex at NDC y = -1 in row 0 of the image, so the two coordinate
//     systems agree numerically even though one calls that row "bottom" and the
//     other "top".  Every consequence of getting this wrong is silent, so it is
//     argued once in docs/DESIGN.md rather than rediscovered per site.
//
//   * A render pass's load operation is the one part of it that does something:
//     it becomes a clear of the corresponding framebuffer attachment.  Store
//     operations have no equivalent because GL always keeps the result, which is
//     a superset of what any store operation allows.

#include "kniz_objects.h"

namespace kniz {

// The replay half, called from kniz_command.cpp with a GL context current.
void render_begin_pass(const RecordedCommand& command);
void render_end_pass(const RecordedCommand& command);
void render_draw(const RecordedCommand& command);

}  // namespace kniz
