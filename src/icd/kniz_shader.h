#pragma once

// SPIR-V -> GLSL.
//
// This is the piece that could not be avoided: Vulkan hands Kniz SPIR-V, and
// OpenGL wants GLSL, so something has to lower one into the other.  SPIRV-Cross
// does the lowering; what belongs to Kniz is the resource mapping, which is the
// part that actually has to be right for a dispatch to address the buffers the
// application meant.

#include "kniz_objects.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kniz {

struct ShaderTranslation {
  bool ok = false;
  std::string glsl;

  // Human readable reason, set whenever ok is false.
  std::string error;
};

// Translate one shader stage from SPIR-V into GLSL 4.50, rewriting every shader
// storage buffer onto the OpenGL binding point recorded in `bindings`.
//
// `entry_point` is the SPIR-V entry point to translate, i.e. what the
// application passed as VkPipelineShaderStageCreateInfo::pName, and `stage` is
// which stage it is: the same SPIR-V can hold several entry points, and the
// execution model is what says which one a given stage means.
//
// The mapping is passed in rather than derived here because GL binding points
// are flat while Vulkan addresses descriptors by (set, binding): two resources
// in different sets may legally share a binding number, and the only way to
// give them distinct GL points is to know the whole pipeline layout.  Deriving
// the mapping in one place -- pipeline creation -- is what keeps the generated
// GLSL and the replay path from disagreeing.
//
// The clip space fixup is applied for vertex stages and must not be applied for
// any other: GL and Vulkan disagree about the depth range ([-w, w] against
// [0, w]) while agreeing about the y direction, which is the opposite of what
// the usual summary of the difference says.  See docs/DESIGN.md.
ShaderTranslation translate_shader(const std::vector<uint32_t>& spirv,
                                   const std::string& entry_point,
                                   VkShaderStageFlagBits stage,
                                   const std::vector<GlBinding>& bindings);

}  // namespace kniz
