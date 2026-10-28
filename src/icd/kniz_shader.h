#pragma once

// SPIR-V -> GLSL.
//
// This is the piece that could not be avoided: Vulkan hands Kniz SPIR-V, and
// OpenGL wants GLSL, so something has to lower one into the other.  SPIRV-Cross
// does the lowering; what belongs to Kniz is the resource mapping, which is the
// part that actually has to be right for a dispatch or a draw to address the
// buffers the application meant.

#include "kniz_objects.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kniz {

struct ShaderTranslation {
  bool ok = false;
  std::string glsl;

  // The descriptors this stage actually declared, resolved to their GL binding
  // points.  Empty for a stage that reads none, which is the common case for a
  // fragment shader that only writes a colour out.
  std::vector<GlBinding> used;

  // Human readable reason, set whenever ok is false.
  std::string error;
};

// Translate one shader stage from SPIR-V into GLSL 4.50, rewriting every
// resource onto the OpenGL binding point the pipeline layout assigned it.
//
// `entry_point` is the SPIR-V entry point to translate, i.e. what the
// application passed as VkPipelineShaderStageCreateInfo::pName, and `stage` is
// which stage it is: the same SPIR-V can hold several entry points, and the
// execution model is what says which one a given stage means.
//
// The layout is passed in rather than the mapping being derived here because GL
// binding points are flat while Vulkan addresses descriptors by (set, binding):
// two resources in different sets may legally share a binding number, and the
// only way to give them distinct GL points is to know the whole pipeline layout.
// Deriving the mapping in one place -- pipeline layout creation -- is what keeps
// the generated GLSL and the replay path from disagreeing.
//
// Three constructs have no GLSL equivalent and are lowered here rather than in
// the replay path, because the shader is the only place that knows they exist:
//
//   * Push constants become a uniform block at `layout.push_constant_gl_index`,
//     which the replay uploads to as the recorded commands replay.
//   * Separate images and samplers become combined image samplers, because GLSL
//     has one sampler type that carries both.  The combined sampler lives at the
//     *image's* texture unit, so the replay binds the sampler object to that same
//     unit; a shader that reads one image through two different samplers cannot
//     be expressed that way and is refused.
//   * A `texelFetch` on a separate image needs a sampler argument in GLSL that
//     Vulkan does not require, so SPIRV-Cross injects a dummy one at
//     `layout.texture_count`, the first unit past every descriptor.
//
// The clip space fixup is applied for vertex stages and must not be applied for
// any other: GL and Vulkan disagree about the depth range ([-w, w] against
// [0, w]) while agreeing about the y direction, which is the opposite of what
// the usual summary of the difference says.  See docs/DESIGN.md.
ShaderTranslation translate_shader(const std::vector<uint32_t>& spirv,
                                   const std::string& entry_point,
                                   VkShaderStageFlagBits stage,
                                   const PipelineLayout& layout);

}  // namespace kniz
