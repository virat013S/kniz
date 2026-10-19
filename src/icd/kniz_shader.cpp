#include "kniz_shader.h"

#include "kniz_common.h"

// SPIRV-Cross's own build keeps its headers at the repository root, while an
// installed copy places them under include/spirv_cross.  Handle both.
#if __has_include(<spirv_cross/spirv_glsl.hpp>)
#include <spirv_cross/spirv_glsl.hpp>
#else
#include <spirv_glsl.hpp>
#endif

#include <exception>

namespace kniz {
namespace {

bool lookup_gl_index(const std::vector<GlBinding>& bindings, uint32_t set,
                     uint32_t binding, uint32_t* gl_index) {
  for (const GlBinding& candidate : bindings) {
    if (candidate.set == set && candidate.binding == binding) {
      *gl_index = candidate.gl_index;
      return true;
    }
  }
  return false;
}

// Which unimplemented feature, if any, this shader needs.  Checked before
// translation because emitting GLSL that quietly drops a binding is worse than
// refusing: the application would get plausible output computed from the wrong
// data.
std::string unsupported_feature(const spirv_cross::ShaderResources& resources) {
  if (!resources.push_constant_buffers.empty())
    return "push constants";
  if (!resources.uniform_buffers.empty())
    return "uniform buffers";
  if (!resources.storage_images.empty())
    return "storage images";
  if (!resources.sampled_images.empty() || !resources.separate_images.empty() ||
      !resources.separate_samplers.empty())
    return "textures or samplers";
  if (!resources.atomic_counters.empty())
    return "atomic counters";
  if (!resources.acceleration_structures.empty())
    return "acceleration structures";
  return {};
}

const char* stage_name(VkShaderStageFlagBits stage) {
  switch (stage) {
    case VK_SHADER_STAGE_VERTEX_BIT:
      return "vertex";
    case VK_SHADER_STAGE_FRAGMENT_BIT:
      return "fragment";
    case VK_SHADER_STAGE_COMPUTE_BIT:
      return "compute";
    default:
      return "unknown";
  }
}

bool execution_model(VkShaderStageFlagBits stage, spv::ExecutionModel* model) {
  switch (stage) {
    case VK_SHADER_STAGE_VERTEX_BIT:
      *model = spv::ExecutionModelVertex;
      return true;
    case VK_SHADER_STAGE_FRAGMENT_BIT:
      *model = spv::ExecutionModelFragment;
      return true;
    case VK_SHADER_STAGE_COMPUTE_BIT:
      *model = spv::ExecutionModelGLCompute;
      return true;
    default:
      return false;
  }
}

}  // namespace

ShaderTranslation translate_shader(const std::vector<uint32_t>& spirv,
                                   const std::string& entry_point,
                                   VkShaderStageFlagBits stage,
                                   const std::vector<GlBinding>& bindings) {
  ShaderTranslation result;

  if (spirv.empty()) {
    result.error = "the shader module holds no SPIR-V";
    return result;
  }

  spv::ExecutionModel model = spv::ExecutionModelMax;
  if (!execution_model(stage, &model)) {
    result.error = std::string("a ") + stage_name(stage) +
                   " stage is not one of the stages Kniz translates";
    return result;
  }

  try {
    spirv_cross::CompilerGLSL compiler(spirv);

    // Always named, even for "main".  The entry point's execution model is what
    // tells SPIRV-Cross which stage it is emitting, and a module can hold more
    // than one entry point, so leaving it to the default is a guess.
    compiler.set_entry_point(entry_point.empty() ? "main" : entry_point, model);

    spirv_cross::CompilerGLSL::Options options;
    // 4.50 is the first version in which compute shaders, shader storage blocks,
    // the std430 layout and explicit varying locations are all core, so no
    // extension pragmas are needed.
    options.version = 450;
    options.es = false;
    options.vulkan_semantics = false;

    // Vulkan's clip space runs z from 0 to w and GL's from -w to w, so a vertex
    // shader written for Vulkan would place its geometry in the near half of GL's
    // depth range -- and geometry at z = w would sit exactly on GL's far plane,
    // where clipping is at best undefined.  SPIRV-Cross rewrites
    // gl_Position.z = 2 * z - w to fix that.
    //
    // The y axis is deliberately *not* flipped, and that is not an oversight: GL
    // and Vulkan both map a vertex at y = -1 to row 0 of the image, so flipping
    // would introduce the very mismatch this is meant to avoid.  See the note in
    // docs/DESIGN.md.
    options.vertex.fixup_clipspace = true;
    options.vertex.flip_vert_y = false;

    compiler.set_common_options(options);

    const spirv_cross::ShaderResources resources =
        compiler.get_shader_resources();

    if (std::string missing = unsupported_feature(resources);
        !missing.empty()) {
      result.error = "the " + std::string(stage_name(stage)) +
                     " shader uses " + missing +
                     ", which Kniz does not implement yet; only shader storage "
                     "buffers and vertex attributes are supported so far";
      return result;
    }

    // Rewrite every storage buffer onto the GL binding point the pipeline
    // layout assigned it.
    for (const spirv_cross::Resource& resource : resources.storage_buffers) {
      const uint32_t set =
          compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
      const uint32_t binding =
          compiler.get_decoration(resource.id, spv::DecorationBinding);

      uint32_t gl_index = 0;
      if (!lookup_gl_index(bindings, set, binding, &gl_index)) {
        // A resource the pipeline layout does not declare.  Sticking it at
        // index 0 would bind it over some other resource, so refuse.
        result.error = "shader declares a storage buffer at set " +
                       std::to_string(set) + " binding " +
                       std::to_string(binding) +
                       ", but the pipeline layout has no such binding";
        return result;
      }

      compiler.set_decoration(resource.id, spv::DecorationBinding, gl_index);
      // GLSL has no descriptor sets.  Leaving the decoration in place makes
      // SPIRV-Cross emit layout(set = N, ...), which will not compile.
      compiler.set_decoration(resource.id, spv::DecorationDescriptorSet, 0);
    }

    result.glsl = compiler.compile();
    if (result.glsl.empty()) {
      result.error = "SPIRV-Cross produced an empty shader";
      return result;
    }

    result.ok = true;
  } catch (const std::exception& error) {
    result.error = error.what();
  }

  return result;
}

}  // namespace kniz
