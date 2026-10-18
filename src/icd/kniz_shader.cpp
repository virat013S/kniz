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

}  // namespace

ShaderTranslation translate_compute(const std::vector<uint32_t>& spirv,
                                    const std::string& entry_point,
                                    const std::vector<GlBinding>& bindings) {
  ShaderTranslation result;

  if (spirv.empty()) {
    result.error = "the shader module holds no SPIR-V";
    return result;
  }

  try {
    spirv_cross::CompilerGLSL compiler(spirv);

    if (entry_point != "main") {
      // SPIRV-Cross translates the first entry point unless told otherwise, so
      // an application that names a different one would silently get the wrong
      // function.
      compiler.set_entry_point(entry_point, spv::ExecutionModelGLCompute);
    }

    spirv_cross::CompilerGLSL::Options options;
    // 4.50 is the first version in which compute shaders, shader storage blocks
    // and the std430 layout this shader was written against are all core, so no
    // extension pragmas are needed.
    options.version = 450;
    options.es = false;
    options.vulkan_semantics = false;
    compiler.set_common_options(options);

    const spirv_cross::ShaderResources resources =
        compiler.get_shader_resources();

    if (std::string missing = unsupported_feature(resources);
        !missing.empty()) {
      result.error = "shader uses " + missing +
                     ", which Kniz does not implement yet; M1 supports shader "
                     "storage buffers only";
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
