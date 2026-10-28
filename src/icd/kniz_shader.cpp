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

// Find the GL binding point the pipeline layout gave this (set, binding) pair.
// The type matters: GL numbers uniform buffers and shader storage buffers in
// separate namespaces, so the same set and binding can appear twice with
// different indices, and the wrong one would bind a uniform block's data where
// the shader reads a storage block.
bool lookup_gl_index(const std::vector<GlBinding>& bindings, uint32_t set,
                     uint32_t binding, VkDescriptorType type,
                     uint32_t* gl_index) {
  for (const GlBinding& candidate : bindings) {
    if (candidate.set == set && candidate.binding == binding &&
        candidate.type == type) {
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
//
// Textures and samplers are *not* on this list any more.  They are lowered
// rather than refused, and the lowering is the interesting part: see
// translate_shader.
std::string unsupported_feature(const spirv_cross::ShaderResources& resources) {
  if (!resources.storage_images.empty())
    return "storage images";
  if (!resources.atomic_counters.empty())
    return "atomic counters";
  if (!resources.acceleration_structures.empty())
    return "acceleration structures";
  return {};
}

// The name of a descriptor type in the words the error messages use.
const char* descriptor_name(VkDescriptorType type) {
  switch (type) {
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      return "uniform buffer";
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      return "storage buffer";
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      return "combined image sampler";
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      return "sampled image";
    case VK_DESCRIPTOR_TYPE_SAMPLER:
      return "sampler";
    default:
      return "descriptor";
  }
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

// Whether the module declares a push constant block at all, asked separately
// from the translation so that a failure can be attributed: the one translation
// failure that is peculiar to push constants is a block std140 cannot express,
// and an unrelated failure should not be described as one.  A malformed module
// makes this answer no, and the translation below is what reports it.
bool declares_push_constants(const std::vector<uint32_t>& spirv) {
  try {
    spirv_cross::CompilerGLSL compiler(spirv);
    return !compiler.get_shader_resources().push_constant_buffers.empty();
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace

ShaderTranslation translate_shader(const std::vector<uint32_t>& spirv,
                                   const std::string& entry_point,
                                   VkShaderStageFlagBits stage,
                                   const PipelineLayout& layout) {
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

  const bool has_push_constants = declares_push_constants(spirv);

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

    // GL has no push constants.  SPIRV-Cross can instead emit the push constant
    // block as a plain uniform block, which is what Kniz binds a buffer to; the
    // block's binding is rewritten below, like every other resource's.
    options.emit_push_constant_as_uniform_buffer = true;

    compiler.set_common_options(options);

    const spirv_cross::ShaderResources resources =
        compiler.get_shader_resources();

    if (std::string missing = unsupported_feature(resources);
        !missing.empty()) {
      result.error = "the " + std::string(stage_name(stage)) +
                     " shader uses " + missing +
                     ", which Kniz does not implement yet; only uniform and "
                     "shader storage buffers, push constants and vertex "
                     "attributes are supported so far";
      return result;
    }

    // A shader that reads push constants needs a layout that declares ranges for
    // them.  Vulkan says so too, but the failure mode here is worse than a
    // validation error: Kniz lowers the block onto the reserved uniform binding
    // point, and a layout with no ranges has a descriptor there instead, so the
    // shader would read a uniform buffer's first bytes as its push constants and
    // produce a plausible wrong answer.
    if (!resources.push_constant_buffers.empty() && !layout.has_push_constants) {
      result.error =
          "the " + std::string(stage_name(stage)) + " shader reads push "
          "constants, but the pipeline layout declares no push constant ranges "
          "for it; Kniz cannot tell which bytes the shader means";
      return result;
    }

    // Rewrite every descriptor onto the GL binding point the pipeline layout
    // assigned it, and remember which ones this stage actually declared.
    auto rewrite = [&](const spirv_cross::SmallVector<spirv_cross::Resource>& list,
                       VkDescriptorType type) -> bool {
      for (const spirv_cross::Resource& resource : list) {
        const uint32_t set =
            compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
        const uint32_t binding =
            compiler.get_decoration(resource.id, spv::DecorationBinding);

        uint32_t gl_index = 0;
        if (!lookup_gl_index(layout.bindings, set, binding, type, &gl_index)) {
          // A resource the pipeline layout does not declare.  Sticking it at
          // index 0 would bind it over some other resource, so refuse.
          result.error =
              "shader declares a " + std::string(descriptor_name(type)) +
              " at set " + std::to_string(set) + " binding " +
              std::to_string(binding) +
              ", but the pipeline layout has no such binding";
          return false;
        }

        compiler.set_decoration(resource.id, spv::DecorationBinding, gl_index);
        // GLSL has no descriptor sets.  Leaving the decoration in place makes
        // SPIRV-Cross emit layout(set = N, ...), which will not compile.
        compiler.set_decoration(resource.id, spv::DecorationDescriptorSet, 0);

        if (!merge_used_binding(result.used, GlBinding{set, binding, gl_index, type},
                                &result.error))
          return false;
      }
      return true;
    };

    if (!rewrite(resources.uniform_buffers, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER))
      return result;
    if (!rewrite(resources.storage_buffers, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER))
      return result;

    // Combined image samplers are the easy half: Vulkan declares one descriptor
    // and GLSL wants one sampler, so only the binding point has to move.
    if (!rewrite(resources.sampled_images,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER))
      return result;

    // The hard half.  GLSL has no separate image type, so a separate image and
    // sampler have to become one combined sampler.  SPIRV-Cross does the
    // rewriting, and what is left to Kniz is choosing the texture unit the
    // combined sampler reads: the image's, so that the replay can bind the
    // sampler object to the same unit and a sampler shared by several images
    // still works.  The pairing is recorded on the used binding because the
    // replay needs it to find the sampler descriptor.
    //
    // texelFetch on a bare image has no sampler argument in Vulkan and needs one
    // in GLSL, so SPIRV-Cross injects a dummy sampler.  It is not a descriptor:
    // it is given the first unit past every descriptor, and nothing has to be
    // bound there, because a fetch ignores sampler state.
    const spirv_cross::VariableID dummy_sampler =
        compiler.build_dummy_sampler_for_combined_images();
    if (dummy_sampler != 0) {
      compiler.set_decoration(dummy_sampler, spv::DecorationBinding,
                              layout.dummy_sampler_gl_index());
      compiler.set_decoration(dummy_sampler, spv::DecorationDescriptorSet, 0);
    }
    compiler.build_combined_image_samplers();

    for (const spirv_cross::CombinedImageSampler& combined :
         compiler.get_combined_image_samplers()) {
      const uint32_t image_set =
          compiler.get_decoration(combined.image_id, spv::DecorationDescriptorSet);
      const uint32_t image_binding =
          compiler.get_decoration(combined.image_id, spv::DecorationBinding);

      uint32_t unit = 0;
      if (!lookup_gl_index(layout.bindings, image_set, image_binding,
                           VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &unit)) {
        result.error =
            "shader reads the sampled image at set " + std::to_string(image_set) +
            " binding " + std::to_string(image_binding) +
            ", but the pipeline layout does not declare a "
            "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE there";
        return result;
      }

      GlBinding used;
      used.set = image_set;
      used.binding = image_binding;
      used.gl_index = unit;
      used.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;

      if (combined.sampler_id != dummy_sampler) {
        const uint32_t sampler_set = compiler.get_decoration(
            combined.sampler_id, spv::DecorationDescriptorSet);
        const uint32_t sampler_binding = compiler.get_decoration(
            combined.sampler_id, spv::DecorationBinding);

        uint32_t sampler_unit = 0;
        if (!lookup_gl_index(layout.bindings, sampler_set, sampler_binding,
                             VK_DESCRIPTOR_TYPE_SAMPLER, &sampler_unit)) {
          result.error =
              "shader pairs the sampled image at set " +
              std::to_string(image_set) + " binding " +
              std::to_string(image_binding) + " with a sampler at set " +
              std::to_string(sampler_set) + " binding " +
              std::to_string(sampler_binding) +
              ", but the pipeline layout does not declare a "
              "VK_DESCRIPTOR_TYPE_SAMPLER there";
          return result;
        }
        used.sampler_set = sampler_set;
        used.sampler_binding = sampler_binding;
      }

      compiler.set_decoration(combined.combined_id, spv::DecorationBinding, unit);
      compiler.set_decoration(combined.combined_id, spv::DecorationDescriptorSet,
                              0);

      if (!merge_used_binding(result.used, used, &result.error))
        return result;
    }

    for (const spirv_cross::Resource& resource : resources.push_constant_buffers) {
      compiler.set_decoration(resource.id, spv::DecorationBinding,
                              layout.push_constant_gl_index);
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

    // SPIRV-Cross's own message for a block it cannot express mentions blocks
    // and layouts, not push constants, so the case that is specific to this
    // driver is spelled out here.  Only for a shader that has a push constant
    // block at all: a shader that failed for some unrelated reason should not be
    // told about push constant packing.
    if (has_push_constants) {
      result.error +=
          " (a push constant block that needs tighter packing than std140 -- an "
          "array of scalars, for instance -- cannot be expressed as an OpenGL "
          "uniform block, which is what Kniz lowers push constants onto)";
    }
  }

  return result;
}

}  // namespace kniz
