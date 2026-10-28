// Shader modules, pipeline layouts' compiled form, and pipelines.
//
// The interesting decision here is *when* the shader is translated.  A
// VkShaderModule on its own is not enough: SPIR-V addresses resources by
// (set, binding), GLSL addresses them by a single flat binding number, and
// mapping one onto the other needs the pipeline layout.  So vkCreateShaderModule
// only keeps the words, and the GLSL is produced at pipeline creation, when the
// layout is known.  That is also why a shader module costs nothing to create and
// a pipeline is where translation errors surface.

#include "kniz_common.h"
#include "kniz_objects.h"
#include "kniz_shader.h"

#include <new>
#include <string>
#include <vector>

namespace kniz {
namespace {

// One translated stage, ready to compile.
struct ShaderSource {
  gl::Enum stage = 0;
  std::string glsl;
};

std::string shader_info_log(gl::Uint shader) {
  gl::Int length = 0;
  gl::f.GetShaderiv(shader, gl::kInfoLogLength, &length);
  if (length <= 0)
    return "(the driver supplied no log)";
  std::string log(static_cast<size_t>(length), '\0');
  gl::Size written = 0;
  gl::f.GetShaderInfoLog(shader, length, &written, log.data());
  if (written > 0 && static_cast<size_t>(written) <= log.size())
    log.resize(static_cast<size_t>(written));
  return log;
}

std::string program_info_log(gl::Uint program) {
  gl::Int length = 0;
  gl::f.GetProgramiv(program, gl::kInfoLogLength, &length);
  if (length <= 0)
    return "(the driver supplied no log)";
  std::string log(static_cast<size_t>(length), '\0');
  gl::Size written = 0;
  gl::f.GetProgramInfoLog(program, length, &written, log.data());
  if (written > 0 && static_cast<size_t>(written) <= log.size())
    log.resize(static_cast<size_t>(written));
  return log;
}

bool compile_program(const std::vector<ShaderSource>& sources, gl::Uint* program,
                     std::string* error) {
  if (sources.empty()) {
    *error = "a program needs at least one shader stage";
    return false;
  }

  gl::Uint linked = gl::f.CreateProgram();
  if (linked == 0) {
    *error = "glCreateProgram failed";
    return false;
  }

  // Every compiled shader is deleted as soon as it is attached.  The program
  // keeps it alive, so the object can go now -- but only *after* attaching, and
  // any path that bails out has to clean up whatever it has made so far.
  std::vector<gl::Uint> shaders;

  auto clean_up = [&]() {
    for (gl::Uint shader : shaders)
      gl::f.DeleteShader(shader);
    gl::f.DeleteProgram(linked);
  };

  for (const ShaderSource& source : sources) {
    const gl::Uint shader = gl::f.CreateShader(source.stage);
    if (shader == 0) {
      *error = "glCreateShader failed";
      clean_up();
      return false;
    }
    shaders.push_back(shader);

    const gl::Char* text = source.glsl.c_str();
    gl::f.ShaderSource(shader, 1, &text, nullptr);
    gl::f.CompileShader(shader);

    gl::Int status = 0;
    gl::f.GetShaderiv(shader, gl::kCompileStatus, &status);
    if (status == 0) {
      *error = "GLSL compilation failed: " + shader_info_log(shader);
      clean_up();
      return false;
    }

    gl::f.AttachShader(linked, shader);
  }

  gl::f.LinkProgram(linked);
  for (gl::Uint shader : shaders)
    gl::f.DeleteShader(shader);

  gl::Int status = 0;
  gl::f.GetProgramiv(linked, gl::kLinkStatus, &status);
  if (status == 0) {
    *error = "GLSL linking failed: " + program_info_log(linked);
    gl::f.DeleteProgram(linked);
    return false;
  }

  *program = linked;
  return true;
}

// Merge the descriptors one translated stage reported into the pipeline's list.
// The mapping itself lives on the pipeline layout, because it is a property of
// the layout; what is collected here is only the subset the shaders actually
// read, so that a layout shared between pipelines does not require every
// pipeline to bind every binding it declares.
bool add_used_bindings(Pipeline* pipeline, const ShaderTranslation& translation) {
  for (const GlBinding& used : translation.used) {
    std::string error;
    if (!merge_used_binding(pipeline->used_bindings, used, &error)) {
      KNIZ_ERR("%s", error.c_str());
      return false;
    }
  }
  return true;
}

VkResult build_compute_pipeline(Device* device,
                                const VkComputePipelineCreateInfo& info,
                                VkPipeline* out) {
  *out = VK_NULL_HANDLE;

  const VkPipelineShaderStageCreateInfo& stage = info.stage;
  if (stage.stage != VK_SHADER_STAGE_COMPUTE_BIT) {
    KNIZ_ERR("vkCreateComputePipelines was given a stage of type 0x%x; only "
             "VK_SHADER_STAGE_COMPUTE_BIT is valid here",
             stage.stage);
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (stage.pSpecializationInfo != nullptr) {
    KNIZ_ERR("specialization constants are not implemented");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  ShaderModule* module = from_handle(stage.module);
  if (module == nullptr) {
    KNIZ_ERR("the shader stage references something that is not a Kniz shader "
             "module");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  PipelineLayout* layout = from_handle(info.layout);
  if (layout == nullptr) {
    KNIZ_ERR("the pipeline references something that is not a Kniz pipeline "
             "layout");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  const std::string entry_point = stage.pName != nullptr ? stage.pName : "main";

  Pipeline* pipeline = new (std::nothrow) Pipeline();
  if (pipeline == nullptr)
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  pipeline->device = device;
  pipeline->bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
  pipeline->layout = layout;

  const ShaderTranslation translation = translate_shader(
      module->spirv, entry_point, VK_SHADER_STAGE_COMPUTE_BIT, *layout);
  if (!translation.ok) {
    KNIZ_ERR("could not translate the compute shader: %s",
             translation.error.c_str());
    delete pipeline;
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  pipeline->glsl = translation.glsl;
  if (!add_used_bindings(pipeline, translation)) {
    delete pipeline;
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  KNIZ_DBG("GLSL generated for pipeline (%zu descriptor binding(s), %zu "
           "push constant range(s)):\n%s",
           pipeline->used_bindings.size(),
           layout->push_constant_ranges.size(), pipeline->glsl.c_str());

  std::string error;
  const ShaderSource source{gl::kComputeShader, translation.glsl};
  if (!compile_program({source}, &pipeline->program, &error)) {
    KNIZ_ERR("%s", error.c_str());
    KNIZ_ERR("the GLSL that failed was:\n%s", pipeline->glsl.c_str());
    delete pipeline;
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  *out = to_handle(pipeline);
  KNIZ_INFO("compute pipeline created: GL program %u, %zu descriptor "
            "binding(s)%s",
            pipeline->program, pipeline->used_bindings.size(),
            layout->has_push_constants ? ", push constants lowered onto a "
                                         "uniform buffer"
                                       : "");
  return VK_SUCCESS;
}

// ---------------------------------------------------------------------------
// Graphics pipelines
// ---------------------------------------------------------------------------

// Whether a piece of dynamic state is declared, needed by the viewport and
// scissor stage above.  Vulkan stores dynamic state as a list rather than a bit
// set, so this is a search.
bool has_dynamic_state(const VkGraphicsPipelineCreateInfo& info,
                       VkDynamicState state) {
  if (info.pDynamicState == nullptr)
    return false;
  for (uint32_t i = 0; i < info.pDynamicState->dynamicStateCount; ++i) {
    if (info.pDynamicState->pDynamicStates[i] == state)
      return true;
  }
  return false;
}

// The vertex formats Kniz can feed to a vertex attribute, and what GL needs to
// hear for each.  This table is the whole of the format translation, and it is
// deliberately a table rather than arithmetic over the format enum: the map from
// a Vulkan format to a (type, components, normalized) triple is not a function of
// its bits.
//
// Two families are absent on purpose.
//
//   * Integer formats (R32G32_SINT and friends) need glVertexAttribIPointer, not
//     glVertexAttribPointer, because GL keeps integer and floating point vertex
//     attributes in separate pipelines.  A float-format table cannot express that
//     and quietly using the float entry point would give the shader garbage.
//   * B8G8R8A8_UNORM has no representation at all: GL has a BGRA *pixel* format
//     for texture uploads but no BGRA vertex attribute type, so a four-byte
//     attribute would be read as if it were RGBA.
struct VertexFormat {
  VkFormat vk = VK_FORMAT_UNDEFINED;
  gl::Enum type = 0;
  gl::Int components = 0;
  bool normalized = false;
};

constexpr VertexFormat kVertexFormats[] = {
    {VK_FORMAT_R32_SFLOAT, gl::kFloat, 1, false},
    {VK_FORMAT_R32G32_SFLOAT, gl::kFloat, 2, false},
    {VK_FORMAT_R32G32B32_SFLOAT, gl::kFloat, 3, false},
    {VK_FORMAT_R32G32B32A32_SFLOAT, gl::kFloat, 4, false},
    {VK_FORMAT_R16_SFLOAT, gl::kHalfFloat, 1, false},
    {VK_FORMAT_R16G16_SFLOAT, gl::kHalfFloat, 2, false},
    {VK_FORMAT_R16G16B16A16_SFLOAT, gl::kHalfFloat, 4, false},
    {VK_FORMAT_R8_UNORM, gl::kUnsignedByte, 1, true},
    {VK_FORMAT_R8G8_UNORM, gl::kUnsignedByte, 2, true},
    {VK_FORMAT_R8G8B8A8_UNORM, gl::kUnsignedByte, 4, true},
    {VK_FORMAT_R8G8B8A8_SNORM, gl::kByte, 4, true},
    {VK_FORMAT_R16_UNORM, gl::kUnsignedShort, 1, true},
    {VK_FORMAT_R16G16_UNORM, gl::kUnsignedShort, 2, true},
    {VK_FORMAT_R16G16B16A16_UNORM, gl::kUnsignedShort, 4, true},
};

const VertexFormat* vertex_format(VkFormat format) {
  for (const VertexFormat& candidate : kVertexFormats) {
    if (candidate.vk == format)
      return &candidate;
  }
  return nullptr;
}

// Vulkan primitive topology to GL draw mode.  Adjacency topologies have GL
// equivalents but no M2b use, and patch lists are tessellation, which is refused
// with the stage that would consume them.
bool draw_mode(VkPrimitiveTopology topology, const char** name, gl::Enum* mode) {
  switch (topology) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
      *name = "VK_PRIMITIVE_TOPOLOGY_POINT_LIST";
      *mode = gl::kPoints;
      return true;
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
      *name = "VK_PRIMITIVE_TOPOLOGY_LINE_LIST";
      *mode = gl::kLines;
      return true;
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
      *name = "VK_PRIMITIVE_TOPOLOGY_LINE_STRIP";
      *mode = gl::kLineStrip;
      return true;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      *name = "VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST";
      *mode = gl::kTriangles;
      return true;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
      *name = "VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP";
      *mode = gl::kTriangleStrip;
      return true;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
      *name = "VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN";
      *mode = gl::kTriangleFan;
      return true;
    default:
      return false;
  }
}

VkResult build_graphics_pipeline(Device* device,
                                 const VkGraphicsPipelineCreateInfo& info,
                                 VkPipeline* out) {
  *out = VK_NULL_HANDLE;

  if (info.pVertexInputState == nullptr || info.pInputAssemblyState == nullptr ||
      info.pViewportState == nullptr || info.pRasterizationState == nullptr ||
      info.pMultisampleState == nullptr || info.pColorBlendState == nullptr) {
    KNIZ_ERR("a graphics pipeline must provide vertex input, input assembly, "
             "viewport, rasterization, multisample and colour blend state");
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (info.pTessellationState != nullptr) {
    KNIZ_ERR("tessellation pipelines are not implemented");
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  RenderPass* render_pass = from_handle(info.renderPass);
  if (render_pass == nullptr) {
    KNIZ_ERR("the graphics pipeline references something that is not a Kniz "
             "render pass");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  PipelineLayout* layout = from_handle(info.layout);
  if (layout == nullptr) {
    KNIZ_ERR("the pipeline references something that is not a Kniz pipeline "
             "layout");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  // ---- stages -------------------------------------------------------------
  // Exactly one vertex and one fragment stage.  Geometry and tessellation are
  // expressible in GL but not implemented here, and a pipeline missing one of the
  // two stages would be refused by GL at link time with a much worse message.
  ShaderModule* vertex_module = nullptr;
  ShaderModule* fragment_module = nullptr;
  std::string vertex_entry = "main";
  std::string fragment_entry = "main";

  for (uint32_t i = 0; i < info.stageCount; ++i) {
    const VkPipelineShaderStageCreateInfo& stage = info.pStages[i];
    if (stage.pSpecializationInfo != nullptr) {
      KNIZ_ERR("specialization constants are not implemented");
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    ShaderModule* module = from_handle(stage.module);
    if (module == nullptr) {
      KNIZ_ERR("shader stage %u references something that is not a Kniz shader "
               "module",
               i);
      return VK_ERROR_INITIALIZATION_FAILED;
    }

    const std::string entry = stage.pName != nullptr ? stage.pName : "main";
    switch (stage.stage) {
      case VK_SHADER_STAGE_VERTEX_BIT:
        vertex_module = module;
        vertex_entry = entry;
        break;
      case VK_SHADER_STAGE_FRAGMENT_BIT:
        fragment_module = module;
        fragment_entry = entry;
        break;
      default:
        KNIZ_ERR("shader stage flags 0x%x are not implemented; only vertex and "
                 "fragment stages are",
                 static_cast<unsigned>(stage.stage));
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
  }

  if (vertex_module == nullptr || fragment_module == nullptr) {
    KNIZ_ERR("a graphics pipeline needs exactly one vertex and one fragment "
             "stage, and this one has%s",
             vertex_module == nullptr
                 ? (fragment_module == nullptr ? " neither" : " no vertex stage")
                 : " no fragment stage");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  Pipeline* pipeline = new (std::nothrow) Pipeline();
  if (pipeline == nullptr)
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  pipeline->device = device;
  pipeline->bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
  pipeline->layout = layout;
  pipeline->render_pass = render_pass;

  // ---- vertex input -------------------------------------------------------
  {
    const VkPipelineVertexInputStateCreateInfo& input = *info.pVertexInputState;
    pipeline->vertex_bindings.resize(input.vertexBindingDescriptionCount);
    for (uint32_t i = 0; i < input.vertexBindingDescriptionCount; ++i) {
      const VkVertexInputBindingDescription& source =
          input.pVertexBindingDescriptions[i];
      if (source.inputRate != VK_VERTEX_INPUT_RATE_VERTEX) {
        KNIZ_ERR("VK_VERTEX_INPUT_RATE_INSTANCE is not implemented (binding %u); "
                 "it needs glVertexAttribDivisor",
                 source.binding);
        delete pipeline;
        return VK_ERROR_FEATURE_NOT_PRESENT;
      }
      pipeline->vertex_bindings[i] =
          VertexBinding{source.binding, source.stride, source.inputRate};
    }

    pipeline->vertex_attributes.reserve(input.vertexAttributeDescriptionCount);
    for (uint32_t i = 0; i < input.vertexAttributeDescriptionCount; ++i) {
      const VkVertexInputAttributeDescription& source =
          input.pVertexAttributeDescriptions[i];

      const VertexFormat* format = vertex_format(source.format);
      if (format == nullptr) {
        KNIZ_ERR("vertex attribute %u (location %u) has format %d, which Kniz "
                 "cannot express as a vertex attribute; integer vertex formats "
                 "need glVertexAttribIPointer and B8G8R8A8_UNORM has no GL "
                 "attribute type at all",
                 i, source.location, static_cast<int>(source.format));
        delete pipeline;
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
      }

      bool binding_known = false;
      for (const VertexBinding& binding : pipeline->vertex_bindings)
        binding_known = binding_known || binding.binding == source.binding;
      if (!binding_known) {
        KNIZ_ERR("vertex attribute %u reads binding %u, which the pipeline's "
                 "vertex input does not describe",
                 i, source.binding);
        delete pipeline;
        return VK_ERROR_INITIALIZATION_FAILED;
      }

      VertexAttribute attribute;
      attribute.location = source.location;
      attribute.binding = source.binding;
      attribute.type = format->type;
      attribute.components = format->components;
      attribute.normalized = format->normalized;
      attribute.offset = source.offset;
      pipeline->vertex_attributes.push_back(attribute);
    }
  }

  // ---- input assembly -----------------------------------------------------
  {
    const VkPipelineInputAssemblyStateCreateInfo& assembly = *info.pInputAssemblyState;
    const char* topology_name = nullptr;
    if (!draw_mode(assembly.topology, &topology_name, &pipeline->draw_mode)) {
      KNIZ_ERR("primitive topology %d is not implemented",
               static_cast<int>(assembly.topology));
      delete pipeline;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (assembly.primitiveRestartEnable) {
      KNIZ_ERR("primitive restart is not implemented (topology %s)",
               topology_name);
      delete pipeline;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
  }

  // ---- viewport and scissor ----------------------------------------------
  {
    const VkPipelineViewportStateCreateInfo& viewport = *info.pViewportState;
    // Vulkan allows several viewports, but only when a geometry shader selects
    // between them; there is at most one here.
    if (viewport.viewportCount != 1 || viewport.scissorCount != 1) {
      KNIZ_ERR("exactly one viewport and one scissor are implemented (asked for "
               "%u and %u)",
               viewport.viewportCount, viewport.scissorCount);
      delete pipeline;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    pipeline->dynamic_viewport =
        has_dynamic_state(info, VK_DYNAMIC_STATE_VIEWPORT);
    pipeline->dynamic_scissor = has_dynamic_state(info, VK_DYNAMIC_STATE_SCISSOR);

    if (!pipeline->dynamic_viewport) {
      if (viewport.pViewports == nullptr) {
        KNIZ_ERR("a static viewport was declared but none was provided");
        delete pipeline;
        return VK_ERROR_INITIALIZATION_FAILED;
      }
      pipeline->viewport = viewport.pViewports[0];

      // The depth range is applied at replay, from the viewport the draw
      // recorded -- this one here, or the dynamic one if the pipeline declares
      // dynamic viewport state.  It is not stored separately because Vulkan
      // defines it as part of the viewport.
    } else if (viewport.pViewports != nullptr) {
      KNIZ_DBG("the pipeline declares dynamic viewport state and also provides "
               "a viewport; the provided one is ignored, as Vulkan says");
    }

    if (!pipeline->dynamic_scissor) {
      if (viewport.pScissors == nullptr) {
        KNIZ_ERR("a static scissor was declared but none was provided");
        delete pipeline;
        return VK_ERROR_INITIALIZATION_FAILED;
      }
      pipeline->scissor = viewport.pScissors[0];
    }
  }

  // ---- rasterization -----------------------------------------------------
  {
    const VkPipelineRasterizationStateCreateInfo& raster = *info.pRasterizationState;
    if (raster.polygonMode != VK_POLYGON_MODE_FILL) {
      KNIZ_ERR("only VK_POLYGON_MODE_FILL is implemented (asked for %d); "
               "glPolygonMode exists but is not wired up",
               static_cast<int>(raster.polygonMode));
      delete pipeline;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (raster.depthClampEnable) {
      KNIZ_ERR("depth clamping is not implemented");
      delete pipeline;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (raster.rasterizerDiscardEnable) {
      KNIZ_ERR("rasterizer discard is not implemented");
      delete pipeline;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    switch (raster.cullMode) {
      case VK_CULL_MODE_NONE:
        pipeline->cull_face = gl::kNone;
        break;
      case VK_CULL_MODE_BACK_BIT:
        pipeline->cull_face = gl::kBack;
        break;
      case VK_CULL_MODE_FRONT_BIT:
        pipeline->cull_face = gl::kFront;
        break;
      default:
        KNIZ_ERR("VK_CULL_MODE_FRONT_AND_BACK discards every triangle and is not "
                 "implemented");
        delete pipeline;
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    if (raster.frontFace != VK_FRONT_FACE_COUNTER_CLOCKWISE &&
        raster.frontFace != VK_FRONT_FACE_CLOCKWISE) {
      KNIZ_ERR("front face %d is not implemented", static_cast<int>(raster.frontFace));
      delete pipeline;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    // No flip is involved in the winding order either: GL and Vulkan both call a
    // triangle counter-clockwise when the same sign comes out of the same
    // arithmetic, because both map NDC y = -1 to row 0.  See docs/DESIGN.md.
    pipeline->front_face = raster.frontFace == VK_FRONT_FACE_COUNTER_CLOCKWISE
                               ? gl::kCcw
                               : gl::kCw;
  }

  // ---- multisample -------------------------------------------------------
  {
    const VkPipelineMultisampleStateCreateInfo& multisample = *info.pMultisampleState;
    if (multisample.rasterizationSamples != VK_SAMPLE_COUNT_1_BIT) {
      KNIZ_ERR("multisampling is not implemented (%u samples requested)",
               static_cast<unsigned>(multisample.rasterizationSamples));
      delete pipeline;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (multisample.sampleShadingEnable) {
      KNIZ_ERR("sample shading is not implemented");
      delete pipeline;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (multisample.alphaToCoverageEnable) {
      KNIZ_ERR("alpha to coverage is not implemented");
      delete pipeline;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (multisample.alphaToOneEnable) {
      KNIZ_ERR("alpha to one is not implemented");
      delete pipeline;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
  }

  // ---- depth and stencil -------------------------------------------------
  //
  // Depth test and depth write map onto GL's depth state one for one.  What does
  // not is the stencil: GL's stencil state is per-face operation triples, and
  // getting it wrong writes to the stencil buffer silently, so it is refused until
  // there is a test that checks it.  The depth bounds test is refused for the same
  // reason -- it is a separate piece of state with no default that matches.
  if (info.pDepthStencilState != nullptr) {
    const VkPipelineDepthStencilStateCreateInfo& depth = *info.pDepthStencilState;

    if (depth.stencilTestEnable) {
      KNIZ_ERR("the stencil test is not implemented");
      delete pipeline;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (depth.depthBoundsTestEnable) {
      KNIZ_ERR("the depth bounds test is not implemented");
      delete pipeline;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (depth.depthTestEnable || depth.depthWriteEnable) {
      if (depth.depthCompareOp > VK_COMPARE_OP_ALWAYS) {
        KNIZ_ERR("depth compare operation %d is not one of the eight",
                 static_cast<int>(depth.depthCompareOp));
        delete pipeline;
        return VK_ERROR_INITIALIZATION_FAILED;
      }
      // VkCompareOp and GL's comparison functions are the same eight, in the same
      // order, which is what makes an index into this table enough.
      constexpr gl::Enum kCompareOps[] = {
          gl::kNever,  gl::kLess,    gl::kEqual,  gl::kLessOrEqual,
          gl::kGreater, gl::kNotEqual, gl::kGreaterOrEqual, gl::kAlways};
      pipeline->depth_test = depth.depthTestEnable != VK_FALSE;
      pipeline->depth_write = depth.depthWriteEnable != VK_FALSE;
      pipeline->depth_func = kCompareOps[depth.depthCompareOp];
    }
  }

  // ---- colour blend ------------------------------------------------------
  {
    const VkPipelineColorBlendStateCreateInfo& blend = *info.pColorBlendState;
    if (blend.logicOpEnable) {
      KNIZ_ERR("logic operations are not implemented");
      delete pipeline;
      return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (blend.attachmentCount != render_pass->color_attachments.size()) {
      KNIZ_ERR("the colour blend state covers %u attachment(s) but the render "
               "pass subpass has %zu; Kniz requires them to match",
               blend.attachmentCount, render_pass->color_attachments.size());
      delete pipeline;
      return VK_ERROR_INITIALIZATION_FAILED;
    }

    pipeline->color_write = 0;
    for (uint32_t i = 0; i < blend.attachmentCount; ++i) {
      if (blend.pAttachments[i].blendEnable) {
        KNIZ_ERR("blending is not implemented yet (colour attachment %u); only "
                 "the write mask is applied",
                 i);
        delete pipeline;
        return VK_ERROR_FEATURE_NOT_PRESENT;
      }
      // Vulkan takes the same four bits as GL, but as a mask rather than four
      // arguments, so they are collected here and passed at replay.
      pipeline->color_write |= blend.pAttachments[i].colorWriteMask;
    }
  }

  // ---- dynamic state -----------------------------------------------------
  if (info.pDynamicState != nullptr) {
    for (uint32_t i = 0; i < info.pDynamicState->dynamicStateCount; ++i) {
      const VkDynamicState state = info.pDynamicState->pDynamicStates[i];
      if (state != VK_DYNAMIC_STATE_VIEWPORT && state != VK_DYNAMIC_STATE_SCISSOR) {
        KNIZ_ERR("dynamic state %d is not implemented; only VIEWPORT and SCISSOR "
                 "are",
                 static_cast<int>(state));
        delete pipeline;
        return VK_ERROR_FEATURE_NOT_PRESENT;
      }
    }
  }

  // ---- shaders -----------------------------------------------------------
  struct StageJob {
    VkShaderStageFlagBits stage;
    ShaderModule* module;
    const std::string* entry;
    gl::Enum gl_stage;
  };
  const StageJob jobs[] = {
      {VK_SHADER_STAGE_VERTEX_BIT, vertex_module, &vertex_entry,
       gl::kVertexShader},
      {VK_SHADER_STAGE_FRAGMENT_BIT, fragment_module, &fragment_entry,
       gl::kFragmentShader},
  };

  std::vector<ShaderSource> sources;
  for (const StageJob& job : jobs) {
    const ShaderTranslation translation =
        translate_shader(job.module->spirv, *job.entry, job.stage, *layout);
    if (!translation.ok) {
      KNIZ_ERR("could not translate the %s shader: %s",
               job.stage == VK_SHADER_STAGE_VERTEX_BIT ? "vertex" : "fragment",
               translation.error.c_str());
      delete pipeline;
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (!add_used_bindings(pipeline, translation)) {
      delete pipeline;
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (sources.empty())
      pipeline->glsl = translation.glsl;  // the vertex stage, kept for diagnostics
    sources.push_back(ShaderSource{job.gl_stage, translation.glsl});
    KNIZ_DBG("GLSL generated for the %s stage:\n%s",
             job.stage == VK_SHADER_STAGE_VERTEX_BIT ? "vertex" : "fragment",
             translation.glsl.c_str());
  }

  std::string error;
  if (!compile_program(sources, &pipeline->program, &error)) {
    KNIZ_ERR("%s", error.c_str());
    for (const ShaderSource& source : sources)
      KNIZ_ERR("the GLSL that failed was:\n%s", source.glsl.c_str());
    delete pipeline;
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  // The vertex array object holds the attribute pointers, and the pointers can
  // only be set with a buffer bound, so it is created here and re-pointed at each
  // draw -- when the buffers a draw names are finally known.
  gl::f.GenVertexArrays(1, &pipeline->vao);
  if (pipeline->vao == 0) {
    KNIZ_ERR("glGenVertexArrays failed");
    gl::f.DeleteProgram(pipeline->program);
    delete pipeline;
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  }
  gl::f.BindVertexArray(pipeline->vao);
  gl::f.BindVertexArray(0);
  if (gl::report_errors("after creating the pipeline's vertex array object")) {
    gl::f.DeleteVertexArrays(1, &pipeline->vao);
    gl::f.DeleteProgram(pipeline->program);
    delete pipeline;
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  *out = to_handle(pipeline);
  KNIZ_INFO("graphics pipeline created: GL program %u, vertex array %u, %zu "
            "attribute(s), %zu colour attachment(s)",
            pipeline->program, pipeline->vao, pipeline->vertex_attributes.size(),
            render_pass->color_attachments.size());
  return VK_SUCCESS;
}

}  // namespace

// ---------------------------------------------------------------------------
// Shader modules
// ---------------------------------------------------------------------------
VKAPI_ATTR VkResult VKAPI_CALL CreateShaderModule(
    VkDevice deviceHandle, const VkShaderModuleCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule) {
  (void)pAllocator;
  if (pShaderModule == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pShaderModule = VK_NULL_HANDLE;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr || pCreateInfo == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  if (pCreateInfo->codeSize == 0 || pCreateInfo->codeSize % sizeof(uint32_t) != 0) {
    KNIZ_ERR("shader module code size %zu is not a non-zero multiple of 4",
             pCreateInfo->codeSize);
    return VK_ERROR_INVALID_SHADER_NV;
  }
  if (pCreateInfo->pCode == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  ShaderModule* module = new (std::nothrow) ShaderModule();
  if (module == nullptr)
    return VK_ERROR_OUT_OF_HOST_MEMORY;

  module->device = device;
  const size_t words = pCreateInfo->codeSize / sizeof(uint32_t);
  module->spirv.assign(pCreateInfo->pCode, pCreateInfo->pCode + words);

  // Cheap sanity check on the magic number.  It costs nothing and turns the
  // common mistake of handing over a non-SPIR-V blob into an error at the point
  // of the mistake rather than deep inside SPIRV-Cross.
  constexpr uint32_t kSpirvMagic = 0x07230203;
  if (module->spirv[0] != kSpirvMagic) {
    KNIZ_ERR("shader module does not begin with the SPIR-V magic number "
             "(found 0x%08x, expected 0x%08x)",
             module->spirv[0], kSpirvMagic);
    delete module;
    return VK_ERROR_INVALID_SHADER_NV;
  }

  *pShaderModule = to_handle(module);
  KNIZ_TRACE("shader module %p: %zu SPIR-V words",
             static_cast<void*>(*pShaderModule), words);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyShaderModule(
    VkDevice deviceHandle, VkShaderModule shaderModuleHandle,
    const VkAllocationCallbacks* pAllocator) {
  (void)pAllocator;
  if (from_handle(deviceHandle) == nullptr)
    return;
  delete from_handle(shaderModuleHandle);
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreateComputePipelines(
    VkDevice deviceHandle, VkPipelineCache pipelineCache,
    uint32_t createInfoCount, const VkComputePipelineCreateInfo* pCreateInfos,
    const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines) {
  (void)pAllocator;
  (void)pipelineCache;  // the GL driver's own program cache does the caching

  Device* device = from_handle(deviceHandle);
  if (device == nullptr || pCreateInfos == nullptr || pPipelines == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (!device->instance->gl_ready) {
    KNIZ_ERR("no OpenGL context is available");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  if (!device->instance->gl.compute_capable) {
    KNIZ_ERR("compute shaders need OpenGL 4.3; this context is %d.%d",
             device->instance->gl.major, device->instance->gl.minor);
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  // Compiling a program is a GL call, and GL state belongs to whichever context
  // is current on the calling thread.
  if (!gl::make_current()) {
    KNIZ_ERR("could not make the OpenGL context current on this thread");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  VkResult worst = VK_SUCCESS;
  for (uint32_t i = 0; i < createInfoCount; ++i) {
    const VkResult result =
        build_compute_pipeline(device, pCreateInfos[i], &pPipelines[i]);
    if (result != VK_SUCCESS && worst == VK_SUCCESS)
      worst = result;
  }
  return worst;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateGraphicsPipelines(
    VkDevice deviceHandle, VkPipelineCache pipelineCache,
    uint32_t createInfoCount, const VkGraphicsPipelineCreateInfo* pCreateInfos,
    const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines) {
  (void)pAllocator;
  (void)pipelineCache;  // the GL driver's own program cache does the caching

  Device* device = from_handle(deviceHandle);
  if (device == nullptr || pCreateInfos == nullptr || pPipelines == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (!device->instance->gl_ready) {
    KNIZ_ERR("no OpenGL context is available");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  // Compiling a program and creating a vertex array object are GL calls, and GL
  // state belongs to whichever context is current on the calling thread.  The
  // context is ours and was created on some thread; this may be another one.
  if (!gl::make_current()) {
    KNIZ_ERR("could not make the OpenGL context current on this thread");
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  if (!device->instance->gl.render_capable) {
    KNIZ_ERR("drawing needs framebuffer objects and vertex array objects, which "
             "this OpenGL context (version %d.%d) does not provide",
             device->instance->gl.major, device->instance->gl.minor);
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  VkResult worst = VK_SUCCESS;
  for (uint32_t i = 0; i < createInfoCount; ++i) {
    const VkResult result =
        build_graphics_pipeline(device, pCreateInfos[i], &pPipelines[i]);
    if (result != VK_SUCCESS && worst == VK_SUCCESS)
      worst = result;
  }
  return worst;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipeline(VkDevice deviceHandle,
                                           VkPipeline pipelineHandle,
                                           const VkAllocationCallbacks* pAllocator) {
  (void)pAllocator;
  if (from_handle(deviceHandle) == nullptr)
    return;
  Pipeline* pipeline = from_handle(pipelineHandle);
  if (pipeline == nullptr)
    return;
  if (pipeline->vao != 0)
    gl::f.DeleteVertexArrays(1, &pipeline->vao);
  if (pipeline->program != 0)
    gl::f.DeleteProgram(pipeline->program);
  delete pipeline;
}

// ---------------------------------------------------------------------------
// Pipeline cache
//
// Kniz stores nothing in one.  It exists because applications create a cache
// unconditionally and would otherwise receive a null handle.
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineCache(
    VkDevice deviceHandle, const VkPipelineCacheCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkPipelineCache* pPipelineCache) {
  (void)pAllocator;
  if (pPipelineCache == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  *pPipelineCache = VK_NULL_HANDLE;

  Device* device = from_handle(deviceHandle);
  if (device == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  if (pCreateInfo != nullptr && pCreateInfo->initialDataSize > 0)
    KNIZ_DBG("ignoring %zu bytes of initial pipeline cache data; Kniz does not "
             "cache compiled pipelines",
             pCreateInfo->initialDataSize);

  PipelineCache* cache = new (std::nothrow) PipelineCache();
  if (cache == nullptr)
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  cache->device = device;

  *pPipelineCache = to_handle(cache);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipelineCache(
    VkDevice deviceHandle, VkPipelineCache pipelineCacheHandle,
    const VkAllocationCallbacks* pAllocator) {
  (void)pAllocator;
  if (from_handle(deviceHandle) == nullptr)
    return;
  delete from_handle(pipelineCacheHandle);
}

VKAPI_ATTR VkResult VKAPI_CALL GetPipelineCacheData(
    VkDevice deviceHandle, VkPipelineCache pipelineCacheHandle, size_t* pDataSize,
    void* pData) {
  (void)pData;
  if (pDataSize == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (from_handle(deviceHandle) == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (from_handle(pipelineCacheHandle) == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  // An empty cache is a valid cache.
  *pDataSize = 0;
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL MergePipelineCaches(
    VkDevice deviceHandle, VkPipelineCache dstCache, uint32_t srcCacheCount,
    const VkPipelineCache* pSrcCaches) {
  (void)srcCacheCount;
  (void)pSrcCaches;
  if (from_handle(deviceHandle) == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  if (from_handle(dstCache) == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  return VK_SUCCESS;
}

}  // namespace kniz
