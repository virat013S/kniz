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

bool compile_compute_program(const std::string& glsl, gl::Uint* program,
                             std::string* error) {
  gl::Uint shader = gl::f.CreateShader(gl::kComputeShader);
  if (shader == 0) {
    *error = "glCreateShader failed";
    return false;
  }

  const gl::Char* source = glsl.c_str();
  gl::f.ShaderSource(shader, 1, &source, nullptr);
  gl::f.CompileShader(shader);

  gl::Int status = 0;
  gl::f.GetShaderiv(shader, gl::kCompileStatus, &status);
  if (status == 0) {
    *error = "GLSL compilation failed: " + shader_info_log(shader);
    gl::f.DeleteShader(shader);
    return false;
  }

  gl::Uint linked = gl::f.CreateProgram();
  if (linked == 0) {
    *error = "glCreateProgram failed";
    gl::f.DeleteShader(shader);
    return false;
  }

  gl::f.AttachShader(linked, shader);
  gl::f.LinkProgram(linked);
  // The program keeps the compiled shader alive, so the object can go now.
  gl::f.DeleteShader(shader);

  gl::f.GetProgramiv(linked, gl::kLinkStatus, &status);
  if (status == 0) {
    *error = "GLSL linking failed: " + program_info_log(linked);
    gl::f.DeleteProgram(linked);
    return false;
  }

  *program = linked;
  return true;
}

// Assign every storage buffer in the pipeline layout a distinct GL binding
// point.  GL has one flat namespace per resource class, while Vulkan's is
// two-dimensional; resources at (set 0, binding 0) and (set 1, binding 0) are
// both legal and would collide if the binding number were used directly.
//
// Compacting rather than computing set * width + binding is deliberate: GL
// bounds indexed storage buffer bindings by
// GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, which can be as low as 8, so the
// numbers must stay dense.
std::vector<GlBinding> build_binding_map(PipelineLayout* layout) {
  std::vector<GlBinding> map;
  uint32_t next_index = 0;

  for (uint32_t set = 0; set < layout->sets.size(); ++set) {
    for (const DescriptorSetLayoutBinding& binding :
         layout->sets[set]->bindings) {
      if (binding.type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
        continue;
      map.push_back(GlBinding{set, binding.binding, next_index++});
    }
  }

  return map;
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
  pipeline->bindings = build_binding_map(layout);

  const ShaderTranslation translation =
      translate_compute(module->spirv, entry_point, pipeline->bindings);
  if (!translation.ok) {
    KNIZ_ERR("could not translate the compute shader: %s",
             translation.error.c_str());
    delete pipeline;
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  pipeline->glsl = translation.glsl;

  KNIZ_DBG("GLSL generated for pipeline (%zu storage buffer binding(s)):\n%s",
           pipeline->bindings.size(), pipeline->glsl.c_str());

  std::string error;
  if (!compile_compute_program(pipeline->glsl, &pipeline->program, &error)) {
    KNIZ_ERR("%s", error.c_str());
    KNIZ_ERR("the GLSL that failed was:\n%s", pipeline->glsl.c_str());
    delete pipeline;
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  *out = to_handle(pipeline);
  KNIZ_INFO("compute pipeline created: GL program %u, %zu storage buffer "
            "binding(s)",
            pipeline->program, pipeline->bindings.size());
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
  (void)pipelineCache;
  (void)pCreateInfos;
  (void)pAllocator;
  (void)deviceHandle;
  for (uint32_t i = 0; i < createInfoCount; ++i)
    pPipelines[i] = VK_NULL_HANDLE;
  KNIZ_ERR("graphics pipelines are not implemented yet (M2); compute only");
  return VK_ERROR_FEATURE_NOT_PRESENT;
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
