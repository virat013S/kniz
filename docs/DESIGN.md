# Kniz — Design Notes

## 1. Premise

Kniz is a Vulkan ICD. The Vulkan loader calls into us; we implement
`VkInstance`/`VkDevice`/`VkQueue`/etc. by driving a desktop OpenGL 4.x core
context that we own.

The direction matters enormously. Zink (OpenGL → Vulkan) translates *upward*
into a more expressive API, which is why it can reach ~95% of native GL and
sometimes beat a native GL driver outright. Kniz translates *downward* into a
less expressive API. Every Vulkan guarantee that GL cannot represent is
**destroyed, not emulated**. This is not a matter of effort; the information is
not present in the target API.

The consequence: set expectations at *correctness for simple workloads*, not
performance, and never parity with Zink.

## 2. The fidelity budget

This table is the project's real constraint. Anything in the "Loss" column is
work we can do; anything in the "Lost" column is unfixable.

| Vulkan construct | Nearest GL equivalent | Verdict |
|---|---|---|
| `VkCommandBuffer` recording | *(nothing — GL core cannot record)* | **Emulated** via our own IR, replayed at submit. Costs CPU. |
| `vkCmd*` ordering | immediate GL calls on replay | OK |
| Render pass, load/store ops, tiling | FBO + `glClear` / `glInvalidateFramebuffer` | Partial — tiling is hidden by GL; load/store can only be approximated |
| Subpasses, input attachments | none | **Lost** — must round-trip through a texture. Bandwidth blowup. |
| `VkPipeline` (PSO) | GL 4.1 SSO + `glProgramUniform` + state | Good |
| Descriptor sets, `VkDescriptorSetLayout` | flat GL binding points, rebound at draw | Partial |
| Descriptor indexing / non-uniform indexing | `ARB_bindless_texture` | **Lost** on drivers without bindless (check `iris`) |
| SPIR-V | GLSL via SPIRV-Cross | **Lossy** — no GLSL form for 8/16-bit types, storage-image formats, some group ops |
| `VkImage` layout + barriers | none; implicit | Mostly harmless; emit `glMemoryBarrier` where coherence matters |
| Fences, binary semaphores, events | `GLSync` / synchronous replay | Approximate |
| Timeline semaphores | none | **Lost** — needs an emulation layer |
| `VkDeviceMemory` heaps, types, `vkMapMemory` | none | **Faked** — shadow object + lazy GL allocation |
| Multiple queues, cross-queue sync | one context per thread | **Lost** |
| Sparse resources, residency | none | **Lost** |
| Queries, timestamps | `GL_TIME_ELAPSED`, occlusion queries | Good |
| Geometry/tessellation stages | directly supported | Good |
| `VK_KHR_ray_tracing`, mesh shaders | none | **Lost** |

Report capabilities conservatively. Over-reporting a feature we emulate badly
produces corruption that is far harder to debug than a clean
`VK_ERROR_FEATURE_NOT_PRESENT`.

## 3. Architecture

### 3.1 ICD bootstrap

Export `vk_icdGetInstanceProcAddr`, `vk_icdGetPhysicalDeviceProcAddr`,
`vk_icdNegotiateLoaderICDInterfaceVersion`. Register via the manifest in
`icd/kniz_icd.json` selected with `VK_ICD_FILENAMES`.

We register as an *additional* device, not a replacement, so `vulkan-intel`
stays available for comparison.

### 3.2 Object model

Vulkan handles are opaque. On 64-bit we point them at our own structs.
Dispatchable handles (`VkInstance`, `VkDevice`, ...) need a leading
`VkLayerInstanceDispatchTable`-style dispatch pointer so the loader can route;
non-dispatchable handles are just `uint64_t` wide enough to hold a pointer.

Keep a per-instance registry of live objects so `vkGetPhysicalDeviceProperties`
etc. can be answered and so validation-style checks are possible.

### 3.3 GL context ownership

- We own the GL context, not the app.
- Start with **EGL**, `EGL_PLATFORM_SURFACELESS_MESA` / `EGL_KHR_surfaceless_context`,
  to get a context with no window. This keeps M0–M2 headless and testable.
- GL state is **per-thread and current-state**. Vulkan callers are
  thread-agnostic. So: one context per Vulkan caller thread, and
  `eglMakeCurrent` bracketing every entrypoint. A `ThreadContext` RAII guard
  around each call.
- This is the single biggest structural difference from Zink: Zink sits inside
  Mesa and can lean on Gallium's context handling. We cannot.

### 3.4 Command buffers as display lists (ADR-1)

GL core cannot record commands. Three options:

1. **Execute immediately in `vkCmd*`.** Breaks reuse and resubmission semantics,
   and breaks recording a command buffer on a different thread from submission.
   Rejected.
2. **`NV_command_list`.** NVIDIA-only. Rejected.
3. **Record into our own IR; replay in `vkQueueSubmit`.** Chosen.

The IR is a flat vector of tagged command structs. Replay makes the GL context
current and walks it. Because replay is synchronous:

- Barriers, layout transitions and most semaphore waits collapse to no-ops.
- `VkFence` becomes a simple completion flag, optionally backed by a `GLSync`.
- We get one well-defined point to implement state-shadowing and redundant-state
  elimination — which is where the CPU cost must be clawed back.

### 3.5 Descriptor sets

Vulkan's set/binding namespace is flattened to GL binding indices, and the whole
set is re-bound at draw time (GL has no "bind descriptor set"). Use
`glBindBufferRange(GL_UNIFORM_BUFFER, ...)` / `GL_SHADER_STORAGE_BUFFER`,
`glBindTextureUnit`, `glBindSampler`.

Cache aggressively: rebinding every descriptor on every draw is the classic way
to lose an order of magnitude of CPU time. Track dirty groups.

Bindless (`ARB_bindless_texture`) is the escape hatch for descriptor indexing
where the driver supports it; needs a feature query and a fallback.

### 3.6 Shaders

SPIR-V → GLSL via **SPIRV-Cross**, then compile with the GL driver and cache by
(SPIR-V hash, specialization constants, stage). `spirv-val` on input for early
diagnosis. Expect to maintain a translation-failure list; those are real bugs
against real apps.

### 3.7 Memory

`VkDeviceMemory` is a shadow record. Actual GL buffers/textures are allocated
lazily at `vkBindBufferMemory` / `vkBindImageMemory`, since until binding we
don't know the shape.

- Buffers: `glBufferStorage` with `GL_MAP_PERSISTENT_BIT|GL_MAP_COHERENT_BIT`
  for persistently mappable memory.
- Images: real GL textures; `vkMapMemory` on an image backs a CPU shadow with
  flush on unmap.
- Report one unified `DEVICE_LOCAL|HOST_VISIBLE` heap (true for the iGPU) and
  over-report `memoryTypeCount`. Apps handle "more" better than "less".

### 3.8 WSI

Late, deliberately. In ICD mode *we* implement `vkCreateXlibSurfaceKHR`, so we
receive the app's `Display` and `Window`; then `glXCreateWindow` + a GL context
bound to that surface, and present via `glXSwapBuffers`. Wayland needs the EGL
path. Deferred until render passes and pipelines are known good, so we're never
debugging WSI and drawing at the same time.

## 4. Milestones

| | Goal | Proves | Gate |
|---|---|---|---|
| **M0** | Manifest + `vk_icdGetInstanceProcAddr`; `vulkaninfo` lists our device; headless EGL context created | Loader handshake | `vulkaninfo --summary` shows Kniz |
| **M1** | Offscreen compute end-to-end, checked against a CPU reference | SPIR-V→GLSL, memory, descriptors — the whole risky spine | numeric test passes |
| **M2** | Offscreen graphics: render pass → FBO, PSO → program, triangle → image | Passes, pipelines, state | image matches reference |
| **M3** | GLX/EGL swapchain | WSI | `vkcube` on screen |
| **M4** | Fences, events, timestamps, queries | Correctness | sample suite runs |

M1 deliberately bundles the three scariest subsystems so failure is cheap and
early.

## 5. Non-goals

CTS conformance. Multi-queue correctness. Sparse resources. Timeline
semaphores. Ray tracing and mesh shaders. Any claim of performance parity —
see §1.
