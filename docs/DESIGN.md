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
| `VkImage` + `vkMapMemory` | GL texture, no host mirror | **Partial** — every read and write goes through a command; `LINEAR` tiling and mapping an image's allocation are refused outright (see §4, M2a) |
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
`vk_icdNegotiateLoaderICDInterfaceVersion` and `vkGetInstanceProcAddr`. Register
via the manifest in `icd/kniz_icd.json`, selected with `VK_DRIVER_FILES` — and
with `VK_ICD_FILENAMES` as well, since the loader honours the former and then
ignores the latter, so setting only one of them is not enough on a machine that
already pins a driver.

We register as an *additional* device, not a replacement, so `vulkan-intel`
stays available for comparison.

Negotiate ICD interface version 5. Version 6 and 7 demand behaviour we cannot
honour yet, and a newer loader correctly falls back when we report the lower
number.

**Export control is load-bearing, not cosmetic.** An ICD shares an address space
with the application and with the loader, and Kniz additionally links EGL and
resolves OpenGL entry points at runtime — so the process frequently already has a
GL stack loaded when the loader `dlopen`s us. Two layers:

1. compile with `-fvisibility=hidden`, so intra-library references bind at link
   time;
2. a version script (`src/kniz.map`) confining the dynamic symbol table to `vk*`.

Layer 1 is not optional. Without it, taking the address of an internal function
from another translation unit in the same library emits a *dynamic* reference to
a symbol the version script has just hidden. That links cleanly, because
unresolved symbols in a shared object are permitted by default, and then fails at
`dlopen`. `-Wl,-z,defs` turns the whole class of mistake into a link error, and
is enabled for that reason.

### 3.2 Object model

Vulkan handles are opaque. On 64-bit we point them at our own structs.

**The first word of every object we hand back is not ours.** The loader stores a
pointer to its own dispatch table there, and it requires that word to have been
initialised with `ICD_LOADER_MAGIC` (via `set_loader_magic_value` from
`vulkan/vk_icd.h`) beforehand, which it asserts on in debug builds. In the
loader's own words, from `terminator_CreateDevice`:

```c
*pDevice = dev->icd_device;
loader_init_dispatch(*pDevice, &dev->loader_dispatch);   // writes offset 0
```

So every dispatchable object begins with a `VK_LOADER_DATA` we never read, and
our own bookkeeping starts immediately after it. This includes `VkQueue`: it is
declared non-dispatchable, but the loader's `vkGetDeviceQueue` trampoline still
calls `loader_set_dispatch(*pQueue, disp)` on it.

Getting this wrong does not fail loudly. The handle is valid at the moment
`vkCreateDevice` returns it and has been silently overwritten by the time the
next call arrives, so the driver appears to work and then rejects its own device
— or jumps through address zero. The magic value used to recognise a handle must
therefore live *after* the reserved word, never at offset 0.

Everything else follows normally: allocate objects individually, return stable
handles (a physical device's handle must not move between calls to
`vkEnumeratePhysicalDevices`), and validate each incoming handle by its magic
before dereferencing.

### 3.2.1 Entry point lookup

Two functions, and the loader uses both:

| Function | Returns |
|---|---|
| `vk_icdGetInstanceProcAddr` | global and instance commands, plus device commands it wants terminated |
| `vkGetDeviceProcAddr` | device commands |

Keeping one flat table and letting each entry point search it is deliberately
more permissive than the specification requires, and matches what real ICDs do:
applications call `vkGetInstanceProcAddr` for device commands and expect a usable
pointer.

**`vkGetDeviceProcAddr` must not validate its device handle.** The loader calls
it while assembling its device dispatch table, and the value it passes at that
point is the *application-visible* handle — whose first word is the loader's
dispatch pointer, not our object — rather than the `VkDevice` we returned. A
magic check there rejects every lookup and leaves the loader with a dispatch
table full of nulls. The registry below is what catches genuine misuse instead.

### 3.2.2 Validation

Keep a per-instance registry of live objects so `vkGetPhysicalDeviceProperties`
etc. can be answered and so validation-style checks are possible. Validation
belongs in the commands that do work, not in the lookup functions.

### 3.3 GL context ownership

- We own the GL context, not the app.
- Two headless paths, tried in order: the **default display with a pbuffer**
  (the normal desktop case) and **`EGL_PLATFORM_SURFACELESS_MESA`** (containers,
  CI, no X11 or Wayland). Supporting both is what lets the test suite run on a
  machine with no display at all.
- `EGL_SURFACE_TYPE` must be present in `eglChooseConfig` **even for a
  surfaceless display**. Omitting it does not mean "any": Mesa then matches zero
  configs and reports it as `EGL_SUCCESS` with a count of zero, so the error code
  offers no clue. Requesting `EGL_PBUFFER_BIT` constraints is correct for both
  display kinds; only surface creation differs.
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

### 3.9 Build portability

Kniz is meant to be cloned and built on any distribution, so the build is
deliberately plain:

- **No code generation.** Vulkan entry points are declared, not generated, and
  GL entry points are resolved at runtime through `eglGetProcAddress`. That
  removes GLAD, Python, Jinja and a generated loader from the dependency list —
  fewer moving parts, and one less thing to break on a distribution we have never
  tested.
- **Two-tier dependencies.** Check the distribution's copy first
  (`find_package`, `pkg-config`), fall back to a pinned download. Packagers get to
  use system libraries; a bare clone still builds. `KNIZ_FETCH_DEPS=OFF` forbids
  the download so that a packaging mistake is a configure error rather than a
  silent network access.
- **No `sudo`, no distribution-specific recipe.** The dependency list is a
  compiler, CMake, Ninja and EGL headers.
- **The test runs through the loader and needs no Vulkan loader development
  package** — it `dlopen`s `libvulkan.so.1` and resolves `vkGetInstanceProcAddr`
  from it. So the only link-time requirement is the Vulkan headers, and the test
  is a real integration test rather than a unit test of our own code.
- **Both loader selection variables are always set, never one.** The loader obeys
  `VK_DRIVER_FILES` when it is set and then ignores `VK_ICD_FILENAMES` entirely.
  A machine that pins a vendor driver with the newer variable therefore redirects
  every test and every `kniz` invocation to the system's own driver no matter what
  the older variable says, and the result looks like a driver bug rather than a
  stale environment. The launcher, the tests and `kniz-info` all set and report
  both, and the tests identify the driver by PCI vendor ID rather than trusting
  any variable at all — a test that ran against ANV would otherwise pass while
  proving nothing.
- CI builds in Ubuntu, Fedora and Arch containers, which between them exercise
  both dependency paths, and additionally in `-DKNIZ_FETCH_DEPS=OFF` mode.

## 4. Milestones

| | Goal | Proves | Gate |
|---|---|---|---|
| **M0** ✅ | Manifest + `vk_icdGetInstanceProcAddr`; `vulkaninfo` lists our device; headless EGL context created | Loader handshake | `vulkaninfo --summary` shows Kniz |
| **M1** ✅ | Offscreen compute end-to-end, checked against a CPU reference | SPIR-V→GLSL, memory, descriptors — the whole risky spine | numeric test passes |
| **M2a** ✅ | Images: creation, memory binding, clear, upload, readback | The first object GL cannot mirror, and the layout asymmetry that comes with it | byte-exact round trip |
| **M2b** | Offscreen graphics: render pass → FBO, PSO → program, triangle → image | Passes, pipelines, state | image matches reference |
| **M3** | GLX/EGL swapchain | WSI | `vkcube` on screen |
| **M4** | Fences, events, timestamps, queries | Correctness | sample suite runs |

M0 is done and was checked on the development machine (Intel HD 4400 / Mesa
26.2.4): the loader negotiated ICD interface version 5 and `vulkaninfo --summary`
reported `Kniz on Mesa Intel(R) HD Graphics 4400 (HSW GT2)`. The same test passes
with no display present, via the surfaceless EGL path.

M1 is done and passed the same way. Its test (`tests/compute.cpp`) records a
dispatch over two storage buffers and compares all 1024 elements against a C++
reference; it also covers `vkCmdFillBuffer`, `vkCmdUpdateBuffer` and
`vkCmdCopyBuffer`, and asserts that a graphics pipeline is *refused* rather than
accepted and ignored. Three things M1 settled, each of which could have been
fatal:

- **SPIR-V→GLSL works, but needs the pipeline layout.** A `VkShaderModule` alone
  is not enough to translate: SPIR-V addresses resources by `(set, binding)`,
  GLSL by one flat binding number, and two sets may legally share a binding
  number. So translation happens at *pipeline* creation, where the layout is
  known, and the resulting `(set, binding) → GL index` table is stored on the
  pipeline. The generated GLSL and the dispatch path read the same table, which
  is what stops them disagreeing. Indices are compacted rather than computed as
  `set * width + binding`, because `GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS` can be
  as low as 8.
- **The emulated `VkDeviceMemory` round trip is sound.** An allocation is host
  memory; buffers copy in and out of it around each submission, keyed off a
  version counter bumped on unmap. The invariant that makes it correct is that
  the host block is a *mirror*: results are copied back after every submission,
  so a later upload cannot resurrect stale bytes.
- **Constant transcription is a real hazard.** `GL_SHADER_STORAGE_BUFFER` is
  `0x90D2`; `0x90D3` is `GL_SHADER_STORAGE_BLOCK`. The wrong value produced
  `GL_INVALID_ENUM` from `glBindBufferRange` and a dispatch that silently wrote
  nothing. Reading the error back is what found it, which is why the replay path
  drains `glGetError` around binding and dispatch rather than trusting itself.

M1 deliberately bundled the three scariest subsystems so failure was cheap and
early.

M2 was split in two once it became clear that "images" and "drawing" share almost
no machinery. **M2a** is images and the data movement to and from them; **M2b** is
render passes, graphics pipelines and draws. M2a is done: its test
(`tests/image.cpp`) round-trips a 64×32 image through a buffer and back and
requires the bytes to be identical, clears an image and requires the exact texels
the float clear value denotes, copies a sub-rectangle out of a buffer whose rows
are deliberately padded and requires every texel inside it to be right and every
texel outside it to be untouched, and asserts the refusals hold.

Three things M2a settled, each of which changed the design:

- **An image has no host mirror, and cannot have one.** A buffer's contents have a
  single representation — bytes — so Kniz can keep a copy and shuffle it in and
  out of GL around each submission. An image has two: the layout Vulkan
  describes, and whatever GL stores, which is opaque and unspecified. There is no
  mapping between them, so an image is backed *only* by its GL texture and every
  read and write goes through a recorded command. Two consequences are therefore
  enforced rather than documented and forgotten:
  - `VK_IMAGE_TILING_LINEAR` is refused. A linear image's layout is *observable*
    — `vkGetImageSubresourceLayout` exists to report it — and GL cannot be told to
    use that layout, so the only honest answer is no. This removes host mirroring,
    mapping, and subresource-layout fidelity problems in one stroke, at the cost
    of a tiling that interactive applications do not use.
  - `vkMapMemory` is refused when any image is bound to the allocation. The
    allocation is still host memory; it just does not hold the image's pixels, so
    a pointer into it would show the application bytes that are not the image.
    Suballocating an image together with a buffer is consequently not yet
    supported.
- **No y flip is needed, and "fixing" that would be a bug.** Vulkan's framebuffer
  origin is upper-left and GL's is lower-left, which sounds like a flip is
  required somewhere; in the other direction, GL→Vulkan, which is Zink's problem,
  it is. Here it is not. GL maps clip space to window coordinates as
  `y_w = (y_d + 1) * h / 2` counted up from the lower left, Vulkan as
  `y_f = (y_d + 1) * h / 2` counted down from the upper left, so *both* put
  `y_d = -1` in row 0 of the image. GL's row 0 is its bottom row and Vulkan's is
  its top row, but that is only a difference in which word describes it, and
  everything that consumes these images — readback, and in time sampling and the
  swapchain — does so through Vulkan's convention. Vulkan image row *r* and GL
  texture row *r* hold the same content.
- **The host mirror and the GL buffer object must be updated together.** The first
  version of `vkCmdCopyImageToBuffer` wrote the copied bytes into the buffer's
  host mirror and stopped there, on the reasoning that the mirror is what the
  application maps. Submission then copies the GL buffer object back over the
  mirror for every buffer the GPU may have written — and the object had never
  been told about the bytes — so the readback returned zeros. Both representations
  are now written. The general rule is worth stating because it will recur: *any*
  command that produces bytes on the host side has to push them to the object as
  well, or the next `sync_buffers_from_gl` will discard them.

M2a also fixed a latent M1 bug that the same reasoning exposed: host writes made
while memory is *mapped* were only noticed on unmap, because the version counter
moved only then. Vulkan lets an application keep an allocation mapped across a
submission and write through the pointer it holds, so submission now re-uploads
any allocation that is mapped at all, whether or not its version changed.

What M2a does *not* cover, and M2b has to add: uniform buffers, push constants,
textures and samplers, render passes, framebuffers, graphics pipelines. All of
them are refused with an explicit error today, at the earliest point the
application reveals it needs one. M2a's own remaining costs, to be revisited:
image-only allocations still allocate a full host block that the image does not
use, a sub-region readback of a mip level costs a temporary of the whole level
(`glGetTexImage` reads a level, not a rectangle), an sRGB view of a UNORM
image is refused rather than silently sampled with the transfer function missing,
and GL's filter and wrap state belongs to a texture while Vulkan's belongs to a
sampler — a real difference that M2b has to place somewhere.

## 5. Non-goals

CTS conformance. Multi-queue correctness. Sparse resources. Timeline
semaphores. Ray tracing and mesh shaders. Any claim of performance parity —
see §1.
