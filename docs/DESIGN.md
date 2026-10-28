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

A related correction, stated once so it does not have to be re-derived: neither
project upgrades anything, because an API is not where performance lives — the
driver at the bottom is. Zink exists because GL drivers stopped being where
investment goes, so it glues the old API onto the maintained Vulkan drivers.
Kniz exists where the arrangement is reversed: machines with a capable GL
driver and no Vulkan driver at all — Intel hardware below Broadwell, the Haswell
GT2 of the development workstation among them — where the GL path is not the
slow path but the fast *present* one. A Vulkan application on that machine has
two options: no GPU at all, or Kniz borrowing the iris driver. The performance
budget is therefore the translation, not the driver: shrink the per-draw
overhead (§6) far enough and a supported tier game plays on borrowed GL as if
the platform had shipped Vulkan.

## 2. The fidelity budget

This table is the project's real constraint. Anything in the "Loss" column is
work we can do; anything in the "Lost" column is unfixable.

| Vulkan construct | Nearest GL equivalent | Verdict |
|---|---|---|
| `VkCommandBuffer` recording | *(nothing — GL core cannot record)* | **Emulated** via our own IR, replayed at submit. Costs CPU. |
| `vkCmd*` ordering | immediate GL calls on replay | OK |
| Render pass, load/store ops, tiling | FBO + `glClearBufferfv` | **Partial** — the load op is honoured and the render area is enforced; the store op is unobservable because GL always stores, and tiling does not exist |
| Depth attachment, depth load op | a depth texture on the FBO, `glClearBufferfv(GL_DEPTH, …)` or `glClearBufferfi` for a packed format | Good for D16 — clear, load and store honoured, and the value a fragment leaves is Vulkan's window depth exactly thanks to the clip space fixup; wider formats and stencil are refused (§3.14) |
| Depth test, write mask, compare op | `glEnable(GL_DEPTH_TEST)`, `glDepthMask`, `glDepthFunc` | Good, but the no-test case has to be mapped to `GL_ALWAYS`: GL stops updating the depth buffer entirely once the test is disabled, and Vulkan does not (§3.14) |
| Depth range from `VkViewport` | `glDepthRange` from the recorded viewport | Good — identity mapping, because the fixup makes NDC z carry Vulkan's convention |
| Subpasses, input attachments | none | **Lost** — must round-trip through a texture. Bandwidth blowup. |
| Render area | the scissor rectangle | Good — intersected with the pipeline's own scissor, so a fragment outside it is discarded the way the specification requires |
| `VkPipeline` (PSO) | a GL program, a VAO and context state | Good — immutable and shareable as Vulkan requires, but GL state lives in the context, so a pipeline has to be re-applied at every draw |
| Descriptor sets, `VkDescriptorSetLayout` | flat GL binding points, rebound at draw | Partial — one flat namespace per resource class, and only the bindings the shaders read have to be bound (§3.5) |
| Uniform buffers | `glBindBufferRange(GL_UNIFORM_BUFFER, …)` | Good — the layout and the shader rewrite share one table, so they cannot disagree |
| Push constants | a driver-owned uniform buffer, uploaded as the commands replay | **Emulated** — exact for a block std140 can express, refused for one it cannot (§3.11) |
| `VkSampler` | `glGenSamplers` / `glBindSampler` | Good — GL separates filter state from the image exactly as Vulkan does, which is what makes two samplers over one image possible |
| Sampled images | `GL_TEXTURE_2D` bound to a texture unit | Partial — 2D, level 0, one layer; a separate image and sampler have no GLSL form and are combined when the shader is translated (§3.12) |
| Comparison samplers, anisotropic filtering | `sampler2DShadow`, `ARB_texture_filter_anisotropic` | Expressible, not implemented — refused at sampler creation |
| Sampling mip levels above 0 | the same texture, `GL_TEXTURE_MAX_LEVEL` | **Lost** until the transfer path covers mip levels, because only level 0 is ever written (§4, M2d) |
| Descriptor indexing / non-uniform indexing | `ARB_bindless_texture` | **Lost** on drivers without bindless (check `iris`) |
| SPIR-V | GLSL via SPIRV-Cross | **Lossy** — no GLSL form for 8/16-bit types, storage-image formats, some group ops |
| Clip space depth | `gl_Position.z = 2z - w`, injected by SPIRV-Cross | Good — needed because Vulkan's depth range is `[0, w]` and GL's is `[-w, w]`; see §4, M2b |
| `VkImage` + `vkMapMemory` | GL texture, no host mirror | **Partial** — every read and write goes through a command; `LINEAR` tiling and mapping an image's allocation are refused outright (see §4, M2a) |
| `VkImage` layout + barriers | none; implicit | Mostly harmless; emit `glMemoryBarrier` where coherence matters |
| `VkSurfaceKHR`, `VkSwapchainKHR` | an EGL window surface plus a list of GL textures | **Emulated** — GL cannot make a texture a window's front buffer, so presentation is a draw from one to the other (§3.13) |
| `vkQueuePresentKHR` | a textured full-screen triangle, then `eglSwapBuffers` | Good for one format and FIFO; synchronous, so there is no queue to drop or skip |
| Surfaces on a second window system | `wl_egl_window` for Wayland | Expressible, not implemented — a GL context lives on one EGL display, so the platform is fixed when the context is created (§3.13) |
| Fences, binary semaphores, events | `GLSync` / synchronous replay | Approximate |
| Timeline semaphores | none | **Lost** — needs an emulation layer |
| `VkDeviceMemory` heaps, types, `vkMapMemory` | none | **Faked** — shadow object + lazy GL allocation |
| Multiple queues, cross-queue sync | one context per thread | **Lost** |
| Sparse resources, residency | none | **Lost** |
| Queries, timestamps | `GL_TIME_ELAPSED`, occlusion queries | Good |
| Geometry/tessellation stages | directly supported by GL | Expressible in GL, not implemented in Kniz |
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
set is re-bound at every draw and dispatch (GL has no "bind descriptor set"), with
`glBindBufferRange` on `GL_UNIFORM_BUFFER` or `GL_SHADER_STORAGE_BUFFER` for
buffers and `glActiveTexture` / `glBindTexture` / `glBindSampler` for the texture
units an image or sampler descriptor owns.

The flattening was M2c's main discovery, and three things about it are worth
stating because the obvious implementation is wrong:

- **Resources are numbered within their class, not across all of them.** GL keeps
  one flat namespace per resource class — uniform buffers and shader storage
  buffers are indexed independently — so a layout whose set 0 has a uniform block
  at binding 0 and an SSBO at binding 1 must emit GLSL with the block at 0 and the
  buffer at 0 as well. Numbering uniformly instead still *works*, because the
  rewrite and the replay agree with each other, but it wastes binding points from
  a namespace that can be as small as 8.
- **The layout owns the mapping, because the mapping is a property of the layout.**
  `PipelineLayout` computes it once and the shader rewrite and the replay both read
  it, so they cannot drift. Two resources at (set 0, binding 0) and (set 1,
  binding 0) are both legal in Vulkan and would collide if the binding number were
  used directly; that shape is what the M2c test is built around, and a driver that
  got it wrong would answer one shader's reads from the other's buffer.
- **Only the bindings a pipeline's shaders actually declared have to be bound.**
  `used_bindings` is the union of what the translated stages reported, not the
  layout's full list, so a layout shared between pipelines does not force every one
  of them to write descriptors it never reads. A binding that *is* read and was
  never written is refused at draw or dispatch time rather than bound from
  whatever the previous command left behind.

The offsets GL wants are checked against the live driver rather than assumed.
`GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT` and `GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT`
are queried at context creation, and a descriptor offset that is not a multiple of
one is refused with the alignment in the message. Vulkan caps its equivalent
limits at 256, so a driver asking for more would be a genuine conflict; the
context reports the value on the log line it prints at startup so that it is
visible rather than inferred.

Still to do, and still the plan: cache aggressively, since rebinding every
descriptor on every draw is the classic way to lose an order of magnitude of CPU
time, and track dirty groups. Bindless (`ARB_bindless_texture`) remains the escape
hatch for descriptor indexing where the driver supports it; needs a feature query
and a fallback.

### 3.6 Shaders

SPIR-V → GLSL via **SPIRV-Cross**, then compile with the GL driver and cache by
(SPIR-V hash, specialization constants, stage). `spirv-val` on input for early
diagnosis. Expect to maintain a translation-failure list; those are real bugs
against real apps.

Translation happens at pipeline creation, because that is the first point where
the pipeline layout is known (see §4, M1), and the translation options are set
explicitly rather than left to their defaults:

- `version = 450`, `es = false`, `vulkan_semantics = false` — we emit desktop
  GLSL, and the GL commands are recorded explicitly rather than left to GL's
  implicit ordering.
- `vertex.fixup_clipspace = true`. This rewrites `gl_Position.z` to `2z - w`.
  Vulkan's clip space runs z from 0 to w and GL's from −w to w, so without the
  rewrite a vertex at Vulkan's near plane lands halfway through GL's depth range
  and everything in front of it sorts behind it. M2b has no depth attachment to
  test it against, but it does have `gl_FragCoord.z`: the M2b test's second
  vertex shader writes depth 0.25 and the fragment shader writes `gl_FragCoord.z`
  back out, so 0.25 is observed exactly. Without the rewrite the same vertex
  would read 0.625, which is 0.25 of the way through GL's `[-w, w]` range.
- `vertex.flip_vert_y = false`. The one thing that *would* need flipping is the
  one thing that must not be flipped; the argument is in §4, M2b.
- `emit_push_constant_as_uniform_buffer = true`, which is what makes a push
  constant block expressible at all: GL has no push constants, so the block is
  emitted as an ordinary `uniform` block and the driver uploads to it (§3.11). The
  option only changes how the block is *declared*; the binding point it is given is
  set by Kniz, as with every other resource.

Sampled images get one extra step before anything is emitted: separate images and
samplers have no GLSL form, so `build_combined_image_samplers()` pairs them up and
the dummy sampler `texelFetch` needs is injected past the end of the layout's
texture units. §3.12 is the argument for the units each of them is given.

A buffer block is declared in the GLSL with an explicit layout, and `std140` is the
only one that is always available — `std430` is not permitted for a uniform block,
and the scalar layout that needs no padding at all is an extension. SPIRV-Cross
tries the tighter standards and, failing all of them, refuses the block with
"Buffer block cannot be expressed as any of std430, std140, scalar, even with
enhanced layouts." That is a real limitation rather than a translation bug — it is
reachable from legal SPIR-V, for instance a push constant block holding a `float`
array, where Vulkan's rules put the elements 4 bytes apart and std140 puts them 16
— so Kniz appends its own sentence to the error saying which construct could not be
expressed and why, rather than passing the message through unexplained.

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

Late, deliberately: deferred until render passes and pipelines were known good, so
that WSI and drawing were never being debugged at the same time. The original
plan was GLX — take the app's `Display` and `Window` and swap buffers on them —
which would have needed a second GL context on the window and a share group to
reach the objects the headless context owns. EGL gives the same context a window
drawable instead, which needs no sharing and no second context, so that is the
path that was taken. Implemented in M3; see §3.13 for what it actually does and
what it refuses.

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

### 3.10 Rendering

A render pass becomes a framebuffer object, an attachment becomes a texture
attached to it, and the load operation becomes `glClearBufferfv`. Three details
are worth recording because they are the ones that were not obvious.

- **The clear is indexed by subpass slot, not by attachment index.**
  `glClearBufferfv(GL_COLOR, i, ...)` clears colour attachment *i of the currently
  bound framebuffer*, which is the subpass's `i`th colour attachment. A render
  pass whose colour attachment 0 is attachment 3 of the render pass clears slot 0
  but tags attachment 3. Kniz stores the subpass slot with the attachment for
  exactly this reason.
- **`vkCmdEndRenderPass` unbinds the framebuffer.** GL declares reading a texture
  that is attached to the currently bound framebuffer undefined, so a read that
  followed a pass without unbinding could return the previous pass's contents
  rather than the texture's. The unbind is followed by
  `glMemoryBarrier(GL_ALL_BARRIER_BITS)`, which is the same over-broad barrier the
  command replay already uses between dependent commands.
- **The render area is a second scissor, not just a description of the clear.**
  Vulkan discards every fragment outside the render area, and applications use
  that: a pass over half a framebuffer must leave the other half alone. The
  pipeline's scissor — its own, or the dynamic one set by `vkCmdSetScissor` — is
  therefore intersected with the render area at the point the draw is recorded,
  and the intersection is what ends up in `glScissor`.
- **The colour write mask is reset before the load operation.** A Vulkan load-op
  clear is not pipeline state, so a pipeline's write mask cannot restrict it,
  whereas GL keeps the mask in the context next to every draw. Whether
  `glClearBufferfv` consults the mask is not something to depend on either way —
  Gallium carries a capability flag for exactly this ambiguity, because the
  hardware underneath differs — so the mask is set explicitly at the start of
  every pass. Dithering is switched off for a related reason: Vulkan has none, and
  it perturbs a clear value in its least significant bit rather than writing it
  exactly.

A graphics pipeline is a GL program, a vertex array object, and a record of the
state to apply. GL keeps that state in the context rather than in the object, so
the pipeline is re-applied at every draw; the object itself stays immutable and
shareable, which is what Vulkan promises. The VAO is created per pipeline and
re-pointed at every draw. `GL_ELEMENT_ARRAY_BUFFER` is deliberately left bound
when the VAO is set up: unbinding it while the VAO is bound would erase it from
the VAO, and it is part of the VAO's state.

Vertex formats come from a small table — 32- and 16-bit floats, and 8- and
16-bit normalised and scaled integers. Integer vertex formats are absent on
purpose: they need `glVertexAttribIPointer` rather than `glVertexAttribPointer`,
and half of that distinction implemented silently converts values. `B8G8R8A8`
is absent for a different reason: GL has no BGRA vertex attribute type at all.

What is refused, each with a message naming the construct: more than one subpass,
a depth format in a colour attachment slot (and the converse in the depth slot),
an input attachment, a resolve attachment, stencil test, blending, logic ops,
multisampling, sample shading, alpha-to-coverage, polygon modes other than `FILL`,
depth clamp, rasterizer discard, multiple viewports or scissors, primitive
restart, instanced vertex input, geometry and tessellation stages, specialization
constants, dynamic state other than `VIEWPORT` and `SCISSOR`,
`VK_CULL_MODE_FRONT_AND_BACK`, integer vertex formats, and a
`VkPipelineColorBlendStateCreateInfo` whose attachment count does not match the
subpass's colour attachments. `vkCmdClearAttachments` is refused too: clearing a
rectangle inside a render pass has no single GL equivalent, because the scissor
would have to be overridden and restored around it, and squaring that with the
pipeline's own scissor deserves to be done deliberately rather than in passing.
Depth attachments, depth test and depth write left this list when M3b landed;
what remains refused about depth — stencil, depth bounds, wide depth formats — is
in §3.14, which is where the depth path is documented.

### 3.11 Push constants

OpenGL has no push constants in any form, and of everything Kniz emulates this is
the one with no partial answer available: a wrong binding usually shows up as a
draw that reads the wrong buffer, whereas a push constant that is dropped produces
a perfectly plausible image built from stale values.

The design is a uniform buffer that the device owns. `VkPipelineLayout` reserves
one GL uniform binding point for the block — the *first* one, ahead of every
descriptor — and the shader's push constant block is rewritten to sit there
(§3.6). `vkCmdPushConstants` is recorded as its own command in the command IR,
carrying the bytes and the layout that gives them meaning, and the replay uploads
them with `glBufferSubData` when it reaches that command. Ordering falls out of the
recording order, which is what Vulkan specifies: the values a shader sees are the
last ones written before the command that reads them, and a pipeline bind in
between does not reset them. That is also why the command records the *layout*
rather than the pipeline: the binding point is a property of the layout, and a
values write is valid across the pipeline binds that may follow it.

Numbering the push constants *first* is what makes that last sentence true in
practice. Vulkan lets an application write push constants with one layout and draw
with another that is merely "compatible for push constants": the two must have the
same push constant ranges, and are otherwise unconstrained, so they may declare
different numbers of uniform buffers. Numbering the descriptors first would make
the reserved index depend on that count, and a pushed value would land where the
drawing shader is not looking. Numbering push constants first makes the index
depend only on whether a layout declares ranges at all, which every compatible
layout agrees about. The remaining asymmetry — a descriptor bind *between* a push
and a draw, in a command buffer that mixes a layout with no push constants into a
sequence that uses one, can overwrite binding point 0 — is recorded among M2c's
remaining costs in §4 rather than quietly tolerated.

Three further details are decisions rather than consequences:

- **The buffer is bound for its whole length, not for the range that was written.**
  A block may declare members the application never wrote, and Vulkan says those
  keep their previous values — which are the ones the buffer already holds.
  Binding the whole buffer makes the shader read exactly those bytes.
- **The buffer is zeroed at the start of every submission.** Vulkan leaves push
  constants undefined at the start of a command buffer, so an application that
  reads before writing is in undefined territory either way; zeroing turns that
  into zeroes rather than into the previous submission's data. 128 bytes is a
  small enough upload to be cheaper than tracking whether the contents are known.
- **The capacity is 128 bytes**, which is what `maxPushConstantsSize` reports. It
  is the Vulkan minimum, and the number has one source so the two cannot disagree.

A push constant range is validated where the application declares it: the stage
flags must name a stage, the size must be non-zero and at most the capacity, the
offset must be a multiple of 4 because Vulkan requires it, and two ranges for a
common stage may not overlap. `vkCmdPushConstants` is validated as well, against
the limit and against a layout that declares no ranges at all — the second of which
Vulkan also calls invalid usage, and which here would otherwise have the shader
read a descriptor's bytes as its constants. A *shader* that reads push constants
against a layout with no ranges is refused at pipeline creation for the same
reason.

The block's bytes are laid out by Vulkan's rules, which are not GLSL's, so a block
that cannot be expressed as a `std140` uniform block is refused at pipeline
creation with the reason appended (§3.6) instead of being emitted with silently
different offsets. The two layouts agree for the common shape — scalars and
vectors, and matrices, all get the same offsets and strides under both sets of
rules — and disagree where a tighter packing is asked for, most visibly for an
array of scalars, whose elements Vulkan places 4 bytes apart and std140 16. So a
push constant block of plain vectors, which is what most applications write, is
expressed exactly; one that needs the tight packing is refused rather than uploaded
with its members read from the wrong place.

### 3.12 Samplers and textures

A Vulkan texture is two objects: an image, which owns the pixels, and a sampler,
which owns the filter and address state. GL can express that split — sampler
objects have existed since 3.3 — and it has to, because the split is observable:
two samplers over one image is a normal thing for an application to do, and any
driver that folds the sampler into the texture answers both reads with one of
them.

So `VkSampler` becomes a `glGenSamplers` object and `vkCmdBindDescriptorSets`
binds it with `glBindSampler` to the texture unit the image occupies. GL then
combines the unit's texture with the unit's sampler, which is exactly the pair
the Vulkan descriptor named.

That leaves the interesting half: GLSL has one sampler type that carries both
halves, and Vulkan's shaders may declare them separately.

- `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` is the easy case. The shader's
  `sampler2D` is rewritten to a texture unit, and the replay binds both the
  texture and the sampler object to it.

- `VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE` plus `VK_DESCRIPTOR_TYPE_SAMPLER` has no
  GLSL form at all. SPIRV-Cross's `build_combined_image_samplers()` synthesises a
  combined sampler for each (image, sampler) pair the shader actually uses, and
  Kniz gives that sampler the *image's* texture unit, because that is where the
  image is bound; the sampler object follows it there. The sampler descriptor's
  own unit is then a unit nothing reads, which is fine and is why sampler
  descriptors do not appear in the pipeline's used-binding list at all.

  The pairing is discovered by the translator and consumed by the replay, so it
  travels in the `GlBinding` the translation returns: `sampler_set` and
  `sampler_binding` name the descriptor whose sampler object belongs on this
  unit. One image read through two different samplers in the same shader would
  need two units for one image, which GLSL cannot express without renaming the
  image; it is refused with an error rather than resolved to one of the two.

- `texelFetch` takes an image and no sampler in Vulkan, and a sampler in GLSL.
  `build_dummy_sampler_for_combined_images()` supplies one, and Kniz decorates it
  with a unit past the end of every descriptor in the layout
  (`PipelineLayout::dummy_sampler_gl_index()`). Nothing needs to be bound there:
  a fetch ignores sampler state, and the unit exists only so the GLSL compiles.
  Reserving it is why a layout's texture-unit count is checked against
  `GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS` with one unit to spare.

Two consequences worth stating plainly, because both are refusals rather than
approximations:

- An image with more than one mip level cannot be sampled. Kniz creates the full
  storage a mipmapped image asks for, but only level 0 is ever written — no copy
  or clear addresses another level — so levels 1 and up hold whatever the driver
  left there. Refusing at the descriptor write is the honest answer until the
  transfer path covers mip levels; sampling level 0 explicitly would work, but
  nothing in the descriptor says the application will only do that.

- A depth image is sampled as a plain value, not as a comparison. Comparison
  samplers need `sampler2DShadow` and `GL_TEXTURE_COMPARE_MODE`, a different
  translation path and a different result type, so `compareEnable` is refused at
  sampler creation. Anisotropic filtering is refused for the same reason: the
  extension is not resolved and the feature is not reported.

### 3.13 Windows, surfaces and swapchains

A Vulkan swapchain is a queue of images the presentation engine will show. GL has
no such object: a window's back buffer belongs to the window system, and a
texture is never that buffer. So a Kniz swapchain is two things joined at present
time — a list of ordinary GL textures and an EGL window surface — and
`vkQueuePresentKHR` is a draw from one into the other followed by
`eglSwapBuffers`.

**The platform is chosen, not inherited.** A GL context lives on exactly one EGL
display and its textures cannot be moved to another, so the window system Kniz
can present on is decided once, when the context is created. Leaving that to
`eglGetDisplay(EGL_DEFAULT_DISPLAY)` is wrong on the most common desktop shape
there is: with both `WAYLAND_DISPLAY` and `DISPLAY` set — a Wayland session
running XWayland — Mesa picks Wayland, and an X11 window handle then means
nothing to it. The first version of this code did exactly that, and the result
was not an error but a segmentation fault inside `eglCreateWindowSurface`: EGL
read the XID as a pointer to a Wayland surface. So `kniz_gl.cpp` now asks for
`EGL_PLATFORM_X11_KHR` explicitly when `DISPLAY` is set and the platform is
advertised, falls back to `EGL_DEFAULT_DISPLAY` with the platform identified
afterwards, and finally to the surfaceless platform. Wayland surfaces are the
next milestone; until then a context that came up on Wayland reports presentation
as unavailable rather than accepting a handle it cannot use.

**A surface owns the drawable; a swapchain owns the images.** The EGL window
surface is created the first time a swapchain needs it and lives in `Surface`,
because it belongs to the window rather than to the images: an application that
resizes destroys its swapchain and creates another on the same surface, and a
second EGL surface for one window is neither necessary nor reliably possible —
Mesa answers the second one with `EGL_BAD_ALLOC`. The images are textures with no
`VkDeviceMemory` behind them, which is what `vkGetSwapchainImagesKHR` handing out
handles rather than objects implies: the application never allocates or binds
memory for them, and `vkDestroyImage` on one is invalid usage, so they are
created and destroyed with the swapchain and appear in no device list. A render
pass that draws into one therefore needs nothing M2b did not already provide.

**Presentation is a draw, not a blit.** `glBlitFramebuffer` would be shorter, but
it requires the source and destination to have compatible formats, and the
window's format is the window system's to choose. A blit would therefore either
refuse configurations that work or swap channels silently — the exact failure
class this project refuses. A textured triangle covering the viewport lets GL
convert on the way out the way it does for every other draw, so the swapchain
format and the window's format need not agree. The triangle is generated from
`gl_VertexID`, so there is no vertex buffer and no attribute to bind.

**The one y flip.** GL's window coordinates count up from the bottom, so texture
row 0 is the *bottom* row of the window, while in Kniz row 0 is Vulkan's *top*
row (§4, M2b). Every other path in this driver draws without flipping, and this
is the single place that flips — in the present fragment shader, as
`1.0 - texcoord.y` — because the window is the only surface whose row order is
the window system's rather than Vulkan's. The M3 test is built so that getting
this wrong cannot pass: the triangle's flat edge and its apex are at opposite
ends of the image.

**Sizes and scaling.** `currentExtent` is reported as `0xFFFFFFFF`, Vulkan's "you
decide", because the window's size is the window system's to decide and asking
would mean a round trip on a connection Kniz deliberately does not hold.
Presentation draws to the window's current size, whatever that is, through a
linear sampler — which is what makes a window manager resizing the window work at
all. On the development machine Hyprland tiled the test's window from 320x240 to
655x674, and presentation scaled to it without any code knowing the window had
changed.

**Refused rather than approximated:** any swapchain format but
`R8G8B8A8_UNORM`; any colour space but `SRGB_NONLINEAR` (an sRGB swapchain needs
the encoding applied by hand to a linear window, and getting that wrong is a
colour that looks almost right); any present mode but FIFO (presentation is
synchronous, so there is no queue to drop or skip); `imageArrayLayers != 1`; any
usage beyond colour attachment and transfer source; any pre-transform but
identity; any composite alpha but opaque; and fewer than two images.

**Synchronisation** follows the rest of the driver. Presentation is synchronous,
so by the time `vkQueuePresentKHR` returns the frame is on its way to the screen
and the next acquire can hand out the next image unconditionally. Acquisition is
round robin; the fence it is given is signalled immediately and the semaphore is
ignored, exactly as submission does (§4, M1).

**`KNIZ_PRESENT_DUMP`.** Presentation is the one result that is normally only
visible to a person, so the driver can be asked to write the frame it is about to
swap to a file as a binary PPM. That is the same buffer `eglSwapBuffers` is
handed, which lets a test compare the presented frame against the image it drew,
pixel for pixel and orientation included, instead of needing a screenshot and an
eye. It costs nothing when unset: the environment is read per present and the
readback path is skipped.

**Costs, recorded rather than discovered later:** the present is a full-screen
draw with a program switch, a VAO bind, a texture bind and a sampler bind per
frame — small, but it is per frame and on the thread that submits;
`eglSwapBuffers` blocks when the compositor is behind, which is the FIFO
behaviour that was asked for and also a stall in the calling thread; recreating a
swapchain recreates its images from scratch, because nothing is shared between an
old swapchain and its replacement; and because the drawable belongs to the
surface, a destroyed window keeps its drawable until the surface is destroyed,
which Vulkan requires the application to do anyway.

### 3.14 Depth attachments, the depth test and depth writes

Depth is the first attachment OpenGL and Vulkan describe almost the same way, and
the first where a disagreement about the *absence* of state rather than the state
itself turns up. What M3b implements and what it refuses:

- **The render pass takes one depth/stencil attachment.** It may be
  `VK_ATTACHMENT_UNUSED` while the colour attachments are listed; the format must
  be one of the depth formats and the image behind the framebuffer must have been
  created with `VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT`, checked at
  framebuffer creation. `LOAD`, `CLEAR` and `DON'T_CARE`-becomes-`CLEAR` are all
  honoured for depth; `stencilLoadOp` is required to match the depth op when the
  format has a stencil aspect, because GL clears the two aspects of a packed
  format with a single `glClearBufferfi` and a stencil-only clear would leave one
  of them wrong.
- **The clear value comes from the render pass, and is observable.** A pass that
  clears depth to 0.5 must reject a draw at 0.75 — if the driver cleared to the
  conventional 1.0 instead of the value the application gave it, the draw lands
  and the rejection is gone. The depth test pins this with a non-default clear.
- **The depth test and write state are per-pipeline, re-applied per draw.**
  `depthCompareOp` maps through an eight-entry table onto `0x0200..0x0207`, which
  are exactly GL's `NEVER..ALWAYS` in enum order. `glDepthRange` is applied from
  the draw's recorded `VkViewport` — the mapping there is the identity, because
  the clip space fixup (§4, M2b) makes NDC z carry Vulkan's `[0, w]` convention,
  so window depth is the number the application wrote.
- **A disabled test still writes, which GL will not do for you.** Vulkan's
  `depthTestEnable = FALSE` with `depthWriteEnable = TRUE` means "never compare,
  keep writing"; OpenGL stops updating the depth buffer the moment
  `GL_DEPTH_TEST` is disabled, on the Intel driver and on llvmpipe alike. Kniz
  maps that case to the test enabled with `GL_ALWAYS` — a comparison that never
  rejects and a write path that runs — and leaves `glDisable(GL_DEPTH_TEST)` for
  the one Vulkan state it correctly describes: no test and no writes. The rule is
  recorded here rather than left in the code because it is the kind of thing the
  GL specification does not put in the sentence about `glDepthMask`, and the phase
  of the depth test that catches it is the reason the mapping is right on both
  the Intel driver and llvmpipe rather than by luck on one.
- **Still refused:** the stencil test, `depthBoundsTestEnable`, depth clamping
  and depth bias, and depth formats wider than 16 bits (`D24_UNORM_S8_UINT`
  excepted from the refusal only until stencil is real). A depth image may be
  created with attachment, transfer-src and transfer-dst usage and nothing else;
  sampled depth is a different API (`sampler2DShadow`) gate and is refused until
  comparison samplers exist.

The format report is the third leg and exists because of vkcube:
`vkGetPhysicalDeviceFormatProperties` was a stub until M3b, returning zero
features for every format. vkcube asks it whether `VK_FORMAT_R8G8B8A8_SRGB` can
be sampled before deciding how to build its texture, takes the linear-tiling
branch when the answer is no — which Kniz refuses — and never creates the optimal
image, then presents a null image view three steps later. The report is now
derived from the same table `vkCreateImage` consults, so the promise and the
behaviour cannot drift apart, and the depth test checks the mirror question on
`D16_UNORM` too: reported as attachable, not reported as sampleable, refused to
be either.

Binary semaphores round the milestone out because vkcube creates them before its
first submit: as objects they exist — created, destroyed, passed around freely —
but they carry no state, because every submission and every present
is synchronous and their only legal observable state window is empty.
`vkCreateSemaphore` with a timeline type is refused at creation, and a timeline
semaphore is one of §7's non-goals.

## 4. Milestones

| | Goal | Proves | Gate |
|---|---|---|---|
| **M0** ✅ | Manifest + `vk_icdGetInstanceProcAddr`; `vulkaninfo` lists our device; headless EGL context created | Loader handshake | `vulkaninfo --summary` shows Kniz |
| **M1** ✅ | Offscreen compute end-to-end, checked against a CPU reference | SPIR-V→GLSL, memory, descriptors — the whole risky spine | numeric test passes |
| **M2a** ✅ | Images: creation, memory binding, clear, upload, readback | The first object GL cannot mirror, and the layout asymmetry that comes with it | byte-exact round trip |
| **M2b** ✅ | Offscreen graphics: render pass → FBO, pipeline → program, draws → image | Passes, pipelines, state, and the coordinate conventions a silently wrong lowering gets wrong | image matches a CPU rasterisation |
| **M2c** ✅ | Uniform buffers and push constants | The descriptor and binding paths, extended from storage buffers to uniforms, and a construct GL has no equivalent for at all | numeric test passes |
| **M2d** ✅ | Samplers and textures | GL's sampler state lives on the texture where Vulkan's lives on a sampler object | numeric test passes |
| **M3a** ✅ | Swapchain and presentation on an X11 window | WSI: surfaces, images the driver owns, and the one legitimate flip | presented frame matches the image |
| **M3b** ✅ | Depth testing, depth writes, binary semaphores; `vkcube` runs | The pipeline state `vkcube` needs before it will draw | The nine-phase depth chain, and `vkcube` on screen |
| **M3c** | Wayland surfaces (`VK_KHR_wayland_surface`) | WSI on a Wayland-only desktop, which several distributions ship as default | `vkcube` under Wayland, swapchain on a `wl_display` |
| **M4** | Fences, events, timestamps, queries | Correctness and sync an indeterminate application assumes exists | sample suite runs |
| **M5** | The smooth tier: per-draw state diffs, descriptor dirty-tracking, persistent buffers | Something close to 60 fps *for the supported subset* — see §6 | a real Vulkan 1.0-subset game at vsync on this machine |
| **M6** | Windows: WGL context backend, ICD manifest placement, CI build | The same driver on the other half of desktop users | `vkcube` on Windows through WGL, or through ANGLE where WGL is not honest |

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
  known, and the `(set, binding) → GL index` table is stored on the layout, which
  the generated GLSL and the replay both read. M1 compacted the indices rather
  than computing `set * width + binding`, because
  `GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS` can be as low as 8; M2c refined that to
  numbering within each resource class, for the reason §3.5 gives.
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

What M2a did not cover, and M2b added: render passes, framebuffers and graphics
pipelines for untextured geometry. Uniform buffers and push constants are M2c,
and textures and samplers M2d; each was refused with an explicit error at the
earliest point the application reveals it needs them until its milestone landed. M2a's own remaining
costs, to be revisited: image-only allocations still allocate a full host block
that the image does not use, a sub-region readback of a mip level costs a
temporary of the whole level (`glGetTexImage` reads a level, not a rectangle), an
sRGB view of a UNORM image is refused rather than silently sampled with the
transfer function missing, and GL's filter and wrap state belongs to a texture
while Vulkan's belongs to a sampler — a real difference that M2d has to place
somewhere.

M2b is done: its test (`tests/draw.cpp`) draws a triangle into an image through a
render pass, a graphics pipeline and two vertex buffers, and compares the result
against a barycentric evaluation of the same vertex data performed in the test —
so the image is checked against a computation rather than against a recording of
what the driver once produced. It then draws the same triangle five more ways —
indexed, with binding offsets and a negative base vertex; with back faces culled
and the winding called counter-clockwise; with the same winding called clockwise,
which has to remove it; with a dynamic viewport; and into a render area covering
half the framebuffer — asks the rasteriser where each fragment is through a shader
that writes `gl_FragCoord` back out, and draws once through a pipeline whose
colour write mask suppresses two of the four channels.

Five things M2b settled.

- **No y flip is needed for vertices or for the winding order either, and this is
  now checked rather than argued.** The M2a note above establishes that image row
  *r* is the same row on both sides. The same coordinate arithmetic settles the
  rest: GL maps `y_d` to a window coordinate counted up from the attachment's
  bottom-left and Vulkan to a framebuffer coordinate counted down from its
  top-left, so *for the same NDC input the two produce the same number*. Nothing
  about vertex positions, the viewport's `y`, the scissor's `y`, or the winding
  needs touching. The temptation to flip "because the origins differ" is exactly
  the bug: a flip would turn an image that is correct under Vulkan's convention
  into an upside-down one. Cancelling the two origin conventions is what Zink has
  to do, because it goes the other way and must present GL's convention to
  Vulkan; Kniz passes it through, so `vertex.flip_vert_y` stays false and
  `VK_FRONT_FACE_COUNTER_CLOCKWISE` maps straight to `GL_CCW`.

  The test does not take this on faith. The triangle's signed area in framebuffer
  coordinates is positive, which both specifications call counter-clockwise, so
  culling back faces with a counter-clockwise front face must reproduce the
  unculled image exactly, and calling the same winding clockwise must remove the
  triangle entirely. A winding flipped anywhere in the chain fails one or the
  other, and a vertex-level flip moves the triangle's apex into the first row,
  where the per-texel checks cannot miss it.

- **The clip space fixup is applied, and it turned out to be observable after all.**
  SPIRV-Cross rewrites `gl_Position.z` to `2z - w` because Vulkan's depth range is
  `[0, w]` and GL's is `[-w, w]`. The expectation when M2b started was that nothing
  in this milestone could tell whether the rewrite had happened, because M2b
  refuses depth attachments — and that expectation was wrong: `gl_FragCoord.z` is
  the fragment's depth in GL's window space whether or not a depth buffer exists,
  so a vertex shader writing Vulkan depth 0.25 has to come back as 0.25, and would
  come back as 0.625 without the rewrite. The test's second shader pair now asks
  exactly that, which is the M1 rule applied in the other direction: a detail
  transcribed from the specification *and then* pinned down by a test, rather than
  a detail rediscovered later as a bug.

- **A pipeline is not an object to GL, so it is a program plus a list of state to
  re-apply.** GL's pipeline state — culling, front face, colour mask, viewport,
  scissor — lives in the context. Two pipelines used in the same pass therefore
  cannot be "bound" the way Vulkan means it; each draw sets the state it needs.
  That is the fidelity cost the table in §2 records as "state is per-context, not
  per-object", and it is paid on every draw rather than at every bind.

- **The render area had to be folded into the scissor, and the test found it.** The
  first version set the scissor from the render area at `vkCmdBeginRenderPass` and
  then set it again from the pipeline at each draw, which is correct for the load
  operation and wrong for everything else: a draw into a half-width render area
  covered the whole attachment. Vulkan's render area is not a description of what
  the clear touches, it is a clipping region for the pass. The two are now
  intersected when the draw is recorded, and a phase that renders into half the
  framebuffer and requires the other half to keep what was there before is what
  keeps it that way.

- **A load-op clear is not pipeline state, and GL does not separate the two.** A
  Vulkan clear happens before any pipeline is in effect; a GL clear sits in the
  same context state space as every draw, and whether `glClearBufferfv` consults
  the colour write mask is exactly the kind of thing hardware vendors disagree
  about. The mask is therefore set explicitly at the start of every pass, and
  dithering — which Vulkan does not have, and which perturbs a clear value in its
  least significant bit — is switched off with it. §3.10 has the details; the
  test's last two phases are the check that a pipeline's mask is applied to its
  own fragments and the record that the next pass's clear is not affected by it.

M2b's remaining costs, recorded so they do not have to be rediscovered: a VAO and
a program compile per pipeline, which is more creation-time work than the object
model strictly needs and will want a cache; there is no state deduplication
between consecutive draws, so a scene of two thousand objects makes two thousand
redundant state calls; and the render area intersection means the effective
scissor is stored per draw, which is cheap but means a dynamic scissor set once
and used by many draws is recomputed many times.

M2c is done: its test (`tests/uniform.cpp`) reads uniform buffers from a compute
shader and from both graphics stages, and checks push constants by running the same
pipeline twice with different values. It is built around two shapes that a driver
can get wrong quietly.

- **Two blocks at the same binding number in different sets.** Vulkan allows it;
  GLSL has no sets at all, so the two must end up at different GL binding points
  (§3.5). In the compute shader both blocks feed the result through arithmetic with
  no accidental symmetry, so a collapse shows up as a wrong number rather than a
  wrong-looking one; in the graphics pipeline the vertex stage takes set 0 and the
  fragment stage set 1, so a collapse makes the geometry move to wherever the
  colour says it should be, and the whole 64×64 image is compared against the
  coverage computed here from the transformed vertices. The check that this test
  would have caught the bug was made by breaking the translation on purpose: with
  the raw binding number used as the GL binding point, both the compute and the
  graphics phases fail.

- **A push constant that arrives late, at the wrong offset, or not at all.** The
  same pipeline and the same layout run twice with only the push constant changed,
  and both results have to be right — a value uploaded once per pipeline creation,
  or uploaded after the draw that reads it, cannot pass both. The block is
  deliberately a scalar followed by a vector, which is the case where Vulkan's
  layout rules and std140's could have disagreed and happen to agree; the one that
  really disagrees, an array of scalars, is refused at pipeline creation (§3.11)
  and the test asserts that refusal too.

- **A descriptor set whose binding was never written is refused, not drawn.** The
  test renders with a set that was allocated and never written and requires the
  image to still be the clear colour. Vulkan calls reading an unwritten descriptor
  undefined, and the useful answer is to skip the draw: the alternative is a draw
  computed from the *previous* phase's transform, which is a plausible-looking
  wrong image rather than an obvious failure.

M2c's remaining costs, recorded here rather than left to be found: descriptors are
re-bound on every draw and dispatch with no caching or dirty tracking (§3.5); the
push constant buffer is zeroed once per submission whether or not anything wrote to
it; dynamic uniform buffers are refused, so an application that wants one buffer
with many different ranges — the common case for per-object data — is turned away
rather than served; and push constants are uploaded where the recorded command
sits, which is correct for every layout an application may legally push with and
draw with, but which a command buffer that binds a no-push-constant layout's
descriptors between a push and a draw can still disturb, because the reserved
binding point is only guaranteed free within one layout rather than across a mixed
sequence. Making that airtight means snapshotting the values per draw and
re-uploading when the binding point has been claimed by something else; it is a
cost paid per draw, and the failure needs an application to deliberately interleave
two layouts, so it is recorded rather than paid.

M2d is done: its test (`tests/sampler.cpp`) samples an 8x8 image from a compute
shader and from a graphics pipeline, and compares every result against arithmetic
done in the test. It is built around the ways a texture path can be wrong while
still producing a plausible picture.

- **One image, two samplers.** The compute shader reads the same image through a
  clamp-to-edge sampler and a repeat sampler, and adds the two reads with
  different weights. If the filter or wrap state had been written to the texture
  rather than to a sampler object, the two reads would have to agree. The repeat
  read is addressed more than a whole wrap past the right edge, so clamp-to-edge
  would answer with the last column instead.

- **One binding number, two sets.** The second image lives at set 1 binding 0, the
  same number the first image uses in set 0. Texture units are one flat namespace
  in GL, so they have to be numbered the way uniform blocks are; a translation
  that used the binding number directly puts both at unit 0 and the second read
  answers with the first image.

- **A separate image and sampler.** The second compute shader declares
  `texture2D` and `sampler` and does both a filtered `texture()` and a
  `texelFetch()`. The filtered read is at a texel corner, exactly between four
  texels, so the LINEAR sampler the application bound gives the average of them
  while a missing sampler object would fall back to the texture's own NEAREST
  state and give one texel. The fetch is the dummy-sampler path, and it is exact
  either way, which is the point: the invented sampler must not change the answer.

- **Filtering in a graphics pipeline.** The fragment shader samples at
  `gl_FragCoord.xy / 64`, so the sample position is fixed by the rasteriser
  rather than chosen by the test. Each 8x8 block of pixels falls inside one texel
  with NEAREST; with LINEAR every position is a fixed binary fraction between two
  texels, so the expected value is exact and the comparison is tight.

The refusals are checked the same way M2c checked its own: a sampler with
comparison, unnormalized coordinates, anisotropy or mirror-clamp-to-edge is
refused at creation; an image created without `VK_IMAGE_USAGE_SAMPLED_BIT`, or one
with two mip levels, is refused at the descriptor write; and in both cases the
draw that would have used the binding is then refused too, so the render target
keeps its clear colour rather than showing whatever the last command left bound.

Two mutations were used to confirm the test fails when the driver is wrong, which
is the only evidence that a gate is a gate. Numbering texture units by their raw
binding number instead of densely per class makes the second image answer for the
first, and the compute comparison fails. Ignoring the sampler descriptor for a
separate image leaves the texture's own NEAREST state in place, and the filtering
comparison fails. Both were reverted.

M2d's remaining costs, recorded rather than left to be found: every texture-unit
descriptor is re-bound on every draw and dispatch, including the sampler object,
with no dirty tracking; the sampler object itself is cheap but the two GL calls
per unit are not free in a scene that binds hundreds; a descriptor write validates
the image view every time, which is once per write rather than per draw and so is
not on the hot path; and the mip-level refusal above means an application that
allocates a full mip chain for a texture it only ever samples at level 0 is turned
away, which is a large class of applications. That last one is the first thing the
transfer path should fix.

M3a is done: its test (`tests/present.cpp`) creates an X11 window, builds a
surface on it, creates a swapchain, draws into one of the images it was given, and
presents it. It is the first milestone whose result is not an image the test holds
but pixels on a screen, which changes what a gate has to do.

- **What the application controls is checked as before.** The swapchain images are
  ordinary textures, so drawing into one and copying it back proves the render
  pass, the pipeline and the attachment agree. The triangle is the M2b one,
  compared against a CPU rasterisation of the same vertex data.

- **The last step is checked by asking the driver to show its work.** The frame
  that is about to be swapped is written to a PPM (`KNIZ_PRESENT_DUMP`), and that
  file is compared against the same reference. A screenshot would prove the same
  thing less precisely and would need a person to look at it. Because the triangle
  is asymmetric about the horizontal axis, a flip anywhere in the present path
  moves the apex to the top row and fails the comparison.

- **The window is deliberately not mapped.** A mapped window is resized by the
  window manager — Hyprland tiled this one from 320x240 to 655x674 — and the frame
  is then a scaled version of the image, which cannot be compared pixel for pixel.
  An unmapped window keeps the geometry the test asked for, and the present path
  (draw, read back, swap) is identical; only the compositor's copy is missing. The
  mapped case was checked by hand, which is where the 655x674 observation came
  from.

- **Scaling is checked with a second swapchain** half the window's size, cleared
  to a single colour: every pixel of the presented frame must be that colour, so a
  present that drew at the image's size instead of the window's leaves three
  quarters of the window untouched.

- **The refusals are asserted**, in the style of M2b and M2d: a B8G8R8A8 format, a
  display-P3 colour space, one image, MAILBOX, a storage usage, a rotated
  pre-transform and pre-multiplied alpha are each required to fail, and presenting
  an index the swapchain does not have is required to fail rather than to draw.

Two mutations confirmed that the gate is a gate. Removing the flip from the
present shader moves the triangle's apex to the top of the window, and 56,964
pixels disagree with the reference. Setting the present viewport to the
swapchain's extent instead of the window's leaves 57,600 of 76,800 pixels
untouched — exactly the three quarters that a 160x120 image does not cover of a
320x240 window, which is the count the scaling check exists to make. Both were
reverted.

A third mutation was tried and is worth recording as a limit rather than a
success: removing the `glBindSampler` call from the present leaves the test
passing, because the test never leaves a mipmap-filtering sampler bound to unit 0
and the swapchain image's own state happens to be benign. The sampler object is
kept anyway, since it is what makes the present independent of whatever sampler
the application last bound to unit 0 — but this gate does not prove it, and saying
so is more useful than implying otherwise.

M3a's remaining costs are in §3.13.

M3b is done: its test (`tests/depth.cpp`) records nine passes against the same
colour-plus-depth framebuffer pair, where every phase's outcome depends on the
buffer the previous phase left behind, and each link is chosen so that one
specific plausible mistake flips it: a clear to a non-default 0.5 that must
reject a draw at 0.75 (a 1.0 clear draws instead); a nearer draw that must pass;
a `loadOp LOAD` pass that must still hold 0.25 (a quietly re-cleared or
reallocated attachment draws instead); a nearer draw against the loaded value;
a second clear that must put 0.5 back, checked in the depth readback because the
colour cannot distinguish "0.5" from "0.1" for a 0.9 draw; a no-test pipeline
that must still write; the value it wrote seen by the next pass; a
`VK_COMPARE_OP_GREATER` pipeline accepting the draw `LESS` rejected; and a
no-write pipeline testing without writing. Every phase checks the entire colour
attachment and the entire depth attachment — the depth against
`round(z · 65535)`, which is what a 16-bit unorm conversion produces — rather
than a sample of either, and both readbacks ride the same submission, so a phase
cannot pass its colour check at the cost of its depth one or the other way
round.

The refusals are the usual negative mirror: depth format with sampled or colour
usage at image creation, colour format with depth usage, a colour format in the
subpass's depth slot, a colour image in the framebuffer's depth slot, a
stencil-aspect clear on a packed format, the stencil test and the depth bounds
test in a pipeline, and a timeline semaphore. Five of them run before a single
pixel is drawn, which is the point — the application that asks wrong learns it
before guessing.

Three other things M3b settled:

- **`vkGetPhysicalDeviceFormatProperties` cannot be a stub.** vkcube was the
  first application to ask it a question, and an M0-era stub — zero features for
  every format — sent it down a path that ended in a null image view three
  calls later. The answer now comes from the same table `vkCreateImage` consults
  (§3.14), and the depth test checks the question vkcube actually asks
  (`vkGetPhysicalDeviceImageFormatProperties`) agrees with what image creation
  would have said.
- **A disabled depth test in GL stops the writes too.** The first M3b build
  honoured `depthTestEnable = FALSE, depthWriteEnable = TRUE` by disabling the
  test and leaving the mask on, and the write silently did not happen — on the
  Intel driver *and* on llvmpipe. GL and Vulkan disagree about what losing the
  test means, and the Vulkan meaning is the one an application invokes for
  depth-only pre-passes, so the mapping is now the test enabled with
  `GL_ALWAYS` (§3.14). The phase that found it is a phase that asks for exactly
  that shape and demands the readback change.
- **`vkcube` had a second crash that was all its own.** Even after the format
  report was fixed, `demo_destroy_texture` frees a buffer's memory before
  destroying the buffer that is still bound to it — a spec violation in the
  application, not the driver. Kniz warns and drops the binding rather than
  freeing memory GL may still hold handles into; vkcube runs to `SIGKILL` after
  four seconds of spinning cube instead of crashing, and the frame it presents is
  the lunar cube (verified through `KNIZ_PRESENT_DUMP` once more after M3b: the
  655×674 frame from M3a, with the depth-tested cube on it).

Binary semaphores are handled without ceremony because there is nothing to
mirror: `vkQueuePresentKHR` and `vkQueueSubmit` are synchronous in Kniz, so the
object exists and its state window is empty by construction (§3.14). The
free-before-destroy warning above is documented rather than punished; vkcube's
own teardown violates the spec and refusing it would make the canonical smoke
test refuse to close cleanly, which serves nobody.

M3b's remaining costs, recorded rather than hidden: stencil is the next real
piece of the depth path and is currently refused at three separate entry points
(render pass, pipeline, image format); `D24_UNORM_S8_UINT` and friends are in the
format table but nothing below 16-bit-unorm conversion semantics is written
through them yet; and a `loadOp LOAD` pass on a framebuffer that has never been
rendered into reads GL's undefined initial contents, which the test never
exercises because a framebuffer that has not been rendered to has no honest
contents to load.

## 5. Windows

A Windows port is why the platform layer is a small, named seam rather than
scattered EGL calls. What the port is concretely, and what it is not.

- **The driver code does not move.** Everything from the object model down
  through translation, memory, buffers, images, render passes and depth was
  written against an `egl::` surface designed to be swapped. The port is: a WGL
  backend that creates a context and loads its functions via `wglGetProcAddress`
  (the same lazy-load pattern the EGL backend already uses internally, so the
  function table itself moves untouched), the swapchain's present step replaced
  by `SwapBuffers` on a window whose pixel format supports depth, and the ICD
  manifest reachable through the loader's search paths.
- **Where the manifest lands.** On Windows the loader finds ICDs by scanning
  registry entries under `HKLM\SOFTWARE\Khronos\Vulkan\Drivers\` whose DWORD 0
  values name each manifest JSON — or, simplest for Kniz, the same
  `VK_DRIVER_FILES` / `VK_ICD_FILENAMES` environment variables the loader
  honours on every platform. The Windows-specific delta is how the DLL exports
  `vk_icdGetInstanceProcAddr` (`__declspec(dllexport)` under Vulkan's name) and
  the manifest's `library_path` resolution, which stays manifest-relative
  exactly as it is.
- **The GL backend on Windows is the risk, not Vulkan.** Desktop GL on NVIDIA and
  AMD is a real 4.x core driver; Intel's Windows GL is thin; some machines have
  only Microsoft's stale GL 1.x shim, which cannot be Kniz's floor (it needs 4.5+
  core for buffer objects, FBOs and shader objects). The fallback that *is*
  portable is ANGLE — EGL over D3D11 — which is how Chrome ships OpenGL on
  Windows. Kniz-on-ANGLE = Vulkan → GL → D3D11, one translation deeper than
  DXVK's native Vulkan → D3D11; the honest statement is the same as on Linux,
  doubled: Kniz is the way to run Vulkan **at all** on a machine whose only
  working path is GL, never the way to race software that skips a link.
- **DX owns the other niche.** This is worth saying once, plainly, because it is
  the reason the Windows port will not be anyone's performance answer: on
  Windows, Vulkan-on-GPU that skips native Vulkan is exactly what DXVK already
  is, mature, and it does one translation instead of two. Kniz's Windows value is
  the same as its Linux value — Vulkan *capability* on GL-only machines — with
  the same ceiling.

## 6. Performance: what the "smooth tier" takes

"Smooth" is tier-shaped, not absolute. The honest decomposition of the goal that
drove M5 into the roadmap above:

- **What it applies to.** Vulkan 1.0 feature subsets — 2D games, mobile-range 3D,
  emulator renderers with simple backends, Khronos samples beyond `vkcube`.
  These use everything Kniz already owns: one render pass, one depth attachment,
  uniform buffers, push constants, one or two textures, indexed draws.
- **What dominates the per-draw cost, measured from the current code.** Every
  draw re-applies the whole pipeline state and rebinds every descriptor —
  recorded in M2b and M2d's remaining costs — and every buffer upload rides the
  host mirror. None of it is free; together it is the reason a supported game
  would stutter today even when nothing is refused. The rest of the frame has no
  translation overhead, so this is where the budget is spent and this is where
  it comes back.
- **The plan, in the order it earns its keep.**
  1. *Pipeline state diffing.* GL context state is already a mutable copy; put
     the copy in the driver and apply only what changed since the last draw of
     the previous pipeline. Most scenes bind the same pipeline for runs of
     draws, and the jump from "all state" to "the delta" is the single largest
     saving.
  2. *Descriptor dirty-tracking.* Same shape as `sync_buffers_from_gl`: keep the
     applied-on-GPU version of each descriptor set in the set object, rebind
     only when a write or a different set touched it.
  3. *Persistent-mapped GL buffers.* The host mirror exists because GL upload
     required it; with `GL_MAP_PERSISTENT_BIT` the mirror and the GL object can
     be one allocation, saving a full copy per submit for every buffer the CPU
     wrote once and the GPU never changes.
  4. *Draw batching on the IR.* Recorded draws that differ only in vertex counts
     and offsets, sharing one buffer and one bound state, collapse at replay into
     one set of bind calls with consecutive `glDrawArrays` — the command IR is an
     intermediate representation, so this is a transform on it rather than a
     rewrite of the replay.
  5. *Program binary caching.* SPIRV-Cross output depends only on the SPIR-V and
     the layout; persist the translated GLSL (and Mesa's program binary) keyed by
     their hashes, so the second run of a game skips translation and compile
     entirely.
- **The gate that says "smooth" has been reached.** Not a synthetic benchmark:
  one real Vulkan 1.0-subset game — choosable at the time — running at the
  display's refresh rate on the development machine (a 2013 iGPU), with the frame
  time distribution recorded rather than the peak claimed. Below that, the
  release notes do not claim smoothness; that is the same conservatism §2 opened
  with, applied one level down.
- **Which games that means, so the gate is choosable now.** Native Vulkan
  renderers inside the supported subset: `vkQuake` and `vkQuake2` (open-source
  id Tech 1/2 on Vulkan; the likely first gate), Croteam's `The Talos Principle`
  (the first native Vulkan PC renderer off the shelf), GZDoom's Vulkan backend
  (the widest content library; its dynamic-descriptor paths will probably meet
  refusals, which is a finding either way). What it expressly does *not* mean,
  and why: GTA V and Saints Row IV are the same confused example twice —
  ordinary D3D11 games that reach Vulkan only as DXVK's translation output
  inside Proton; running them over Kniz stacks `D3D11 → DXVK → Vulkan → Kniz →
  GL` on a heavy world, and RDR2's switchable Vulkan renderer is a native path
  that far exceeds anything this iGPU can hold at 60. The honest target is the
  games that render Vulkan themselves, not the famous ones whose DXVK fame
  makes them look Vulkan — the reason §1 opened with the DXVK caution.

## 7. Non-goals

CTS conformance. Multi-queue correctness. Sparse resources. Timeline
semaphores. Ray tracing and mesh shaders. Any claim of performance parity —
see §1 — or of smoothness outside the tier §6 defines.
