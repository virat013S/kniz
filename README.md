# Kniz

A Vulkan Installable Client Driver (ICD) implemented on top of OpenGL.

> **The name** is "Zink" mirrored. [Zink](https://docs.mesa3d.org/drivers/zink.html)
> translates OpenGL → Vulkan. Kniz attempts the opposite: it presents a Vulkan
> device to an application and implements that device by driving an OpenGL
> context.

## Read this before you expect anything

**Kniz is a learning project. It is not, and cannot become, a way to get native
performance for Vulkan applications, and it will never let you run games better
than DXVK, Wine or Proton.**

The reason is structural, not a matter of effort. Zink reaches near-native speed
because it translates *upward* into a more expressive API: every OpenGL
operation has a Vulkan way to express it. Kniz translates *downward*, into an API
that cannot represent what Vulkan guarantees — descriptor indexing, subpasses,
timeline semaphores, multiple queues, sparse resources, render pass load/store
ops, and a great deal of what SPIR-V can express. Those are **discarded, not
emulated**. [`docs/DESIGN.md`](docs/DESIGN.md) tabulates exactly what survives.

Two consequences worth stating plainly:

- **Kniz cannot beat DXVK.** It sits *underneath* it in the stack. In the chain
  `game → DXVK → Vulkan → Kniz → OpenGL → GPU`, Kniz re-translates DXVK's output
  downward, throwing away the features DXVK depends on. It can only add cost.
- **Wine and Proton are a different layer.** They implement Win32, PE loading,
  the registry, threads and files; a Vulkan ICD implements a few hundred graphics
  entry points. There is no overlap to compete on.

The honest, reachable goal:

> Run `vkcube`, the Khronos Vulkan samples, and simple Vulkan 1.x applications
> correctly on a desktop OpenGL 4.x core context.

## Status

**M3b complete.** Kniz registers as a Vulkan ICD, the loader hands it an instance
and a device, it brings up a real OpenGL context to back them, and it can run a
compute shader over storage buffers and uniform buffers, move image contents to
and from buffers, draw textured geometry into an image through a render pass and a
graphics pipeline, with uniform buffers, push constants, samplers and textures
reaching both kinds of shader, test and write depth against a D16 attachment —
and create a swapchain on an X11 window and present to it. `vkcube` runs.

Working today:

- Loader handshake, ICD interface version negotiation
- `VkInstance` / `VkPhysicalDevice` / `VkDevice` / `VkQueue` creation and teardown
- Physical device properties, limits, features, queue families, memory heaps
- A headless OpenGL context, with and without a window system present
- Buffers, `VkDeviceMemory`, mapping, `vkCmdFillBuffer` / `vkCmdUpdateBuffer` /
  `vkCmdCopyBuffer`
- SPIR-V → GLSL, compute pipelines, descriptor sets, dispatch, submission and
  fences — verified numerically against a CPU reference, not merely exercised
- Images and image views, image memory requirements and binding,
  `vkCmdClearColorImage`, `vkCmdCopyBufferToImage`, `vkCmdCopyImageToBuffer` —
  verified by byte-exact round trips, not merely exercised
- Render passes, framebuffers, graphics pipelines, vertex input from one or two
  buffers at once, indexed draws, dynamic viewport and scissor, and back-face
  culling — verified against a CPU rasterisation of the same triangle, including
  which way round the image came out and which winding the triangle has
- Uniform buffers, in compute and in graphics, including two blocks that share a
  binding number in different descriptor sets — a shape GLSL cannot express, so it
  is rewritten onto separate GL binding points and checked by moving the geometry
  with one block and colouring it with the other
- Push constants, which OpenGL has no equivalent of, lowered onto a uniform buffer
  the driver owns and uploaded as the recorded commands replay — verified by
  running the same pipeline twice with different values and requiring both results
- Samplers and textures, in compute and in graphics: combined image samplers,
  separate image and sampler pairs, and `texelFetch` on a bare image — verified
  against a CPU evaluation of the same filtering, with the same image read through
  two different samplers and two images at the same binding number in different
  sets, which are the two ways a texture path can be quietly wrong
- Surfaces, swapchains and presentation on an X11 window: the images a swapchain
  hands out are ordinary GL textures, so a render pass draws into one with no new
  code, and `vkQueuePresentKHR` is a draw from that texture into the window
  followed by `eglSwapBuffers` — verified by comparing the frame the driver was
  about to swap against the image it was told to draw, pixel for pixel and
  orientation included, rather than by looking at a window
- Depth attachments (D16_UNORM): render-pass clear and load operations, the
  pipeline's depth test, write mask and compare operation, and `gl_Position.z`
  for every depth in `[0, 1]` — verified by a chain of nine passes where each
  pass's outcome depends on the buffer the previous one left behind, with both
  the colour attachment and the depth attachment read back every time. Binary
  semaphores are objects but carry no state, because submission and present are
  synchronous; timeline ones are refused at creation.
- `vkcube`, the canonical smoke test, runs and spins until closed — it had been
  the first application to demand depth and to find two real driver bugs along
  the way, both now fixed and pinned by the depth test below.

Not implemented: the stencil test, the depth bounds test, depth clamping and
depth bias, blending, multisampling, instancing, geometry and tessellation
stages, dynamic uniform buffers, comparison samplers, anisotropic filtering, mip
levels above 0, timeline semaphores, Wayland surfaces, and every swapchain format
but `VK_FORMAT_R8G8B8A8_UNORM`. Each is refused with a specific
error at the point the application reveals it needs one, rather than accepted and
silently ignored. Three refusals are permanent rather than pending, and all three
are argued in [`docs/DESIGN.md`](docs/DESIGN.md):

- `VK_IMAGE_TILING_LINEAR` — a linear image's layout is observable and OpenGL's
  storage layout is not it.
- Mapping a `VkDeviceMemory` that an image is bound to — an image is a GL
  texture, so the allocation does not hold its pixels.
- `VK_PRESENT_MODE_MAILBOX_KHR` and `IMMEDIATE` — presentation is synchronous, so
  there is no queue to drop a frame from or to skip.

Next is Wayland surfaces, which is what a Wayland-only session needs before the
swapchain path works at all there.

## Building

Requires a C++20 compiler, CMake ≥ 3.20 and EGL headers. Nothing else — no
distribution-specific packages are assumed, and no Python or code generation
step exists.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```

Dependencies are resolved in two tiers: your distribution's copy first, a pinned
download second. Vulkan-Headers is header-only and is fetched if not found; EGL
comes from pkg-config.

SPIRV-Cross does the SPIR-V → GLSL translation and is the one dependency where the
distribution's copy is usually *not* the useful one: Arch and Fedora build it with
`SPIRV_CROSS_SHARED`, which exports only the C API and installs no CMake package,
so the C++ GLSL backend Kniz uses is absent. The pinned download is therefore what
normally runs. It is the slowest part of a first build — about a minute, once. If
your distribution does install the static C++ targets, they are used instead.

| Distribution | Packages needed |
|---|---|
| Debian / Ubuntu | `build-essential cmake ninja-build pkg-config libegl1-mesa-dev` |
| Fedora | `gcc-c++ cmake ninja-build pkgconf-pkg-config mesa-libEGL-devel` |
| Arch | `base-devel cmake ninja pkgconf mesa` |

Distributions that package Vulkan headers (`vulkan-headers` on Fedora and Arch)
use them; on Ubuntu the headers are downloaded instead. Force either behaviour
with `-DKNIZ_FETCH_DEPS=OFF` or `ON`.

## Running

The easiest way is the launcher at the top of the tree. It builds if it needs to,
finds the manifest, and prints a status report when given no command:

```sh
./kniz                       # build if needed, then report status
./kniz vulkaninfo --summary  # run any Vulkan program on Kniz
./kniz --log 3 vkcube        # the same, with trace logging
```

Run with no arguments, `./kniz` executes `kniz-info`, which reports what the ICD
gives an application and how much of Vulkan is actually implemented:

```
Driver
  manifest               /home/mitsuki/Projects/kniz/build/src/kniz_icd.json
  selected by            VK_DRIVER_FILES
  loader                 libvulkan.so.1
  loader API             1.4.363
kniz: [info] OpenGL 4.6 core context: Mesa Intel(R) HD Graphics 4400 (HSW GT2) / 4.6 (Core Profile) Mesa 26.2.4-arch3.1
kniz: [info] compute shaders available; shader storage buffer offset alignment 4
kniz: [info] uniform buffers available; uniform buffer offset alignment 32
kniz: [info] images available: textures, with glTexStorage2D, glClearTexImage and glGetTexImage
kniz: [info] samplers available: sampler objects, 160 texture unit(s)
kniz: [info] presentation available: X11 windows, through EGL on the X11 platform
kniz: [info] rendering available: framebuffer objects, vertex array objects, draws
kniz: [info] instance created: requested Vulkan 1.0, exposing 1.0

Instance
  vkCreateInstance       ok
  physical devices       1

Device
  name                   Kniz on Mesa Intel(R) HD Graphics 4400 (HSW GT2)
  type                   other (Kniz does not claim to be real hardware)
  vendorID               0x4b4e  (Kniz)
  deviceID               0x2617
  apiVersion             1.0.0
  driverVersion          0.0.1
  device extensions      1

Queues and memory
  families               0: 1 queue(s), graphics compute transfer
  heaps                  0: 1024 MiB, device local
  types                  0: heap 0, device-local
                         1: heap 0, device-local host-visible host-coherent

Limits (Vulkan's required floor)
  maxImage2D             4096
  maxUBO range           16384
  maxSSBO range          134217728
  maxDescSets            4
  maxWorkGroup           128 x 128 x 64
  maxViewports           1
kniz: [info] device created (queue family 0)

Device creation
  vkCreateDevice         ok
  vkGetDeviceQueue       ok
  vkDeviceWaitIdle       ok

What is implemented
  loader handshake, instance/device        yes  M0
  properties, limits, memory model         yes  M0
  buffers and memory objects               yes  M1
  SPIR-V -> GLSL translation               yes  M1
  descriptor sets (storage buffers)        yes  M1
  compute dispatch                         yes  M1
  images, image views, memory binding      yes  M2a
  images: clear, upload, readback          yes  M2a
  render passes, framebuffers, draws       yes  M2b
  graphics pipelines and rasterisation     yes  M2b
  uniform buffers                          yes  M2c
  push constants                           yes  M2c (lowered onto a uniform buffer)
  samplers                                 yes  M2d (GL sampler objects)
  textures in shaders                      yes  M2d (combined or separate)
  swapchain / window presentation          yes  M3 (X11 windows, FIFO)
  acquire / present / image cycling        yes  M3
  depth attachment, depth test/write/compare yes  M3b (stencil and depth bounds refused)
  binary semaphores                        yes  M3b (no state: submission and present are synchronous)
  linear tiling (layout is observable)     no   never planned
  images: storage, input attachments       no   not implemented
  mapping an image's allocation            no   impossible, see DESIGN.md
  stencil, depth bounds, blending, multisampling no   refused on purpose, see DESIGN.md
  dynamic uniform buffers                  no   M2c refused on purpose
  push constants: non-std140 packing       no   OpenGL cannot express it, see DESIGN.md
  comparison samplers, anisotropy          no   M2d refused on purpose
  sampling mip levels above 0              no   only level 0 is ever written, see DESIGN.md
  Wayland surfaces                         no   the EGL context is on X11, see DESIGN.md
  sRGB and BGRA swapchain formats          no   M3 refused on purpose
  MAILBOX and IMMEDIATE present modes      no   presentation is synchronous, see DESIGN.md
  fences, queries, timestamps              no   M4

  Images exist and their contents move correctly -- a buffer to image and
  back again returns the bytes it was given, a clear lands as the exact
  texels its float value denotes, and a sub-rectangle copy respects both its
  offset and a padded buffer row length.  Lines and triangles are drawn into
  an image through a render pass, a graphics pipeline and one or two vertex
  buffers, and the result is checked against a CPU rasterisation of the same
  geometry rather than merely eyeballed.  Uniform buffers are read by both
  compute and graphics shaders, including two blocks that share a binding
  number in different sets, and push constants arrive in the shader that
  declared them.  Textures are sampled in both compute and graphics shaders,
  through combined image samplers and through a separate image and sampler,
  with the filter and address modes coming from the sampler object rather
  than from the image.  A swapchain is created on an X11 window, drawn into
  through an ordinary render pass, and presented: the frame that is about to
  be shown is read back and compared against the image, so the present path
  is checked rather than looked at.  A D16 depth attachment can be cleared,
  tested with the pipeline's compare operation, written and loaded across a
  pass; vkcube draws and spins through the whole stack.

  Not implemented yet, and refused with a specific error rather than
  approximated: the stencil test, the depth bounds test, depth clamping and
  depth bias, blending, multisampling, instancing, geometry and tessellation
  stages, more than one subpass, input attachments, dynamic uniform buffers,
  comparison samplers, anisotropic filtering, mip levels above 0, timeline
  semaphores, Wayland surfaces, and every swapchain format but
  R8G8B8A8_UNORM.
  Programs that create a device, ask questions about it, run a compute shader
  over buffers, move image contents, draw depth-tested geometry textured from
  uniform and sampler data, or draw into a window will work.
  See docs/DESIGN.md for what OpenGL cannot express at all.

  The 'kniz:' lines above are the driver's own log, including the
  OpenGL context it created to back this device.

OK
```

The launcher sets `VK_ICD_FILENAMES`, `VK_DRIVER_FILES` and `KNIZ_LOG`, so you can
always do the same by hand. Both manifest variables are set on purpose: the loader
prefers `VK_DRIVER_FILES` and then *ignores* `VK_ICD_FILENAMES`, so a machine that
pins a vendor driver with the newer variable would otherwise send every program to
the system's own Vulkan driver. The launcher also prints which variable is in
force. The loader resolves the manifest's `library_path` relative to the
manifest's own directory, which is why the manifest is copied next to the built
library:

```sh
VK_ICD_FILENAMES=$PWD/build/src/kniz_icd.json \
VK_DRIVER_FILES=$PWD/build/src/kniz_icd.json vulkaninfo --summary
```

Expected output:

```
GPU0:
	apiVersion    = 1.0.0
	vendorID      = 0x4b4e
	deviceType    = PHYSICAL_DEVICE_TYPE_OTHER
	deviceName    = Kniz on Mesa Intel(R) HD Graphics 4400 (HSW GT2)
```

Because Kniz registers as an *additional* device rather than replacing your real
driver, the system driver stays available, which makes A/B comparison easy.

Set `KNIZ_LOG=0..4` (`error`..`trace`) to control logging; the default is `2`.

## Testing

```sh
ctest --test-dir build --output-on-failure
```

Eight tests, all driven through the **real Vulkan loader** rather than by calling
the ICD directly. That is deliberate: the most damaging class of ICD bug — a
dispatchable object that does not reserve its first word for the loader's
dispatch pointer — is invisible when the ICD is called directly, and produces a
device handle that `vkCreateDevice` returns happily and every later call rejects.
See the note in [`docs/DESIGN.md`](docs/DESIGN.md).

- `lifecycle` — instance, device and queue creation and teardown (M0).
- `compute` — records a dispatch over two storage buffers and compares all 1024
  elements against a C++ reference (M1). The comparison is exact rather than a
  tolerance, because the failure this guards against — a descriptor bound to the
  wrong buffer — produces plausible numbers rather than an error. It also covers
  `vkCmdFillBuffer`, `vkCmdUpdateBuffer` and `vkCmdCopyBuffer`, and asserts that an
  empty graphics pipeline is *refused* rather than accepted and ignored.
- `image` — uploads a 64×32 image from a buffer and reads it back, requiring the
  bytes to be identical (M2a). The pattern varies along both axes, so a flip or a
  transposed stride cannot survive it. It also clears an image and requires the
  exact texels the float clear value denotes, copies a sub-rectangle out of a
  buffer whose rows are deliberately padded, and asserts the refusals hold —
  including that mapping an allocation an image is bound to fails rather than
  handing back a pointer to bytes that are not the image.
- `draw` — draws a triangle into an image and reads it back (M2b). The three
  vertices carry red, green and blue, the triangle is asymmetric about the
  horizontal axis, and the colours it should produce are computed on the CPU from
  barycentric coordinates of the same vertex data — so an upside-down image, a
  transposed one, a swapped colour channel or a wrong interpolation all show up as
  a numeric disagreement rather than as an image that still contains a triangle.
  The same triangle is then produced five more ways: indexed, with the vertex
  buffers bound at non-zero offsets and a negative base vertex; with a dynamic
  viewport; with a render area covering half the framebuffer, which must leave the
  other half untouched; once with back faces culled, which must keep the triangle,
  and once with the winding called clockwise, which must remove it; and once
  through a pipeline writing only two of its four colour channels. A separate
  shader writes `gl_FragCoord` back out, so the rasteriser is asked where it thinks
  each fragment is instead of being told — which is also what pins down the clip
  space depth conversion, since a vertex written at Vulkan depth 0.25 has to read
  back as 0.25 rather than as 0.625. The refusals — a second subpass, a colour
  format in the depth slot, an input attachment, blending, a geometry stage, an
  integer vertex format, a pipeline used with another render pass — are asserted
  too.
- `uniform` — uniform buffers and push constants (M2c). Two uniform blocks are
  placed at the *same* binding number in *different* descriptor sets, which Vulkan
  allows and GLSL cannot express: GL numbers uniform blocks in one flat namespace,
  so a translation that used the binding number as the GL binding point would leave
  one buffer answering for both. The compute shader is checked numerically against a
  CPU computation that no wrong binding reproduces; the graphics shader puts the
  vertex stage's block in set 0 and the fragment stage's in set 1, so the same
  mistake moves the geometry to wherever the colour says it should be, and the whole
  image is compared against the coverage computed on the CPU from the transformed
  vertices. Push constants are checked by running the same pipeline and layout
  twice with different values and requiring two different, both correct, results —
  and by a draw whose descriptor set was never written, which must be refused
  instead of reading whatever the previous draw left bound. A push constant block
  that std140 cannot express — Vulkan packs an array of scalars more tightly than a
  GLSL uniform block can be packed — is asserted to be refused at pipeline
  creation rather than lowered with its members at the wrong offsets.
- `sampler` — samplers and textures (M2d), in a compute and a graphics pipeline.
  One 8×8 image is read through a clamp-to-edge sampler and a repeat sampler at
  once, which is the check that filter and wrap state lives on the sampler rather
  than on the texture; a second image sits at the *same* binding number in the
  *next* descriptor set, which is the check that texture units are numbered
  densely rather than by binding number; and a separate image and sampler pair is
  sampled and `texelFetch`ed, which are the two paths SPIR-V has and GLSL does
  not. Every result is compared against arithmetic performed in the test: the
  filtered reads sit at texel corners, exactly between four texels, so a missing
  sampler object falls back to the texture's own NEAREST state and disagrees. The
  graphics phase samples at `gl_FragCoord.xy / 64`, so the sample position is
  fixed by the rasteriser instead of chosen by the test, and each of the 4096
  pixels is compared. Comparison samplers, unnormalized coordinates, anisotropy and
  mirror-clamp-to-edge are asserted to be refused at sampler creation; an image
  without `VK_IMAGE_USAGE_SAMPLED_BIT`, and one with more than one mip level, are
  refused at the descriptor write, which is then proved by the draw that would
  have used the binding being refused too.

- `present` — swapchain and presentation (M3a), on an X11 window the test makes
  for itself through `libX11` opened with `dlopen`, so the build gains no X11
  dependency and a machine with no display skips the test instead of failing it.
  The surface reports what it can do and the refusals are asserted — a BGRA
  format, a display-P3 colour space, one image, MAILBOX, storage usage, a rotated
  pre-transform, pre-multiplied alpha. A swapchain is created, a triangle is drawn
  into one of its images and copied back, and the result is compared against a CPU
  rasterisation of the same vertices. Then the image is presented, and the frame
  the driver was about to swap is compared against that same reference: the
  triangle's flat edge and its apex are at opposite ends of the image, so a flip
  anywhere in the present path moves the apex to the top and fails. A second
  swapchain half the window's size is cleared and presented, which checks that
  presentation scales to the window rather than drawing at the image's size.
  Acquisition is checked to walk the images in order and wrap, and presenting an
  index the swapchain does not have is checked to be refused. The window is
  deliberately left unmapped: a mapped window is resized by the window manager,
  and a frame scaled to a size the test did not choose cannot be compared pixel
  for pixel.

- `depth` — depth attachments, the depth test and depth writes (M3b). Nine passes
  are recorded against the same 64×64 colour+depth framebuffer pair, and each
  phase's outcome depends on what the previous phase left in the buffer, so a
  quiet reallocation, a re-clear or a lost load operation flips colour early: a
  clear to a non-default 0.5 must reject a draw at 0.75; a draw at 0.25 must then
  pass and write; a pass with `loadOp LOAD` must still hold 0.25 and reject 0.75
  again; a nearer draw at 0.1 must pass against that; clearing again must put 0.5
  back (checked in the depth readback, since 0.9 is rejected either way); a
  pipeline with the test disabled and writes enabled must still write, which GL
  does not do for a genuinely disabled depth test and which the driver maps to
  `GL_ALWAYS` instead; the written 0.9 must be seen by the next pass; a
  `VK_COMPARE_OP_GREATER` pipeline must accept the draw `LESS` rejected; and a
  pipeline with writes off must test without writing. Every phase checks the whole
  colour attachment and the whole depth attachment — against `round(z · 65535)`
  with two counts of slack, in the way a D16 conversion rounds — rather than a
  sampling of either. `vkGetPhysicalDeviceFormatProperties` is checked to agree
  with what the driver accepts, which is not decoration: vkcube asked it whether
  an sRGB format could be sampled, was told no by an earlier stub, and skipped
  creating its texture. The refusals follow: depth format with sampled or colour
  usage, colour format with depth usage in an image, a render pass, or a
  framebuffer's depth slot; a stencil-aspect clear of a packed format; the
  stencil test and the depth bounds test in a pipeline; and a timeline semaphore
  at creation.

The shaders the tests use are checked in twice: as GLSL, and as the compiled
SPIR-V an application would actually submit. The command that regenerates each
SPIR-V file is recorded at the top of the GLSL it came from
(`tests/shaders/scale.comp`, the four that `draw` uses, the four that `uniform`
uses, the four that `sampler` uses, the two that `present` reuses from `draw` and
the two that `depth` uses), and the compiled forms are committed so that the test
suite does not depend on `glslangValidator` being installed.

## Layout

```
src/icd/     the ICD: bootstrap, object model, GL backend, shader translation
icd/         ICD manifest installed for loader discovery
tests/       tests driven through the loader, and the shaders they use
tools/       kniz-info, the status report
kniz         the launcher (top of tree)
docs/        design notes and the fidelity budget
```

## Prior art

- **[Ashes](https://github.com/DragonJoker/Ashes)** (MIT) — the same idea, mature.
  A Vulkan ICD and drop-in `libvulkan.so` replacement with OpenGL 3.3–4.6, D3D11
  and Vulkan backends. 817 commits since 2018 and still not CTS-conformant, which
  is a useful calibration for what this project can expect.
- **Zink** (Mesa) — OpenGL on Vulkan. The opposite direction, and the source of
  the name.
- **MoltenVK** — Vulkan on Metal. The best example of Vulkan as a source API, and
  a good illustration of where fidelity stops.

## License

MIT. See [`LICENSE`](LICENSE).
