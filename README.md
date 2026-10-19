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

**M2c complete.** Kniz registers as a Vulkan ICD, the loader hands it an instance
and a device, it brings up a real OpenGL context to back them, and it can run a
compute shader over storage buffers and uniform buffers, move image contents to
and from buffers, and draw untextured geometry into an image through a render pass
and a graphics pipeline, with uniform buffers and push constants reaching both
kinds of shader. It does not present anything to a window yet.

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

Not implemented: samplers, depth and stencil attachments, blending,
multisampling, instancing, geometry and tessellation stages, dynamic uniform
buffers, and presentation. Each is refused with a specific error at the point the
application reveals it needs one, rather than accepted and silently ignored. Two
refusals are permanent rather than pending, and both are argued in
[`docs/DESIGN.md`](docs/DESIGN.md):

- `VK_IMAGE_TILING_LINEAR` — a linear image's layout is observable and OpenGL's
  storage layout is not it.
- Mapping a `VkDeviceMemory` that an image is bound to — an image is a GL
  texture, so the allocation does not hold its pixels.

Next is M2d for samplers and textures, then M3 for a swapchain and a window.

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
Instance
  vkCreateInstance       ok
  physical devices       1

Device
  name                   Kniz on Mesa Intel(R) HD Graphics 4400 (HSW GT2)
  vendorID               0x4b4e  (Kniz)
  apiVersion             1.0.0
  device extensions      0

What is implemented
  loader handshake, instance/device    yes  M0
  properties, limits, memory model     yes  M0
  buffers and memory objects           yes  M1
  SPIR-V -> GLSL translation           yes  M1
  descriptor sets (storage buffers)    yes  M1
  compute dispatch                     yes  M1
  images, image views, memory binding      yes  M2a
  images: clear, upload, readback          yes  M2a
  render passes, framebuffers, draws       yes  M2b
  graphics pipelines and rasterisation     yes  M2b
  uniform buffers                          yes  M2c
  push constants                           yes  M2c (lowered onto a uniform buffer)
  linear tiling (layout is observable)     no   never planned
  images: storage, input attachments       no   not implemented
  mapping an image's allocation            no   impossible, see DESIGN.md
  depth, stencil, blending, multisampling  no   M2b refused on purpose
  dynamic uniform buffers                  no   M2c refused on purpose
  push constants: non-std140 packing       no   OpenGL cannot express it, see DESIGN.md
  samplers and textures in shaders         no   M2d
  swapchain / window presentation          no   M3
  fences, queries, timestamps              no   M4
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

Five tests, all driven through the **real Vulkan loader** rather than by calling
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
  back as 0.25 rather than as 0.625. The refusals — a second subpass, a depth
  attachment, an input attachment, blending, a geometry stage, an integer vertex
  format, a pipeline used with another render pass — are asserted too.
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

The shaders the tests use are checked in twice: as GLSL, and as the compiled
SPIR-V an application would actually submit. The command that regenerates each
SPIR-V file is recorded at the top of the GLSL it came from
(`tests/shaders/scale.comp`, the four that `draw` uses, and the four that
`uniform` uses), and the compiled forms are committed so that the test suite does
not depend on `glslangValidator` being installed.

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
