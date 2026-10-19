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

**M2a complete.** Kniz registers as a Vulkan ICD, the loader hands it an instance
and a device, it brings up a real OpenGL context to back them, and it can run a
compute shader over storage buffers and move image contents to and from buffers.
It does not render anything yet.

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

Not implemented: uniform buffers, push constants, samplers, render passes,
graphics pipelines, presentation. Each is refused with a specific error at the
point the application reveals it needs one, rather than accepted and silently
ignored. Two refusals are permanent rather than pending, and both are argued in
[`docs/DESIGN.md`](docs/DESIGN.md):

- `VK_IMAGE_TILING_LINEAR` — a linear image's layout is observable and OpenGL's
  storage layout is not it.
- Mapping a `VkDeviceMemory` that an image is bound to — an image is a GL
  texture, so the allocation does not hold its pixels.

Next is M2b, offscreen graphics: render pass → framebuffer, pipeline → program, a
triangle into an image compared against a reference.

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
  images, image views, memory binding  yes  M2a
  images: clear, upload, readback      yes  M2a
  linear tiling (layout is observable) no   never planned
  images: storage, input attachments   no   not implemented
  mapping an image's allocation        no   impossible, see DESIGN.md
  uniform buffers, push constants      no   M2b
  render passes, pipelines, draws      no   M2b
  samplers and textures in shaders     no   M2b
  swapchain / window presentation      no   M3
  fences, queries, timestamps          no   M4
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

Three tests, all driven through the **real Vulkan loader** rather than by calling
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
  `vkCmdFillBuffer`, `vkCmdUpdateBuffer` and `vkCmdCopyBuffer`, and asserts that a
  graphics pipeline is *refused* rather than accepted and ignored.
- `image` — uploads a 64×32 image from a buffer and reads it back, requiring the
  bytes to be identical (M2a). The pattern varies along both axes, so a flip or a
  transposed stride cannot survive it. It also clears an image and requires the
  exact texels the float clear value denotes, copies a sub-rectangle out of a
  buffer whose rows are deliberately padded, and asserts the refusals hold —
  including that mapping an allocation an image is bound to fails rather than
  handing back a pointer to bytes that are not the image.

The shader the second test uses is checked in twice: as GLSL, and as the compiled
SPIR-V an application would actually submit. The command that regenerates the
SPIR-V is recorded at the top of `tests/shaders/scale.comp`, and the compiled form
is committed so that the test suite does not depend on `glslangValidator` being
installed.

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
