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

**M0 complete.** Kniz registers as a Vulkan ICD, the loader hands it an instance
and a device, and it brings up a real OpenGL context to back them. It renders
nothing yet.

Working today:

- Loader handshake, ICD interface version negotiation
- `VkInstance` / `VkPhysicalDevice` / `VkDevice` / `VkQueue` creation and teardown
- Physical device properties, limits, features, queue families, memory heaps
- A headless OpenGL context, with and without a window system present

Not implemented: everything that draws or computes. Next is M1, an offscreen
compute dispatch verified against a CPU reference, which exercises SPIR-V → GLSL,
memory and descriptor sets in one step.

## Building

Requires a C++20 compiler, CMake ≥ 3.20 and EGL headers. Nothing else — no
distribution-specific packages are assumed, and no Python or code generation
step exists.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```

Dependencies are resolved in two tiers: your distribution's copy first, a pinned
download second. Vulkan-Headers is header-only and is fetched if not found;
EGL comes from pkg-config.

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
  loader handshake, instance/device yes  M0
  properties, limits, memory model yes  M0
  compute dispatch                 no   M1
  render passes, pipelines, draws  no   M2
  swapchain / window presentation  no   M3
```

The launcher only sets `VK_ICD_FILENAMES` and `KNIZ_LOG`, so you can always do the
same by hand. The loader resolves the manifest's `library_path` relative to the
manifest's own directory, which is why the manifest is copied next to the built
library:

```sh
VK_ICD_FILENAMES=$PWD/build/src/kniz_icd.json vulkaninfo --summary
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

The test drives the ICD through the **real Vulkan loader**, not by calling it
directly. That is deliberate: the most damaging class of ICD bug — a
dispatchable object that does not reserve its first word for the loader's
dispatch pointer — is invisible when the ICD is called directly, and produces a
device handle that `vkCreateDevice` returns happily and every later call rejects.
See the note in [`docs/DESIGN.md`](docs/DESIGN.md).

## Layout

```
src/icd/     the ICD: bootstrap, object model, GL backend
icd/         ICD manifest installed for loader discovery
tests/       lifecycle test, driven through the loader
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
