# Kniz

A Vulkan Installable Client Driver (ICD) implemented on top of OpenGL.

> **Name:** "Zink" mirrored. Zink translates OpenGL → Vulkan; Kniz attempts
> Vulkan → OpenGL.

## Status

**Pre-alpha.** Nothing compiles yet. This repo currently contains design notes
and an ICD manifest.

## What this is (and isn't)

Kniz is an *inverted* translation layer. It presents a Vulkan device to an
application, and implements that device's commands by driving an OpenGL
context.

It is **not** a path to native performance, and it will never reach the ~95%
of native that Zink achieves in its own direction. See
[`docs/DESIGN.md`](docs/DESIGN.md) for why that asymmetry is fundamental rather
than a matter of effort.

The honest, reachable goal:

> Run `vkcube`, the Khronos Vulkan samples, and simple Vulkan 1.x applications
> correctly on a desktop OpenGL 4.x core context.

## Prior art

- **[Ashes](https://github.com/DragonJoker/Ashes)** (MIT) — the same idea,
  mature. A Vulkan ICD / drop-in `libvulkan.so` replacement with OpenGL
  3.3–4.6, D3D11 and Vulkan backends. Runs on Linux and Windows. 817 commits
  since 2018 and still not CTS-conformant. Read this before writing code.
- **Zink** (Mesa) — OpenGL on Vulkan. The opposite direction. Valuable as a
  reference for how a Gallium-style translation layer is structured.
- **MoltenVK** — Vulkan on Metal. The best-in-class example of "Vulkan as a
  source API, something else as the target", and where its fidelity stops.

## Layout

```
icd/        ICD manifest JSON, installed for loader discovery
docs/       design notes
src/        (not yet created) the ICD implementation
```

## Usage (once it exists)

```sh
VK_ICD_FILENAMES=/path/to/icd/kniz_icd.json vkcube
```

Because Kniz registers as a *separate* device rather than replacing
`vulkan-intel`, both remain available — useful for A/B-ing our output against a
real driver.

## License

TBD.
