#pragma once

// The OpenGL backend.
//
// Kniz owns its GL context rather than borrowing one from the application:
// the application is talking Vulkan and should not have to care that OpenGL
// exists.  M0 creates a headless (window system independent) context purely to
// prove the backend is reachable and to report what we actually got.
//
// Later milestones replace/extend this with:
//   - one context per Vulkan calling thread (GL state is per-thread and
//     "current", while Vulkan is thread agnostic), and
//   - a window-system surface for the swapchain (see docs/DESIGN.md, WSI).

#include <cstddef>
#include <cstdint>
#include <string>

namespace kniz::gl {

// GL's own scalar types, declared locally.  Including <GL/gl.h> would risk
// picking up a GL or GLES header that disagrees with the EGL we resolved
// against, and we only need the shape of a handful of entry points.
using Enum = unsigned int;
using Uint = unsigned int;
using Int = int;
using Size = int;
using SizePtr = std::ptrdiff_t;
using IntPtr = std::ptrdiff_t;
using Char = char;

struct ContextInfo {
  bool valid = false;
  int major = 0;
  int minor = 0;
  bool core = false;
  std::string vendor;
  std::string renderer;
  std::string version;
  std::string shading_language;

  // Compute shaders and shader storage blocks are core from 4.3.  Checked from
  // the version string rather than from the entry points, because a driver can
  // export glDispatchCompute on a context that is not allowed to call it.
  bool compute_capable = false;

  // Textures and, later, framebuffers: the image half of the backend.  False on
  // a context that cannot back a Vulkan image, in which case image creation is
  // refused with a message rather than producing something that does not render.
  bool texture_capable = false;

  // GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT.  Reported to applications as
  // minStorageBufferOffsetAlignment, and checked when a descriptor is bound:
  // Vulkan caps that limit at 256 while GL's own minimum is also 256, so if the
  // driver wants more, some otherwise legal offsets cannot be represented.
  int ssbo_offset_alignment = 0;
};

// Create a headless GL context and make it current on the calling thread.
// Idempotent: repeated calls are cheap no-ops.  Returns false on failure, in
// which case the reason has already been logged.
bool create_headless();

// Release the context.  Safe to call when nothing was created.
void destroy();

// Describes the context returned by the most recent create_headless().
const ContextInfo& info();

// Make the context current on the calling thread.  Every public entry point
// calls this first, because GL state is per-thread and the application is free
// to call Vulkan from any thread.  Returns false if there is no context.
bool make_current();

// ---------------------------------------------------------------------------
// Entry point table
//
// Deliberately not GLAD: a generated loader would add a Python/codegen step to
// every build.  We resolve what we need through eglGetProcAddress at runtime,
// which also means we do not link libGL and therefore cannot conflict with the
// application's own GL stack.
//
// Only the functions the current milestone uses are declared.  Add to this
// struct as the backend grows.
// ---------------------------------------------------------------------------

struct Functions {
  // ---- M0 ----------------------------------------------------------------
  const unsigned char* (*GetString)(Enum name) = nullptr;
  void (*GetIntegerv)(Enum pname, Int* data) = nullptr;
  Enum (*GetError)() = nullptr;

  // ---- M1: buffer objects ------------------------------------------------
  void (*GenBuffers)(Size n, Uint* buffers) = nullptr;
  void (*DeleteBuffers)(Size n, const Uint* buffers) = nullptr;
  void (*BindBuffer)(Enum target, Uint buffer) = nullptr;
  void (*BufferData)(Enum target, SizePtr size, const void* data, Enum usage) = nullptr;
  void (*BufferSubData)(Enum target, IntPtr offset, SizePtr size, const void* data) = nullptr;
  void (*GetBufferSubData)(Enum target, IntPtr offset, SizePtr size, void* data) = nullptr;
  void (*BindBufferBase)(Enum target, Uint index, Uint buffer) = nullptr;
  void (*BindBufferRange)(Enum target, Uint index, Uint buffer, IntPtr offset,
                          SizePtr size) = nullptr;
  void (*CopyBufferSubData)(Enum readTarget, Enum writeTarget, IntPtr readOffset,
                            IntPtr writeOffset, SizePtr size) = nullptr;

  // ---- M1: shaders and programs -----------------------------------------
  Uint (*CreateShader)(Enum type) = nullptr;
  void (*ShaderSource)(Uint shader, Size count, const Char* const* string,
                       const Int* length) = nullptr;
  void (*CompileShader)(Uint shader) = nullptr;
  void (*GetShaderiv)(Uint shader, Enum pname, Int* params) = nullptr;
  void (*GetShaderInfoLog)(Uint shader, Size bufSize, Size* length, Char* infoLog) = nullptr;
  void (*DeleteShader)(Uint shader) = nullptr;
  Uint (*CreateProgram)() = nullptr;
  void (*AttachShader)(Uint program, Uint shader) = nullptr;
  void (*LinkProgram)(Uint program) = nullptr;
  void (*GetProgramiv)(Uint program, Enum pname, Int* params) = nullptr;
  void (*GetProgramInfoLog)(Uint program, Size bufSize, Size* length, Char* infoLog) = nullptr;
  void (*DeleteProgram)(Uint program) = nullptr;
  void (*UseProgram)(Uint program) = nullptr;

  // ---- M1: compute and synchronisation -----------------------------------
  void (*DispatchCompute)(Uint x, Uint y, Uint z) = nullptr;
  void (*MemoryBarrier)(Uint barriers) = nullptr;
  void (*Finish)() = nullptr;
  void (*Flush)() = nullptr;

  // ---- M2: textures -------------------------------------------------------
  //
  // Vulkan images become GL textures.  Unlike buffers there is no host mirror:
  // a texture's storage is defined by GL and does not correspond to any layout
  // Vulkan describes, so all writes and reads go through commands.  See
  // kniz_image.cpp.
  void (*GenTextures)(Size n, Uint* textures) = nullptr;
  void (*DeleteTextures)(Size n, const Uint* textures) = nullptr;
  void (*BindTexture)(Enum target, Uint texture) = nullptr;
  void (*ActiveTexture)(Enum texture) = nullptr;
  void (*TexParameteri)(Enum target, Enum pname, Int param) = nullptr;
  void (*TexStorage2D)(Enum target, Size levels, Enum internalformat, Size width,
                       Size height) = nullptr;
  void (*TexSubImage2D)(Enum target, Int level, Int xoffset, Int yoffset,
                        Size width, Size height, Enum format, Enum type,
                        const void* pixels) = nullptr;
  void (*GetTexImage)(Enum target, Int level, Enum format, Enum type,
                      void* pixels) = nullptr;
  void (*ClearTexImage)(Enum texture, Int level, Enum format, Enum type,
                        const void* data) = nullptr;
  void (*PixelStorei)(Enum pname, Int param) = nullptr;
};

extern Functions f;

// GL reports one error at a time and keeps reporting it until it is read, so a
// single check after a sequence of calls can be blamed on the wrong call.
// Draining first and reporting everything found is the only way to attribute a
// failure to it, and returning whether anything was found lets a caller that has
// to distrust its own result act on it.
bool report_errors(const char* where);

// GL's size parameters are signed; a Vulkan size is unsigned and one the
// application has already validated against an object can still exceed what a
// pointer difference can represent.  Saturating is not a silent truncation: no
// GL call could name such a size anyway.
inline SizePtr as_size_ptr(uint64_t bytes) {
  constexpr uint64_t kMax = static_cast<uint64_t>(PTRDIFF_MAX);
  return static_cast<SizePtr>(bytes > kMax ? kMax : bytes);
}

// ---------------------------------------------------------------------------
// GL tokens used by the backend
//
// Spelled out locally for the same reason the types are.  Only the ones Kniz
// actually passes to GL appear here.
//
// Transcribe these from a real header, never from memory: the storage buffer
// target is 0x90D2 and the *block* interface block token is 0x90D3, and using
// the wrong one produces GL_INVALID_ENUM at the point of use rather than at the
// point of definition.  The values below were checked against
// /usr/include/GL/glcorearb.h.  Anything added here should be too.
// ---------------------------------------------------------------------------

inline constexpr Enum kArrayBuffer = 0x8892;
inline constexpr Enum kShaderStorageBuffer = 0x90D2;
inline constexpr Enum kUniformBuffer = 0x8A11;
inline constexpr Enum kCopyReadBuffer = 0x8F36;
inline constexpr Enum kCopyWriteBuffer = 0x8F37;

inline constexpr Enum kComputeShader = 0x91B9;
inline constexpr Enum kVertexShader = 0x8B31;
inline constexpr Enum kFragmentShader = 0x8B30;

inline constexpr Enum kStreamDraw = 0x88E0;
inline constexpr Enum kStaticDraw = 0x88E4;
inline constexpr Enum kDynamicDraw = 0x88E8;

inline constexpr Enum kCompileStatus = 0x8B81;
inline constexpr Enum kLinkStatus = 0x8B82;
inline constexpr Enum kInfoLogLength = 0x8B84;

inline constexpr Enum kNoError = 0;
inline constexpr Uint kAllBarrierBits = 0xFFFFFFFFu;

// ---- M2: textures ---------------------------------------------------------
//
// Sized internal formats are what glTexStorage2D wants; the external formats
// and types are what reads and writes use.  Both are needed because GL keeps
// them separate and the round trip has to use the same pair on both sides.

inline constexpr Enum kTexture2D = 0x0DE1;
inline constexpr Enum kTexture0 = 0x84C0;
inline constexpr Enum kMaxTextureSize = 0x0D33;

inline constexpr Enum kTextureMinFilter = 0x2801;
inline constexpr Enum kTextureMagFilter = 0x2800;
inline constexpr Enum kTextureWrapS = 0x2802;
inline constexpr Enum kTextureWrapT = 0x2803;
inline constexpr Int kNearest = 0x2600;
inline constexpr Int kLinear = 0x2601;
inline constexpr Int kClampToEdge = 0x812F;
inline constexpr Enum kTextureBaseLevel = 0x813C;
inline constexpr Enum kTextureMaxLevel = 0x813D;

inline constexpr Enum kUnpackAlignment = 0x0CF5;
inline constexpr Enum kUnpackRowLength = 0x0CF2;
inline constexpr Enum kPackAlignment = 0x0D05;

inline constexpr Enum kR8 = 0x8229;
inline constexpr Enum kRg8 = 0x822B;
inline constexpr Enum kRgb8 = 0x8051;
inline constexpr Enum kRgba8 = 0x8058;
inline constexpr Enum kSrgb8Alpha8 = 0x8C43;
inline constexpr Enum kR32f = 0x822E;
inline constexpr Enum kRgba16f = 0x881A;
inline constexpr Enum kRgba32f = 0x8814;
inline constexpr Enum kDepthComponent16 = 0x81A5;
inline constexpr Enum kDepthComponent24 = 0x81A6;
inline constexpr Enum kDepthComponent32f = 0x8CAC;
inline constexpr Enum kDepth24Stencil8 = 0x88F0;

inline constexpr Enum kRed = 0x1903;
inline constexpr Enum kRg = 0x8227;
inline constexpr Enum kRgb = 0x1907;
inline constexpr Enum kRgba = 0x1908;
inline constexpr Enum kDepthComponent = 0x1902;
inline constexpr Enum kDepthStencil = 0x84F9;
inline constexpr Enum kUnsignedByte = 0x1401;
inline constexpr Enum kUnsignedShort = 0x1403;
inline constexpr Enum kUnsignedInt = 0x1405;
inline constexpr Enum kFloat = 0x1406;
inline constexpr Enum kHalfFloat = 0x140B;
inline constexpr Enum kBgra = 0x80E1;
inline constexpr Enum kUnsignedInt248 = 0x84FA;

}  // namespace kniz::gl
