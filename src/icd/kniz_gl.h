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
using Float = float;
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

  // Framebuffers, vertex arrays and draws, i.e. everything M2b needs to render
  // into an image.  Framebuffer objects and vertex array objects are core in
  // 3.0, but the set also wants glDrawBuffers, so this is the "can Kniz draw"
  // flag rather than a version test.
  bool render_capable = false;

  // Uniform buffer objects.  Core in 3.1, so in practice this is available
  // whenever anything else Kniz does is: it is checked rather than assumed
  // because a 3.0 context with buffers and shaders is a configuration the rest
  // of the driver tolerates.
  bool uniform_capable = false;

  // GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT.  Reported to applications as
  // minStorageBufferOffsetAlignment, and checked when a descriptor is bound:
  // Vulkan caps that limit at 256 while GL's own minimum is also 256, so if the
  // driver wants more, some otherwise legal offsets cannot be represented.
  int ssbo_offset_alignment = 0;

  // GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, the same problem in the uniform
  // namespace.  Vulkan's own maximum for minUniformBufferOffsetAlignment is also
  // 256, and this driver reports 256, so no legal offset is refused here unless
  // a GL driver wanted more than the specification permits.
  int ubo_offset_alignment = 0;

  // Sampler objects, i.e. glGenSamplers and friends.  Core in 3.3, so this is
  // the point at which a Vulkan sampler can be represented at all: without one,
  // filtering and addressing would have to live on the texture, which is
  // per-image state rather than per-sampler state and would make two samplers
  // over one image indistinguishable.
  bool sampler_capable = false;

  // GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, which bounds the texture-unit
  // namespace the same way GL_MAX_*_BUFFER_BINDINGS bounds the buffer ones.
  int max_texture_units = 0;

  // Whether this context can present to a window: an EGL display that creates
  // window surfaces and a config that supports them.  False on the surfaceless
  // platform, which is the normal case in a container or a CI runner, and where
  // a swapchain has to be refused rather than created and never shown.
  bool window_capable = false;
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
// M3: presenting to a window
//
// A swapchain image is a GL texture like any other, so drawing into it needs
// nothing new.  Getting it onto the screen does: GL cannot make a texture *be*
// the window's front buffer, so presentation is a blit from the image to the
// default framebuffer of an EGL window surface, followed by eglSwapBuffers.
//
// The window surface is created on the same EGL display and from the same
// config as the context.  That is what lets one context draw to both: the GL
// objects an application created stay valid, and only the drawable changes.
// ---------------------------------------------------------------------------

// An EGL window surface, opaque outside kniz_gl.cpp.
using WindowSurface = void*;

// The window system the GL context's EGL display belongs to.
//
// This matters because a GL context lives on exactly one EGL display, and its
// objects cannot be moved to another: presenting to a window from a different
// window system is not a matter of opening a second display but of rebuilding
// every texture and buffer the application owns.  So the platform is decided
// when the context is created, and a surface from any other platform is refused
// rather than handed to EGL, which would treat the handle as something it is not
// -- on Mesa, as a pointer.
enum class WindowPlatform {
  None,      // no display at all
  X11,       // X11 or XWayland
  Wayland,   // Wayland
  Other,     // some other platform, or one that could not be identified
};

// Which platform the current context belongs to.  None before create_headless.
WindowPlatform window_platform();

// Create a window surface for a native window handle, which is an X11 XID on
// every platform Kniz supports so far.  Returns nullptr on failure, having
// logged why.
WindowSurface create_window_surface(uint64_t native_window);

void destroy_window_surface(WindowSurface surface);

// Make the window surface current on this thread, and afterwards go back to the
// headless one.  Between the two the default framebuffer is the window.
bool bind_window_surface(WindowSurface surface);
bool bind_headless_surface();

// The drawable's size in pixels.  It is the window's, not the swapchain's: the
// two can differ, and the blit scales.
bool window_surface_size(WindowSurface surface, int* width, int* height);

// eglSwapBuffers.
bool swap_window_surface(WindowSurface surface);

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

  // ---- M2d: samplers ------------------------------------------------------
  //
  // A Vulkan sampler is filter, wrap, LOD and border state with no image in it,
  // which is exactly what a GL sampler object is.  The alternative -- setting
  // the state on the texture -- would make two samplers over one image
  // impossible to distinguish, which is the whole point of the Vulkan object.
  void (*GenSamplers)(Size n, Uint* samplers) = nullptr;
  void (*DeleteSamplers)(Size n, const Uint* samplers) = nullptr;
  void (*BindSampler)(Uint unit, Uint sampler) = nullptr;
  void (*SamplerParameteri)(Uint sampler, Enum pname, Int param) = nullptr;
  void (*SamplerParameterf)(Uint sampler, Enum pname, Float param) = nullptr;
  void (*SamplerParameterfv)(Uint sampler, Enum pname, const Float* param) = nullptr;

  // ---- M2b: framebuffers, vertex arrays and draws -------------------------
  //
  // A Vulkan render pass and framebuffer become a GL framebuffer object whose
  // attachments are the images' own textures; a Vulkan pipeline's vertex input
  // becomes attribute pointers captured in a vertex array object.  Neither has a
  // host-side representation, so both are rebuilt when the recorded command stream
  // is replayed.
  void (*GenFramebuffers)(Size n, Uint* framebuffers) = nullptr;
  void (*DeleteFramebuffers)(Size n, const Uint* framebuffers) = nullptr;
  void (*BindFramebuffer)(Enum target, Uint framebuffer) = nullptr;
  void (*FramebufferTexture2D)(Enum target, Enum attachment, Enum textarget,
                               Uint texture, Int level) = nullptr;
  Enum (*CheckFramebufferStatus)(Enum target) = nullptr;
  void (*DrawBuffers)(Size n, const Enum* bufs) = nullptr;
  void (*ClearBufferfv)(Enum buffer, Int drawbuffer, const float* value) = nullptr;
  // The depth clear needs a different entry point from the colour one, and a
  // combined depth/stencil clear needs this one, which takes both values at once
  // because GL will not clear one aspect of a packed format alone.
  void (*ClearBufferfi)(Enum buffer, Int drawbuffer, float depth, Int stencil) =
      nullptr;

  void (*GenVertexArrays)(Size n, Uint* arrays) = nullptr;
  void (*DeleteVertexArrays)(Size n, const Uint* arrays) = nullptr;
  void (*BindVertexArray)(Uint array) = nullptr;
  void (*EnableVertexAttribArray)(Uint index) = nullptr;
  void (*DisableVertexAttribArray)(Uint index) = nullptr;
  void (*VertexAttribPointer)(Uint index, Int size, Enum type,
                              unsigned char normalized, Size stride,
                              const void* pointer) = nullptr;

  void (*DrawArrays)(Enum mode, Int first, Size count) = nullptr;
  void (*DrawElements)(Enum mode, Size count, Enum type,
                       const void* indices) = nullptr;
  // Vulkan's vkCmdDrawIndexed has a vertexOffset that GL's plain glDrawElements
  // does not; this is the entry point that does, and it is core in 3.2.
  void (*DrawElementsBaseVertex)(Enum mode, Size count, Enum type,
                                 const void* indices, Int base_vertex) = nullptr;

  void (*Viewport)(Int x, Int y, Size width, Size height) = nullptr;
  void (*Scissor)(Int x, Int y, Size width, Size height) = nullptr;
  void (*Enable)(Enum cap) = nullptr;
  void (*Disable)(Enum cap) = nullptr;
  void (*CullFace)(Enum mode) = nullptr;
  void (*FrontFace)(Enum mode) = nullptr;
  void (*ColorMask)(unsigned char red, unsigned char green, unsigned char blue,
                    unsigned char alpha) = nullptr;
  // Depth state, which Vulkan puts in the pipeline and GL keeps in the context.
  void (*DepthMask)(unsigned char flag) = nullptr;
  void (*DepthFunc)(Enum func) = nullptr;
  void (*DepthRange)(double near_value, double far_value) = nullptr;

  // ---- M3: presentation ---------------------------------------------------
  //
  // The only GL operation presentation needs that drawing did not: a blit from
  // the swapchain image's texture to the window's default framebuffer.  Core
  // since 3.0, and it is the whole of vkQueuePresentKHR besides the swap.
  // Used only by the KNIZ_PRESENT_DUMP debug path, which reads the frame that
  // is about to be shown so that presentation can be tested without a person
  // looking at a window.
  void (*ReadPixels)(Int x, Int y, Size width, Size height, Enum format,
                     Enum type, void* pixels) = nullptr;
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
inline constexpr Enum kTextureWrapR = 0x8072;
inline constexpr Enum kTextureBorderColor = 0x1004;
inline constexpr Enum kTextureMinLod = 0x813A;
inline constexpr Enum kTextureMaxLod = 0x813B;
inline constexpr Enum kTextureLodBias = 0x8501;
inline constexpr Int kNearest = 0x2600;
inline constexpr Int kLinear = 0x2601;
inline constexpr Int kNearestMipmapNearest = 0x2700;
inline constexpr Int kLinearMipmapNearest = 0x2701;
inline constexpr Int kNearestMipmapLinear = 0x2702;
inline constexpr Int kLinearMipmapLinear = 0x2703;
inline constexpr Int kRepeat = 0x2901;
inline constexpr Int kMirroredRepeat = 0x8370;
inline constexpr Int kClampToEdge = 0x812F;
inline constexpr Int kClampToBorder = 0x812D;
inline constexpr Enum kMaxCombinedTextureImageUnits = 0x8B4D;
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

// ---- M2b: framebuffers, vertex arrays and drawing -------------------------
//
// The GL tokens for rendering.  Same rule as above: transcribed from
// /usr/include/GL/glcorearb.h, never from memory.

inline constexpr Enum kFramebuffer = 0x8D40;
inline constexpr Enum kDrawFramebuffer = 0x8CA9;
inline constexpr Enum kReadFramebuffer = 0x8CA8;
inline constexpr Enum kColorAttachment0 = 0x8CE0;
inline constexpr Enum kFramebufferComplete = 0x8CD5;
inline constexpr Enum kColorBufferBit = 0x00004000;
inline constexpr Enum kColor = 0x1800;
inline constexpr Enum kDepth = 0x1801;
inline constexpr Enum kDepthAttachment = 0x8D00;
inline constexpr Enum kDepthStencilAttachment = 0x821A;
inline constexpr Enum kDepthBufferBit = 0x00000100;

inline constexpr Enum kVertexArray = 0x8074;
inline constexpr Enum kElementArrayBuffer = 0x8893;

inline constexpr Enum kPoints = 0x0000;
inline constexpr Enum kLines = 0x0001;
inline constexpr Enum kLineStrip = 0x0003;
inline constexpr Enum kTriangles = 0x0004;
inline constexpr Enum kTriangleStrip = 0x0005;
inline constexpr Enum kTriangleFan = 0x0006;

inline constexpr Enum kByte = 0x1400;
inline constexpr Enum kShort = 0x1402;
inline constexpr Enum kInt = 0x1404;

inline constexpr Enum kCullFace = 0x0B44;
inline constexpr Enum kScissorTest = 0x0C11;
// Enables that Vulkan spells as pipeline state and GL keeps in the context.
// Presentation has to clear them explicitly, because the last pipeline an
// application drew with may have left any of them on.
inline constexpr Enum kDepthTest = 0x0B71;
inline constexpr Enum kStencilTest = 0x0B90;
inline constexpr Enum kBlend = 0x0BE2;
// GL's own sRGB framebuffer encoding, off by default and never enabled by Kniz.
// Presentation turns it off rather than trusting the default, because a swapchain
// format that is not sRGB must not be encoded on the way out.
inline constexpr Enum kFramebufferSrgb = 0x8DB9;
// Dithering has no Vulkan counterpart, and it perturbs a clear value in the
// least significant bit rather than writing it exactly.
inline constexpr Enum kDither = 0x0BD0;
inline constexpr Enum kBack = 0x0405;
inline constexpr Enum kFront = 0x0404;
inline constexpr Enum kFrontAndBack = 0x0408;
// The depth comparison functions, in the order VkCompareOp declares them:
// NEVER, LESS, EQUAL, LESS_OR_EQUAL, GREATER, NOT_EQUAL, GREATER_OR_EQUAL,
// ALWAYS.  GL numbers the same eight the same way, one after another, which is
// why kniz_pipeline.cpp can index a table with a VkCompareOp.
inline constexpr Enum kNever = 0x0200;
inline constexpr Enum kLess = 0x0201;
inline constexpr Enum kEqual = 0x0202;
inline constexpr Enum kLessOrEqual = 0x0203;
inline constexpr Enum kGreater = 0x0204;
inline constexpr Enum kNotEqual = 0x0205;
inline constexpr Enum kGreaterOrEqual = 0x0206;
inline constexpr Enum kAlways = 0x0207;
inline constexpr Enum kCcw = 0x0901;
inline constexpr Enum kCw = 0x0900;
inline constexpr Uint kNone = 0;

}  // namespace kniz::gl
