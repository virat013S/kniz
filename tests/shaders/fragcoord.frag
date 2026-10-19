#version 450

// M2b test fragment shader: write down where the fragment is.
//
// gl_FragCoord is the rasteriser's own answer, so nothing Kniz does on the way in
// can make it agree with the test by accident.  The three channels say:
//
//   x  gl_FragCoord.x / 64, so texel column c reads back as (c + 0.5) / 64
//   y  gl_FragCoord.y / 64, same for the row
//   z  gl_FragCoord.z, which is the fragment's depth
//
// y is the interesting one.  Vulkan measures a framebuffer from its top-left and
// OpenGL measures a framebuffer object from its bottom-left, so the two disagree
// about which row is row 0 -- but a Vulkan fragment at y = 0 and an OpenGL
// fragment at y = 0 are both in the first row of the image, because both APIs
// put a vertex at NDC y = -1 there.  A flip anywhere would make this channel
// come back as (63 - row + 0.5) / 64 instead.
//
// z is the other one: the shader writes 0.25 in Vulkan's clip space, which is
// 0.25 of the way through Vulkan's [0, 1] depth range.  Without SPIRV-Cross's
// clip space fixup, OpenGL would read the same vertex as 0.25 of the way through
// its [-1, 1] range and this channel would come back as 0.625.
//
// 64 is the test's image size.  A real application would pass it in a uniform,
// which is the next milestone; a literal keeps this shader from depending on the
// feature it is meant to test.
//
// Regenerate with:
//
//   glslangValidator -V -o fragcoord.vert.spv fragcoord.vert
//   glslangValidator -V -o fragcoord.frag.spv fragcoord.frag

layout(location = 0) out vec4 outCoordinate;

void main() {
    outCoordinate =
        vec4(gl_FragCoord.x / 64.0, gl_FragCoord.y / 64.0, gl_FragCoord.z, 1.0);
}
