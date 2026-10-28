#version 450

// M2d test fragment shader: sample an 8x8 image across a 64x64 target.
//
// gl_FragCoord.xy / 64 is the pixel centre in [0, 1], so each 8x8 block of
// pixels lands inside one texel of the source.  With NEAREST that is exactly
// that texel; with LINEAR the sample position sits a fixed fraction of the way
// between two texels, which is also exact and is what makes the filtering
// testable without a tolerance worth arguing about.
//
// There is no y flip here.  The M2b test established that gl_FragCoord.y / 64
// and the readback row r / 64 are the same number, so pixel row r reads texture
// row r / 8 and the image arrives the right way up.  A flip anywhere in the
// driver, in either the image path or the swapchain-shaped one, would show up as
// the pattern upside down.
//
// Regenerate with:
//
//   glslangValidator -V -o texture.frag.spv texture.frag

layout(set = 0, binding = 0) uniform sampler2D source;

layout(location = 0) out vec4 colour;

void main() {
    colour = texture(source, gl_FragCoord.xy / 64.0);
}
