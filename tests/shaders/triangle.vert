#version 450

// M2b test vertex shader: pass a position and a colour straight through.
//
// Nothing is transformed, so the rasteriser's own coordinate arithmetic is what
// the test is reading back.  That is deliberate: an interpolation test that also
// involved a projection matrix would fail ambiguously between the matrix, the
// depth range and the viewport.
//
// The depth is 0.5 rather than 0.  Vulkan's clip space runs z from 0 to w and
// OpenGL's from -w to w, so SPIRV-Cross rewrites gl_Position.z to 2z - w; half
// way along Vulkan's range is 0 in OpenGL's, which is as far from either clip
// plane as a point can be.  Writing 0 would leave the triangle exactly on
// OpenGL's near plane, where whether it survives is a question about boundary
// behaviour rather than about the vertex stage.
//
// The compiled form is checked in beside this file.  Regenerate with:
//
//   glslangValidator -V -o triangle.vert.spv triangle.vert
//   glslangValidator -V -o triangle.frag.spv triangle.frag

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 vColor;

void main() {
    gl_Position = vec4(inPosition, 0.5, 1.0);
    vColor = inColor;
}
