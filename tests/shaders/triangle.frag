#version 450

// M2b test fragment shader: the input colour, unchanged.
//
// The test computes the interpolated colour it expects on the CPU, from
// barycentric coordinates, so anything the rasteriser gets wrong about which
// pixels the triangle covers -- or which way up it is -- shows up as a numeric
// disagreement rather than as an image that still looks like a triangle.

layout(location = 0) in vec4 vColor;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vColor;
}
