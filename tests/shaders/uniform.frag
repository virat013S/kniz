#version 450

// M2c test fragment shader: a flat colour from a uniform block in descriptor
// set 1, scaled by a push constant.
//
// The uniform block is in set 1 while the vertex shader's is in set 0, and both
// are at binding 0.  Vulkan allows that; GLSL has no sets, so the two must end up
// at *different* GL binding points.  A translation that used the binding number
// as the GLSL binding point would put both blocks at GL binding 0, and whichever
// buffer was bound last would be what the vertex shader read -- the geometry
// would move to wherever the fragment shader's data says it should.
//
// The push constant is where Kniz has the least to stand on: GL has no push
// constants at all, so they are lowered onto a uniform buffer Kniz owns.  A push
// constant that never arrives, or that arrives at the wrong offset, changes the
// colour rather than producing an obviously broken image.

layout(set = 1, binding = 0, std140) uniform Tint {
    vec4 base;
} tint;

layout(push_constant) uniform Push {
    vec4 weight;
} pc;

layout(location = 0) out vec4 outColour;

void main() {
    outColour = tint.base * pc.weight;
}
