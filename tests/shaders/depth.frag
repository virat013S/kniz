#version 450
// M3b test fragment shader: the colour the vertex stage asked for.
//
//   glslangValidator -V -o depth.frag.spv depth.frag
// The depth test happens after this
// shader runs, so a fragment that fails it must leave no trace at all.

layout(location = 0) in vec4 v_colour;
layout(location = 0) out vec4 o_colour;

void main() { o_colour = v_colour; }
