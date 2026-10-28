#version 450
// M3b test vertex shader: one triangle covering the whole render area, at whatever
// depth the push constant names, so a test can draw the same geometry at several
// depths without touching a vertex buffer.
//
//   glslangValidator -V -o depth.vert.spv depth.vert
//
// The depth is written as Vulkan's gl_Position.z, in [0, 1]; the clip space fixup
// turns it into what GL expects, and the value that ends up in the depth buffer is
// the same number.  A test that writes 0.25 and reads back 16384 of 65535 is
// therefore checking the fixup as well as the depth path.
// therefore checking the fixup as well as the depth path.

//
// The geometry: three gl_VertexIndex values land on (-1, -1), (3, -1) and
// (-1, 3), which encloses the clip rectangle while keeping every fragment's
// interpolants exactly linear across the whole render area.

layout(push_constant) uniform Push {
  vec4 colour;
  float z;
} pc;

layout(location = 0) out vec4 v_colour;

void main() {
  // A single triangle large enough to cover the viewport: (-1, -1), (3, -1),
  // (-1, 3).  Vertices outside the clip volume are clipped away and what is left
  // covers every pixel.
  vec2 corner = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
  gl_Position = vec4(corner * 2.0 - 1.0, pc.z, 1.0);
  v_colour = pc.colour;
}
