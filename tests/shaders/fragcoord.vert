#version 450

// M2b test: a second vertex shader that builds its triangle out of nothing but
// gl_VertexIndex, covering the whole viewport.
//
// It exists to interrogate the rasteriser rather than to draw something.  With
// no vertex buffers and no attributes, the only things that can move the result
// are the viewport, the clip space convention and the index builtin -- which is
// exactly the set of conventions that a silently wrong lowering gets wrong.
//
// Deliberately not a quad: all three indices are needed to cover the viewport,
// so a gl_VertexIndex that always came back as zero would leave most of the
// image untouched.
//
// Regenerate with:
//
//   glslangValidator -V -o fragcoord.vert.spv fragcoord.vert
//   glslangValidator -V -o fragcoord.frag.spv fragcoord.frag

void main() {
    vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(positions[gl_VertexIndex], 0.25, 1.0);
}
