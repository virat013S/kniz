#version 450

// M2c test vertex shader: a triangle built from gl_VertexIndex alone, moved and
// scaled by a uniform block in descriptor set 0.
//
// There are no vertex buffers and no attributes here on purpose.  The only
// things that can move the geometry are the index builtin and the uniform block,
// so a wrong colour or a wrong position cannot be blamed on the vertex input
// path that M2b already tests.
//
// Regenerate with:
//
//   glslangValidator -V -o uniform.vert.spv uniform.vert
//   glslangValidator -V -o uniform.frag.spv uniform.frag

layout(set = 0, binding = 0, std140) uniform Transform {
    vec4 offset;
    vec4 scale;
} transform;

void main() {
    vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    vec2 position = positions[gl_VertexIndex];
    gl_Position = vec4(position * transform.scale.xy + transform.offset.xy,
                       0.0, 1.0);
}
