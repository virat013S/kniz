#version 450

// M2d test vertex shader: cover the viewport, and nothing else.
//
// The sampling position is taken from gl_FragCoord in the fragment shader rather
// than interpolated from here, so that the pixel a fragment writes and the texel
// it reads are tied together by the rasteriser instead of by an interpolator
// that could hide a convention error.
//
// Regenerate with:
//
//   glslangValidator -V -o texture.vert.spv texture.vert

void main() {
    vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
