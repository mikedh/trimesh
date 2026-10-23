#version 330 core
// One triangle covering the whole viewport with no vertex inputs:
// `glDrawArrays(GL_TRIANGLES, 0, 3)` for passes which shade pixels.

void main() {
    vec2 corner = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
