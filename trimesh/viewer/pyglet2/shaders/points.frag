#version 330 core

in vec4 VertexColor;
out vec4 FragColor;

void main() {
    // Make points circular
    vec2 coord = gl_PointCoord - vec2(0.5);
    if(dot(coord, coord) > 0.25) {
        discard;
    }
    FragColor = VertexColor;
}
