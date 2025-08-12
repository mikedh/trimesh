#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float pointSize;

out vec4 VertexColor;

void main() {
    VertexColor = color;
    gl_PointSize = pointSize;
    gl_Position = projection * view * model * vec4(position, 1.0);
}
