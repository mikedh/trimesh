#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;
layout(location = 3) in vec4 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec4 VertexColor;

void main() {
    FragPos = vec3(model * vec4(position, 1.0));
    // Use model matrix for normals for now (works for uniform scaling)
    Normal = normalize(mat3(model) * normal);
    TexCoord = texCoord;
    VertexColor = color;
    
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
