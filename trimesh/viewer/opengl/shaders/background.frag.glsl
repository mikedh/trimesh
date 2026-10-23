#version 330 core
// What the camera sees behind the scene when it sees the environment:
// the environment in every direction, sharp, tone mapped like a surface.

// per-frame values from `shaders.FRAME`, including the environment
$frame_block

$shared_functions

out vec4 fragment_color;

void main() {
    // the view direction through this pixel, for any projection
    vec2 clip = gl_FragCoord.xy / frame.viewport_size.xy * 2.0 - 1.0;
    vec4 near = frame.inverse_projection * vec4(clip, -1.0, 1.0);
    vec4 far = frame.inverse_projection * vec4(clip, 1.0, 1.0);
    vec3 direction = normalize(far.xyz / far.w - near.xyz / near.w);
    // a very sharp kernel is the environment itself
    vec3 light = environment_light(direction, 1e4);
    fragment_color = vec4(to_display(light), 1.0);
}
