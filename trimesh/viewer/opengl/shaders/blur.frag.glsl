#version 330 core
// Average the 4x4 tile of noisy ambient occlusion around each pixel,
// which is exactly one of each spin `occlusion.frag` used, skipping
// pixels at another depth so darkness doesn't bleed across edges.

// per-frame values from `shaders.FRAME`, including the projection
$frame_block

uniform sampler2D noisy_occlusion;
uniform sampler2D scene_depth;

out vec4 fragment_color;

$shared_functions

// how far in front of the camera the middle of a pixel is
float view_depth(ivec2 pixel) {
    return depth_to_view(scene_depth, (vec2(pixel) + 0.5) / vec2(textureSize(scene_depth, 0))).z;
}

void main() {
    ivec2 size = textureSize(noisy_occlusion, 0);
    ivec2 center = ivec2(gl_FragCoord.xy);
    float depth = view_depth(center);
    float total = 0.0;
    float weight = 0.0;
    for (int x = -2; x < 2; x++) {
        for (int y = -2; y < 2; y++) {
            ivec2 pixel = clamp(center + ivec2(x, y), ivec2(0), size - 1);
            float same = step(abs(view_depth(pixel) - depth), 0.25 * frame.occlusion_radius.x);
            total += same * texelFetch(noisy_occlusion, pixel, 0).r;
            weight += same;
        }
    }
    fragment_color = vec4(total / max(weight, 1.0));
}
