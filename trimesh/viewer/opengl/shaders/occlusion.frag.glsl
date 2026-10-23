#version 330 core
// Screen space ambient occlusion: how much of the sky above a pixel's
// surface nearby geometry hides, from the depth of the scene alone.
// Every pixel probes 16 points spread over the hemisphere above its
// surface with a spin that changes over a 4x4 tile, which `blur.frag`
// averages away.

// per-frame values from `shaders.FRAME`, including the projection
$frame_block

uniform sampler2D scene_depth;

out vec4 fragment_color;

$shared_functions

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(scene_depth, 0));
    float occlusion_radius = frame.occlusion_radius.x;
    vec3 position = depth_to_view(scene_depth, uv);
    // the surface's tangent frame from neighboring pixels
    vec3 along = normalize(dFdx(position));
    vec3 normal = normalize(cross(along, dFdy(position)));
    vec3 across = cross(normal, along);
    // nothing was drawn here so nothing is hidden
    if (texture(scene_depth, uv).r == 1.0) {
        fragment_color = vec4(1.0);
        return;
    }
    ivec2 tile = ivec2(gl_FragCoord.xy) % 4;
    float spin = float(tile.x * 4 + tile.y) * PI / 8.0;
    float hidden = 0.0;
    for (int index = 0; index < 16; index++) {
        // cosine weighted directions on a golden angle spiral, with
        // more of the probes close to the surface
        float fraction = (float(index) + 0.5) / 16.0;
        float angle = spin + float(index) * 2.39996;
        vec3 direction = (along * cos(angle) + across * sin(angle)) * sqrt(fraction)
            + normal * sqrt(1.0 - fraction);
        vec3 probe = position + direction * occlusion_radius * mix(0.1, 1.0, fraction * fraction);
        vec4 projected = frame.projection * vec4(probe, 1.0);
        vec3 seen = depth_to_view(scene_depth, projected.xy / projected.w * 0.5 + 0.5);
        // something in front of the probe and within reach hides the sky
        hidden += step(probe.z + 0.02 * occlusion_radius, seen.z)
            * step(abs(seen.z - position.z), occlusion_radius);
    }
    fragment_color = vec4(1.0 - hidden / 16.0);
}
