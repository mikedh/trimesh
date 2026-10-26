// What every shader shares, after what `shaders.shader_source` generates.
const float PI = 3.14159265;

// the exact sRGB curves, which textures and Blender use too
vec3 srgb_to_linear(vec3 color) {
    return mix(color / 12.92, pow((color + 0.055) / 1.055, vec3(2.4)), step(0.04045, color));
}
vec3 linear_to_srgb(vec3 color) {
    return mix(color * 12.92, 1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, color));
}

// where a pixel of a depth image is in view space for any projection
vec3 depth_to_view(sampler2D depth, vec2 uv) {
    vec4 position = frame.inverse_projection * (vec4(uv, texture(depth, uv).r, 1.0) * 2.0 - 1.0);
    return position.xyz / position.w;
}

// the Khronos PBR Neutral tone mapper of linear light exposed, then as the
// sRGB a display shows: colors up to 0.8 are left alone and brighter ones
// roll off to white from here, after the 0.04 it takes off
// https://github.com/KhronosGroup/ToneMapping/tree/main/PBR_Neutral
const float START_COMPRESSION = 0.8 - 0.04;
vec3 to_display(vec3 light) {
    vec3 color = light * frame.exposure.x;
    float darkest = min(color.r, min(color.g, color.b));
    color -= mix(0.04, darkest - 6.25 * darkest * darkest, float(darkest < 0.08));
    float peak = max(color.r, max(color.g, color.b));
    if (peak >= START_COMPRESSION) {
        float headroom = 1.0 - START_COMPRESSION;
        float new_peak = 1.0 - headroom * headroom / (peak + headroom - START_COMPRESSION);
        // desaturated towards white by 0.15 of how much the peak came down
        float fade = 1.0 - 1.0 / (0.15 * (peak - new_peak) + 1.0);
        color = mix(color * (new_peak / peak), vec3(new_peak), fade);
    }
    return linear_to_srgb(clamp(color, 0.0, 1.0));
}

// the `EnvironmentLight` cube map, whose mip level `k` of `n` is blurred
// like a GGX reflection of roughness `k / (n - 1)` from a mirror at zero
uniform samplerCube environment;

// the environment's light towards a view space direction, blurred like
// a reflection off a surface of a GGX `roughness`
vec3 environment_light(vec3 direction, float roughness) {
    vec3 local = mat3(frame.environment_rotation) * direction;
    return textureLod(environment, local, roughness * frame.environment_color.w).rgb
        * frame.environment_color.rgb;
}
