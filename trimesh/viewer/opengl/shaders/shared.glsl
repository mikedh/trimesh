// Functions used by more than one shader, which are filled in after
// the frame block by `shaders.shader_source`.

const float PI = 3.14159265;

// the exact sRGB curves, which textures and Blender use too
vec3 srgb_to_linear(vec3 color) {
    return mix(color / 12.92, pow((color + 0.055) / 1.055, vec3(2.4)), step(0.04045, color));
}
vec3 linear_to_srgb(vec3 color) {
    return mix(color * 12.92, 1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, color));
}

// where a pixel of a depth image is in view space, which works for
// perspective and orthographic projections alike
vec3 depth_to_view(sampler2D depth, vec2 uv) {
    vec4 position = frame.inverse_projection * (vec4(uv, texture(depth, uv).r, 1.0) * 2.0 - 1.0);
    return position.xyz / position.w;
}

// the Khronos PBR Neutral tone mapper: colors up to 0.8 are left
// alone so base colors look like themselves, and brighter colors
// roll off towards white instead of clipping, which starts here
// after the 0.04 it takes off every color
// https://github.com/KhronosGroup/ToneMapping/tree/main/PBR_Neutral
const float START_COMPRESSION = 0.8 - 0.04;
vec3 tonemap(vec3 color) {
    const float desaturation = 0.15;
    float darkest = min(color.r, min(color.g, color.b));
    float offset = 0.04;
    if (darkest < 0.08) {
        offset = darkest - 6.25 * darkest * darkest;
    }
    color -= offset;
    float peak = max(color.r, max(color.g, color.b));
    if (peak < START_COMPRESSION) {
        return color;
    }
    float headroom = 1.0 - START_COMPRESSION;
    float new_peak = 1.0 - headroom * headroom / (peak + headroom - START_COMPRESSION);
    color *= new_peak / peak;
    float fade = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return mix(color, vec3(new_peak), fade);
}

// linear light as the sRGB a display shows, exposed then tone mapped
vec3 to_display(vec3 light) {
    return linear_to_srgb(clamp(tonemap(light * frame.exposure.x), 0.0, 1.0));
}

// the integral over the sphere of the product of two spherical Gaussians
// with peaks of one, written so it stays accurate when one is very sharp
float gaussian_overlap(vec3 axis_a, float sharpness_a, vec3 axis_b, float sharpness_b) {
    float combined = length(sharpness_a * axis_a + sharpness_b * axis_b);
    float exponent = 2.0 * sharpness_a * sharpness_b * (dot(axis_a, axis_b) - 1.0)
        / (combined + sharpness_a + sharpness_b);
    return 2.0 * PI * exp(exponent) * (1.0 - exp(-2.0 * combined)) / max(combined, 1e-6);
}

// the environment's light around a direction, averaged by a spherical
// Gaussian kernel of sharpness `blur` which integrates to one
vec3 environment_light(vec3 direction, float blur) {
    // the gradient is linear in height so averaging it only scales the
    // height by the kernel's mean cosine
    float spread = exp(-2.0 * blur);
    float mean_cosine = (1.0 + spread) / (1.0 - spread) - 1.0 / blur;
    float height = dot(direction, frame.up_direction.xyz) * mean_cosine;
    vec3 light = mix(frame.ground_color.rgb, frame.sky_color.rgb, 0.5 + 0.5 * height);
    float kernel_peak = blur / (2.0 * PI * (1.0 - spread));
    for (int index = 0; index < frame.softbox_count.x; index++) {
        Softbox softbox = frame.softboxes[index];
        light += softbox.color.rgb * kernel_peak
            * gaussian_overlap(softbox.direction.xyz, softbox.direction.w, direction, blur);
    }
    return light;
}
