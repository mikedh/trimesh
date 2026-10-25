#version 330 core
// Bloom: light too bright for the display glows onto the pixels around
// it, like glare in an eye or a lens. Drawn three times: across a mip
// level of a half size copy of the picture keeping only what the tone
// mapper compressed, then down that, then added onto the picture with
// `blur_direction` zero.

// per-frame values from `shaders.FRAME`, including the bloom
$frame_block

$shared_functions

// the half size picture, the blur across it, or the finished blur
uniform sampler2D bloom_source;
// one texel across, one down, or zero to add the blur to the picture
uniform vec2 blur_direction;
// read the picture at mip level `frame.bloom.z` and keep only what glows
uniform bool bright_pass;

out vec4 fragment_color;

void main() {
    if (blur_direction == vec2(0.0)) {
        // scaled up smoothly and added to the picture, leaving its alpha
        vec2 uv = gl_FragCoord.xy / frame.viewport_size.xy;
        fragment_color = vec4(frame.bloom.x * texture(bloom_source, uv).rgb, 0.0);
        return;
    }
    float level = 0.0;
    if (bright_pass) {
        level = frame.bloom.z;
    }
    vec2 texel = 1.0 / vec2(textureSize(bloom_source, int(level)));
    vec2 uv = gl_FragCoord.xy * texel;
    // the tone mapper only compresses light brighter than its start, which
    // the display shows as this or brighter: that glows, fading in from
    // nothing here to all of it at white
    float threshold = linear_to_srgb(vec3(START_COMPRESSION)).x;
    // a Gaussian out to three standard deviations
    float spread = frame.bloom.y;
    int radius = int(ceil(3.0 * spread));
    vec3 total = vec3(0.0);
    float weights = 0.0;
    for (int tap = -radius; tap <= radius; tap++) {
        vec3 color = textureLod(bloom_source, uv + float(tap) * blur_direction * texel, level).rgb;
        if (bright_pass) {
            float peak = max(color.r, max(color.g, color.b));
            color *= clamp((peak - threshold) / (1.0 - threshold), 0.0, 1.0);
        }
        float weight = exp(-0.5 * float(tap * tap) / (spread * spread));
        total += weight * color;
        weights += weight;
    }
    fragment_color = vec4(total / weights, 1.0);
}
