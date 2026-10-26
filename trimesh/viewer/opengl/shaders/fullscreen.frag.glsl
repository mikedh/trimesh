// Every pass of `shaders.PASSES` over a triangle covering its output,
// told apart by `pass_index`, reading the samplers its inputs name.
uniform int pass_index;
uniform sampler2D scene_depth, noisy_occlusion, bloom_source;

out vec4 fragment_color;

// how far in front of the camera the middle of a pixel of `scene_depth` is
float view_depth(ivec2 pixel) {
    return depth_to_view(scene_depth, (vec2(pixel) + 0.5) / vec2(textureSize(scene_depth, 0))).z;
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (pass_index == PASS_BACKGROUND) {
        // the environment through this pixel for any projection, sharp,
        // tone mapped like a surface
        vec2 clip = gl_FragCoord.xy / frame.viewport_size.xy * 2.0 - 1.0;
        vec4 near = frame.inverse_projection * vec4(clip, -1.0, 1.0);
        vec4 far = frame.inverse_projection * vec4(clip, 1.0, 1.0);
        vec3 direction = normalize(far.xyz / far.w - near.xyz / near.w);
        fragment_color = vec4(to_display(environment_light(direction, 0.0)), 1.0);
    } else if (pass_index == PASS_OCCLUSION) {
        // how much of the sky above a pixel nearby geometry hides, from 16
        // probes with a spin changing over a 4x4 tile `PASS_BLUR` averages
        vec2 uv = gl_FragCoord.xy / vec2(textureSize(scene_depth, 0));
        float radius = frame.occlusion_radius.x;
        vec3 position = depth_to_view(scene_depth, uv);
        // the surface's tangent frame from neighboring pixels
        vec3 along = normalize(dFdx(position));
        vec3 normal = normalize(cross(along, dFdy(position)));
        vec3 across = cross(normal, along);
        // where nothing was drawn nothing is hidden
        if (texture(scene_depth, uv).r == 1.0) {
            fragment_color = vec4(1.0);
            return;
        }
        ivec2 tile = pixel % 4;
        float spin = float(tile.x * 4 + tile.y) * PI / 8.0;
        float hidden = 0.0;
        for (int index = 0; index < 16; index++) {
            // cosine weighted directions on a golden angle spiral, with
            // more of the probes close to the surface
            float fraction = (float(index) + 0.5) / 16.0;
            float angle = spin + float(index) * 2.39996;
            vec3 direction = (along * cos(angle) + across * sin(angle)) * sqrt(fraction)
                + normal * sqrt(1.0 - fraction);
            vec3 probe = position + direction * radius * mix(0.1, 1.0, fraction * fraction);
            vec4 projected = frame.projection * vec4(probe, 1.0);
            vec3 seen = depth_to_view(scene_depth, projected.xy / projected.w * 0.5 + 0.5);
            // something in front of the probe and within reach hides the sky
            hidden += step(probe.z + 0.02 * radius, seen.z) * step(abs(seen.z - position.z), radius);
        }
        fragment_color = vec4(1.0 - hidden / 16.0);
    } else if (pass_index == PASS_BLUR) {
        // the 4x4 tile around each pixel, which is one of each spin,
        // skipping pixels at another depth so darkness doesn't bleed
        ivec2 size = textureSize(noisy_occlusion, 0);
        float depth = view_depth(pixel);
        vec2 total = vec2(0.0);
        for (int x = -2; x < 2; x++) {
            for (int y = -2; y < 2; y++) {
                ivec2 nearby = clamp(pixel + ivec2(x, y), ivec2(0), size - 1);
                float same = step(abs(view_depth(nearby) - depth), 0.25 * frame.occlusion_radius.x);
                total += same * vec2(texelFetch(noisy_occlusion, nearby, 0).r, 1.0);
            }
        }
        fragment_color = vec4(total.x / max(total.y, 1.0));
    } else if (pass_index == PASS_BLOOM_ADD) {
        // scaled up smoothly and added to the frame, leaving its alpha
        vec2 uv = gl_FragCoord.xy / frame.viewport_size.xy;
        fragment_color = vec4(frame.bloom.x * texture(bloom_source, uv).rgb, 0.0);
    } else {
        // a Gaussian out to three standard deviations across the mip level
        // `frame.bloom.z` of the frame, keeping only what the tone mapper
        // compressed, then down that
        float across = float(pass_index == PASS_BLOOM_ACROSS);
        float level = frame.bloom.z * across;
        vec2 texel = 1.0 / vec2(textureSize(bloom_source, int(level)));
        vec2 direction = vec2(across, 1.0 - across) * texel;
        // what the tone mapper compressed shows as this or brighter: that
        // glows, fading in from nothing here to all of it at white
        float threshold = linear_to_srgb(vec3(START_COMPRESSION)).x;
        float spread = frame.bloom.y;
        int radius = int(ceil(3.0 * spread));
        vec3 total = vec3(0.0);
        float weights = 0.0;
        for (int tap = -radius; tap <= radius; tap++) {
            vec3 color = textureLod(bloom_source, gl_FragCoord.xy * texel + float(tap) * direction, level).rgb;
            float peak = max(color.r, max(color.g, color.b));
            color *= mix(1.0, clamp((peak - threshold) / (1.0 - threshold), 0.0, 1.0), across);
            float weight = exp(-0.5 * float(tap * tap) / (spread * spread));
            total += weight * color;
            weights += weight;
        }
        fragment_color = vec4(total / weights, 1.0);
    }
}
