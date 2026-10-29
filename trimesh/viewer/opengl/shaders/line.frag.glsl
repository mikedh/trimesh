// A line segment as an antialiased capsule with a one pixel edge fade.

in vec4 segment_color;
in vec2 pixel_offset;
in float half_length, view_depth, line_width_pixels;

// draw points as lit spheres
uniform bool points;
uniform float node_index;
out vec4 fragment_color;

void main() {
    // pixels from the segment
    float past_end = max(abs(pixel_offset.x) - half_length, 0.0);
    float distance = length(vec2(past_end, pixel_offset.y));
    float coverage = clamp(0.5 * line_width_pixels + 0.5 - distance, 0.0, 1.0);
    // depth and segmentation are not antialiased
    bool index = frame.output_mode.x == OUTPUT_INDEX;
    if (coverage <= 0.0 || (index && coverage < 0.5)) {
        discard;
    }
    if (index) {
        fragment_color = vec4(view_depth, node_index, 0.0, 1.0);
        return;
    }
    vec3 color = segment_color.rgb;
    if (points) {
        // a sphere normal lit from the upper left
        vec2 across = pixel_offset / max(0.5 * line_width_pixels, 1.0);
        vec3 normal = vec3(across, sqrt(max(1.0 - dot(across, across), 0.0)));
        vec3 light = normalize(vec3(-0.3, 0.4, 0.87));
        float lambert = max(dot(normal, light), 0.0);
        float highlight = pow(max(dot(normal, normalize(light + vec3(0.0, 0.0, 1.0))), 0.0), 40.0);
        color = color * (0.35 + 0.65 * lambert) + 0.4 * highlight;
    }
    // premultiplied alpha
    float alpha = segment_color.a * coverage;
    fragment_color = vec4(color * alpha, alpha);
}
