// Shade a line segment rectangle as an antialiased capsule: full
// color inside the line and a one pixel fade at the edge.

in vec4 segment_color;
in vec2 pixel_offset;
in float half_length, view_depth, line_width_pixels;

// draw each segment, which is a point, as a little lit sphere
uniform bool points;
// which of `scene.graph.nodes_geometry` this is, for segmentation
uniform float node_index;

out vec4 fragment_color;

void main() {
    // distance in pixels from the segment, past either end or across it
    float past_end = max(abs(pixel_offset.x) - half_length, 0.0);
    float distance = length(vec2(past_end, pixel_offset.y));
    float coverage = clamp(0.5 * line_width_pixels + 0.5 - distance, 0.0, 1.0);
    // don't write depth outside the line
    if (coverage <= 0.0) {
        discard;
    }
    // depth images and segmentation have no antialiasing: a pixel is
    // either on the line or it isn't
    if (frame.output_mode.x == OUTPUT_INDEX) {
        if (coverage < 0.5) {
            discard;
        }
        fragment_color = vec4(view_depth, node_index, 0.0, 1.0);
        return;
    }
    vec3 color = segment_color.rgb;
    if (points) {
        // the normal of a sphere under this pixel, lit from the upper
        // left of the view, so a point reads as a round particle
        vec2 across = pixel_offset / max(0.5 * line_width_pixels, 1.0);
        vec3 normal = vec3(across, sqrt(max(1.0 - dot(across, across), 0.0)));
        vec3 light = normalize(vec3(-0.3, 0.4, 0.87));
        float lambert = max(dot(normal, light), 0.0);
        float highlight = pow(max(dot(normal, normalize(light + vec3(0.0, 0.0, 1.0))), 0.0), 40.0);
        color = color * (0.35 + 0.65 * lambert) + 0.4 * highlight;
    }
    // premultiplied by alpha, which is how the framebuffer blends
    float alpha = segment_color.a * coverage;
    fragment_color = vec4(color * alpha, alpha);
}
