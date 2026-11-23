// A line segment as an instanced screen space rectangle with round caps,
// so width is in pixels on every driver. A point has identical endpoints.

uniform mat4 model_view;
// `frame.line_widths.y` across rather than `.x`
uniform bool points;

// pixel offset is from the segment middle, x along and y across
out vec4 segment_color;
out vec2 pixel_offset;
out float line_width_pixels, half_length, view_depth;

const vec2 CORNERS[6] = vec2[6](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
                                vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
void main() {
    // cut at the near plane, as behind the camera W goes negative
    vec4 near_plane = frame.inverse_projection * vec4(0.0, 0.0, -1.0, 1.0);
    float NEAR_Z = near_plane.z / near_plane.w;
    vec4 view_start = model_view * vec4(segment_start, 1.0);
    vec4 view_end = model_view * vec4(segment_end, 1.0);
    // entirely behind the camera: move off screen
    if (view_start.z > NEAR_Z && view_end.z > NEAR_Z) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    if (view_start.z > NEAR_Z) {
        view_start = mix(view_start, view_end, (NEAR_Z - view_start.z) / (view_end.z - view_start.z));
    }
    if (view_end.z > NEAR_Z) {
        view_end = mix(view_end, view_start, (NEAR_Z - view_end.z) / (view_start.z - view_end.z));
    }
    vec4 clip_start = frame.projection * view_start;
    vec4 clip_end = frame.projection * view_end;
    vec2 viewport_size = frame.viewport_size.xy;
    vec2 pixel_start = (clip_start.xy / clip_start.w * 0.5 + 0.5) * viewport_size;
    vec2 pixel_end = (clip_end.xy / clip_end.w * 0.5 + 0.5) * viewport_size;
    vec2 pixel_delta = pixel_end - pixel_start;
    float pixel_length = length(pixel_delta);
    vec2 along = vec2(1.0, 0.0);
    if (pixel_length > 1e-3) {
        along = pixel_delta / pixel_length;
    }
    vec2 across = vec2(-along.y, along.x);
    // half the line width plus a pixel for antialiasing
    float line_width = mix(frame.line_widths.x, frame.line_widths.y, points);
    float radius = 0.5 * line_width + 1.0;
    vec2 corner = CORNERS[gl_VertexID];
    half_length = 0.5 * pixel_length;
    pixel_offset = corner * vec2(half_length + radius, radius);
    segment_color = color;
    line_width_pixels = line_width;
    vec2 pixel = 0.5 * (pixel_start + pixel_end) + along * pixel_offset.x + across * pixel_offset.y;
    // depth of the nearest endpoint so lines depth test against meshes
    vec4 clip_corner = mix(clip_end, clip_start, bvec4(corner.x < 0.0));
    view_depth = -mix(view_end.z, view_start.z, corner.x < 0.0);
    gl_Position = vec4((pixel / viewport_size * 2.0 - 1.0) * clip_corner.w, clip_corner.z, clip_corner.w);
}
