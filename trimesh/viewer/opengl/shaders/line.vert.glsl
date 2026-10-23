#version 330 core
// Draw one line segment as a rectangle on screen with round caps, so
// lines are the same width in pixels everywhere and on every driver.
// A point is a segment with two identical endpoints, which is a dot.
//
// Every segment is one instance and every instance runs this shader
// six times (two triangles), with each run moving to one corner:
// `glDrawArraysInstanced(GL_TRIANGLES, 0, 6, segment_count)`

// both endpoints and the color advance once per instance, from `shaders.SEGMENT`
$segment_inputs

// per-frame values from `shaders.FRAME`
$frame_block

uniform mat4 model_view;
// line width or point diameter in pixels
uniform float line_width;

out vec4 segment_color;
// pixel offset from the segment middle: x along, y across
out vec2 pixel_offset;
// half the length of the segment on screen in pixels
out float half_length;
// distance along the view axis for depth images
out float view_depth;

// the corners of two triangles making a rectangle
const vec2 CORNERS[6] = vec2[6](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)
);
// view space Z of the plane segments are cut at: anything larger is
// behind the camera, which for perspective is where W goes negative
const float NEAR_Z = -1e-5;

void main() {
    vec4 view_start = model_view * vec4(segment_start, 1.0);
    vec4 view_end = model_view * vec4(segment_end, 1.0);

    // a segment entirely behind the camera is moved off screen
    if (view_start.z > NEAR_Z && view_end.z > NEAR_Z) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    // cut segments crossing the camera plane before projecting,
    // otherwise the end behind the camera flips across the screen
    if (view_start.z > NEAR_Z) {
        view_start = mix(view_start, view_end, (NEAR_Z - view_start.z) / (view_end.z - view_start.z));
    }
    if (view_end.z > NEAR_Z) {
        view_end = mix(view_end, view_start, (NEAR_Z - view_end.z) / (view_start.z - view_end.z));
    }
    vec4 clip_start = frame.projection * view_start;
    vec4 clip_end = frame.projection * view_end;

    // both endpoints in pixels
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

    // half the rectangle size: half the line width plus a pixel for antialiasing
    float radius = 0.5 * line_width + 1.0;
    vec2 corner = CORNERS[gl_VertexID];
    half_length = 0.5 * pixel_length;
    pixel_offset = corner * vec2(half_length + radius, radius);
    segment_color = color;

    vec2 pixel = 0.5 * (pixel_start + pixel_end) + along * pixel_offset.x + across * pixel_offset.y;
    // take depth from the nearest endpoint so lines depth test against meshes
    vec4 clip_corner = clip_end;
    view_depth = -view_end.z;
    if (corner.x < 0.0) {
        clip_corner = clip_start;
        view_depth = -view_start.z;
    }
    gl_Position = vec4((pixel / viewport_size * 2.0 - 1.0) * clip_corner.w, clip_corner.z, clip_corner.w);
}
