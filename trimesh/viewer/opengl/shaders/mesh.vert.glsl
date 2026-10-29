// One mesh vertex for one instance. Lighting is in view space, so the
// camera is at the origin looking down -Z.

// four rows per joint of `world(joint) * inverse_bind`
uniform samplerBuffer joint_matrices;

out vec3 view_position, view_normal;
out vec2 texture_uv;
out vec4 vertex_color;
flat out float fragment_node_index;
mat4 joint_rows(float joint) {
    int row = int(joint_offset) + 4 * int(joint);
    return mat4(
        texelFetch(joint_matrices, row), texelFetch(joint_matrices, row + 1),
        texelFetch(joint_matrices, row + 2), texelFetch(joint_matrices, row + 3));
}

void main() {
    // skinned vertices are put in world space by their joints, and their
    // node's transform is the identity
    mat4 skin = mat4(1.0);
    mat3 skin_normal = mat3(1.0);
    if (joint_offset >= 0.0) {
        skin = transpose(
            weights.x * joint_rows(joints.x) + weights.y * joint_rows(joints.y)
            + weights.z * joint_rows(joints.z) + weights.w * joint_rows(joints.w));
        // normals move by the cofactor of the blend, like `normal_matrix`
        mat3 linear = mat3(skin);
        skin_normal = mat3(
            cross(linear[1], linear[2]), cross(linear[2], linear[0]), cross(linear[0], linear[1]))
            * sign(determinant(linear));
    }
    vec4 camera_position = model_view * skin * vec4(position, 1.0);
    view_position = camera_position.xyz;
    view_normal = normal_matrix * skin_normal * normal;
    texture_uv = uv;
    vertex_color = color;
    fragment_node_index = node_index;
    gl_Position = frame.projection * camera_position;
}
