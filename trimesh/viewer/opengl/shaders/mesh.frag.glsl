// glTF metallic-roughness shading: GGX, Schlick fresnel, Smith-Schlick
// geometry and Lambert diffuse, in `KHR_lights_punctual` units.
//
// Adapted from pyrender (MIT license, copyright 2019 Matthew Matl)
// https://github.com/mmatl/pyrender/blob/master/pyrender/shaders/mesh.frag
// with the metallic-roughness channels corrected to the glTF spec
// (green is roughness, blue is metallic) and emissive applied once.

in vec3 view_position, view_normal;
in vec2 texture_uv;
in vec4 vertex_color;
flat in float fragment_node_index;

// missing textures are 1x1 white, except normal maps `material.has_textures.x` skips
uniform sampler2D base_color_texture, metallic_roughness_texture, emissive_texture,
    normal_texture, occlusion_texture, transmission_texture, diffuse_transmission_texture,
    diffuse_transmission_color_texture, ambient_occlusion;
// RGBA per triangle by `gl_PrimitiveID` if `material.has_textures.y`
uniform samplerBuffer face_colors;
uniform sampler2DArrayShadow shadow_maps;
out vec4 fragment_color;

// environment irradiance over pi from l=2 spherical harmonics, ordered
// as `lighting.HARMONICS` with constants included in `frame.irradiance`
vec3 environment_irradiance(vec3 normal) {
    vec3 local = mat3(frame.environment_rotation) * normal;
    vec3 squared = local * local;
    vec4 light = frame.irradiance[0] + frame.irradiance[1] * local.y
        + frame.irradiance[2] * local.z + frame.irradiance[3] * local.x
        + frame.irradiance[4] * local.x * local.y + frame.irradiance[5] * local.y * local.z
        + frame.irradiance[6] * (3.0 * squared.z - 1.0) + frame.irradiance[7] * local.x * local.z
        + frame.irradiance[8] * (squared.x - squared.y);
    return light.rgb;
}

// 3x3 PCF, offset two shadow map texels along the normal against acne
float shadow_visibility(vec4 shadow, vec3 position, vec3 normal, float distance) {
    if (shadow.x < 0.0) {
        return 1.0;
    }
    mat4 light_clip = frame.shadow_matrices[int(shadow.x)];
    // texel width at this distance, which is one for directional lights
    mat3 rows = transpose(mat3(light_clip));
    float size = float(textureSize(shadow_maps, 0).x);
    float texel = 2.0 * distance / (min(length(rows[0]), length(rows[1])) * size);
    vec4 projected = light_clip * vec4(position + normal * 2.0 * texel, 1.0);
    vec3 coordinates = projected.xyz / projected.w * 0.5 + 0.5;
    float visible = 0.0;
    for (int index = 0; index < 9; index++) {
        vec2 offset = coordinates.xy + vec2(index / 3 - 1, index % 3 - 1) / size;
        visible += texture(shadow_maps, vec4(offset, shadow.x, coordinates.z));
    }
    return visible / 9.0;
}

void main() {
    // a floor showing only shadows, invisible from below
    bool shadow_catcher = material.flags.z == 1;
    if (shadow_catcher && !gl_FrontFacing) {
        discard;
    }
    // vertex and face colors are sRGB
    vec4 srgb_color = vertex_color;
    if (material.has_textures.y == 1) {
        srgb_color *= texelFetch(face_colors, gl_PrimitiveID);
    }
    vec4 base_color = material.base_color
        * texture(base_color_texture, texture_uv)
        * vec4(srgb_to_linear(srgb_color.rgb), srgb_color.a);
    // transmission as alpha, see `shaders.ALPHA_MODES`
    float alpha = base_color.a
        * (1.0 - material.transmission_color.w * texture(transmission_texture, texture_uv).r);
    if (material.flags.x == ALPHA_OPAQUE) {
        alpha = 1.0;
    }
    if (material.flags.x == ALPHA_MASK && alpha < material.factors.z) {
        discard;
    }
    if (frame.output_mode.x == OUTPUT_DEPTH) {
        return;
    }
    if (frame.output_mode.x == OUTPUT_INDEX) {
        fragment_color = vec4(-view_position.z, fragment_node_index, 0.0, 1.0);
        return;
    }
    // `KHR_materials_unlit`
    if (material.flags.y == 1) {
        fragment_color = vec4(linear_to_srgb(clamp(base_color.rgb, 0.0, 1.0)) * alpha, alpha);
        return;
    }
    vec2 roughness_metallic = texture(metallic_roughness_texture, texture_uv).gb;
    float roughness = clamp(material.factors.y * roughness_metallic.x, 0.04, 1.0);
    float metallic = clamp(material.factors.x * roughness_metallic.y, 0.0, 1.0);
    // flat normals face the camera, smooth ones are flipped on back faces
    vec3 surface_normal = normalize(cross(dFdx(view_position), dFdy(view_position)));
    if (frame.smooth_shading.x == 1) {
        surface_normal = normalize(view_normal) * (float(gl_FrontFacing) * 2.0 - 1.0);
    }
    if (material.has_textures.x == 1) {
        // tangent frame from screen derivatives: http://www.thetenthplanet.de/archives/1180
        vec3 across_y = cross(dFdy(view_position), surface_normal);
        vec3 across_x = cross(surface_normal, dFdx(view_position));
        vec2 uv_x = dFdx(texture_uv), uv_y = dFdy(texture_uv);
        vec3 tangent = across_y * uv_x.x + across_x * uv_y.x;
        vec3 bitangent = across_y * uv_x.y + across_x * uv_y.y;
        // the max keeps meshes without uv from dividing by zero
        float scale = inversesqrt(max(max(dot(tangent, tangent), dot(bitangent, bitangent)), 1e-30));
        vec3 mapped = texture(normal_texture, texture_uv).xyz * 2.0 - 1.0;
        mat3 tangent_frame = mat3(tangent * scale, bitangent * scale, surface_normal);
        surface_normal = normalize(tangent_frame * mapped);
    }
    // orthographic cameras look down -Z from infinitely far away
    bvec3 orthographic = bvec3(frame.projection[3][3] == 1.0);
    vec3 view_direction = mix(normalize(-view_position), vec3(0.0, 0.0, 1.0), orthographic);
    float normal_dot_view = max(dot(surface_normal, view_direction), 1e-4);
    vec3 reflectance = mix(vec3(0.04), base_color.rgb, metallic);
    vec3 diffuse_color = base_color.rgb * (1.0 - metallic);
    // `KHR_materials_diffuse_transmission`: light from behind a thin surface
    float transmission = clamp(material.factors.w
        * texture(diffuse_transmission_texture, texture_uv).a, 0.0, 1.0);
    vec3 transmission_color = material.transmission_color.rgb
        * texture(diffuse_transmission_color_texture, texture_uv).rgb
        * (1.0 - metallic) * transmission;
    diffuse_color *= 1.0 - transmission;
    // GGX uses alpha = roughness^2 and this needs alpha^2
    float alpha_squared = pow(roughness, 4.0);
    // remapping of roughness for Smith-Schlick from UE4
    float geometry_k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    // diffuse is scaled by alpha later and specular isn't
    vec3 diffuse_light = vec3(0.0), specular_light = vec3(0.0);
    // directional light reaching a white diffuse surface, for the shadow catcher
    float received = 0.0;
    for (int index = 0; index < frame.light_count.x; index++) {
        Light light = frame.lights[index];
        // a directional light is a unit vector away with W zero
        vec3 to_light = light.position.xyz - view_position * light.position.w;
        float distance_squared = dot(to_light, to_light);
        vec3 light_direction = to_light * inversesqrt(distance_squared);
        float range_ratio = distance_squared * light.falloff.x * light.falloff.x;
        float window = clamp(1.0 - range_ratio * range_ratio, 0.0, 1.0);
        float cone = clamp(dot(light.direction.xyz, -light_direction) * light.falloff.y
            + light.falloff.z, 0.0, 1.0);
        // shadows are looked up off the side facing the light
        float normal_dot_light = dot(surface_normal, light_direction);
        vec3 light_side = surface_normal;
        if (normal_dot_light < 0.0) {
            light_side = -surface_normal;
        }
        float visibility = shadow_visibility(
            light.shadow, view_position, light_side, sqrt(distance_squared));
        vec3 radiance = light.color.rgb * window * window * cone * cone / distance_squared
            * visibility;
        // transmitted fresnel uses the light mirrored through the surface
        float behind_dot_light = max(-normal_dot_light, 0.0);
        vec3 mirrored = light_direction + 2.0 * behind_dot_light * surface_normal;
        float mirrored_dot_half = clamp(
            dot(view_direction, normalize(mirrored + view_direction)), 0.0, 1.0);
        normal_dot_light = max(normal_dot_light, 0.0);
        vec3 half_vector = normalize(light_direction + view_direction);
        // unit vectors can dot to just over one where `pow` would be NaN
        float normal_dot_half = clamp(dot(surface_normal, half_vector), 0.0, 1.0);
        float view_dot_half = clamp(dot(view_direction, half_vector), 0.0, 1.0);
        vec3 fresnel = reflectance + (1.0 - reflectance) * pow(1.0 - view_dot_half, 5.0);
        float denominator = normal_dot_half * normal_dot_half * (alpha_squared - 1.0) + 1.0;
        float distribution = alpha_squared / (PI * denominator * denominator);
        float geometry = (normal_dot_view / (normal_dot_view * (1.0 - geometry_k) + geometry_k))
            * (normal_dot_light / (normal_dot_light * (1.0 - geometry_k) + geometry_k));
        vec3 specular = fresnel * distribution * geometry
            / (4.0 * normal_dot_view * normal_dot_light + 1e-4);
        vec3 diffuse = (1.0 - fresnel) * diffuse_color / PI;
        vec3 transmitted = (1.0 - reflectance) * (1.0 - pow(1.0 - mirrored_dot_half, 5.0))
            * transmission_color / PI;
        diffuse_light += (diffuse * normal_dot_light + transmitted * behind_dot_light)
            * radiance;
        specular_light += specular * normal_dot_light * radiance;
        // only directional lights count towards `frame.ground_light`
        received += (1.0 - light.position.w) * dot(radiance, LUMINANCE) * normal_dot_light / PI;
    }
    // environment specular scaled by Karis's BRDF fit
    // https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
    vec4 fit = roughness * vec4(-1.0, -0.0275, -0.572, 0.022) + vec4(1.0, 0.0425, 1.04, -0.04);
    float fit_bias = min(fit.x * fit.x, exp2(-9.28 * normal_dot_view)) * fit.x + fit.y;
    vec2 scale_bias = vec2(-1.04, 1.04) * fit_bias + fit.zw;
    vec3 reflected = reflect(-view_direction, surface_normal);
    vec3 environment_specular = environment_light(reflected, roughness)
        * (reflectance * scale_bias.x + scale_bias.y);
    vec3 environment_diffuse = environment_irradiance(surface_normal);
    // occlusion only darkens environment light as glTF specifies, and
    // screen space occlusion only the light on the side the camera sees
    float material_occlusion = texture(occlusion_texture, texture_uv).r;
    float occlusion = material_occlusion
        * texture(ambient_occlusion, gl_FragCoord.xy / frame.viewport_size.xy).r;
    diffuse_light += diffuse_color * environment_diffuse * occlusion
        + transmission_color * environment_irradiance(-surface_normal) * material_occlusion;
    diffuse_light += material.emissive.rgb * texture(emissive_texture, texture_uv).rgb;
    specular_light += environment_specular * occlusion;
    if (shadow_catcher) {
        // black, as opaque as the fraction of the floor's light blocked
        float reaching = received + dot(environment_diffuse, LUMINANCE) * occlusion;
        float shadow = 1.0 - reaching / max(frame.ground_light.x, 1e-6);
        fragment_color = vec4(0.0, 0.0, 0.0, clamp(shadow, 0.0, 1.0));
        return;
    }
    // blending adds `(1 - alpha) * behind`, so reflections add on top of the
    // tone mapped diffuse: opaque is `to_display(D + S)`, glass `to_display(S)`
    vec3 covered = diffuse_light * alpha;
    vec3 color = alpha * to_display(diffuse_light)
        + to_display(covered + specular_light) - to_display(covered);
    fragment_color = vec4(color, alpha);
}
