#version 330 core
// glTF metallic-roughness shading of `Scene.lights`: Cook-Torrance
// specular with a GGX distribution, Schlick fresnel and Smith-Schlick
// geometry, plus Lambert diffuse, in `KHR_lights_punctual` units. An
// environment of a sky to ground gradient and softboxes surrounds
// everything, then a neutral tone mapper maps it to the display.
//
// Adapted from pyrender (MIT license, copyright 2019 Matthew Matl)
// https://github.com/mmatl/pyrender/blob/master/pyrender/shaders/mesh.frag
// with the metallic-roughness channels corrected to the glTF spec
// (green is roughness, blue is metallic) and emissive applied once.

in vec3 view_position;
in vec3 view_normal;
in vec2 texture_uv;
in vec4 vertex_color;
flat in float fragment_node_index;

// per-frame values from `shaders.FRAME`, including the lights
$frame_block

// this mesh's material from `shaders.MATERIAL`
$material_block

// how alpha is used, from `shaders.ALPHA_MODES` which explains them
$alpha_modes
// missing textures are a 1x1 white texture so they change nothing,
// except a normal map which `material.has_textures.x` skips entirely
uniform sampler2D base_color_texture;
uniform sampler2D metallic_roughness_texture;
uniform sampler2D emissive_texture;
uniform sampler2D normal_texture;
uniform sampler2D occlusion_texture;
// alpha multiplies the diffuse transmission factor and color multiplies
// `material.transmission_color`
uniform sampler2D diffuse_transmission_texture;
uniform sampler2D diffuse_transmission_color_texture;
// how much of the environment nearby surfaces leave visible, per pixel
uniform sampler2D ambient_occlusion;
// one RGBA color per triangle looked up with `gl_PrimitiveID`, if
// `material.has_textures.y` is one
uniform samplerBuffer face_colors;
// the depth each shadow casting light sees, one layer per light
uniform sampler2DArrayShadow shadow_maps;

out vec4 fragment_color;

// Rec.709 weights of linear RGB for how bright a color looks
const vec3 LUMINANCE = vec3(0.2126, 0.7152, 0.0722);
// the sharpness of a spherical Gaussian with the mean cosine of the
// cosine lobe diffuse surfaces reflect with, from `scene.metadata`
const float DIFFUSE_SHARPNESS = $DIFFUSE_SHARPNESS;
// a cubic in log roughness for the sharpness of the spherical Gaussian
// as wide as the GGX reflection lobe, from `scene.metadata`
const vec4 SPECULAR_SHARPNESS = $SPECULAR_SHARPNESS;

$shared_functions

// how much of a light reaches a point: the average of nine nearby
// comparisons against what the light saw, each blended by the hardware
float shadow_visibility(vec4 shadow, vec3 position, vec3 normal, float distance) {
    if (shadow.x < 0.0) {
        return 1.0;
    }
    // look up from a couple of shadow map pixels off the surface so a
    // lit surface never shadows itself
    float pixel_width = shadow.y + shadow.z * distance;
    vec3 lifted = position + normal * 2.0 * pixel_width;
    vec4 projected = frame.shadow_matrices[int(shadow.x)] * vec4(lifted, 1.0);
    vec3 coordinate = projected.xyz / projected.w;
    vec2 texel = 1.0 / vec2(textureSize(shadow_maps, 0).xy);
    float visible = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 offset = coordinate.xy + vec2(x, y) * texel;
            visible += texture(shadow_maps, vec4(offset, shadow.x, coordinate.z));
        }
    }
    return visible / 9.0;
}

// tilt a normal by a tangent space normal map, with the tangent and
// bitangent from how position and uv change between neighboring
// pixels so meshes need no tangents: http://www.thetenthplanet.de/archives/1180
vec3 perturb_normal(vec3 normal) {
    vec3 across_y = cross(dFdy(view_position), normal);
    vec3 across_x = cross(normal, dFdx(view_position));
    vec2 uv_x = dFdx(texture_uv);
    vec2 uv_y = dFdy(texture_uv);
    vec3 tangent = across_y * uv_x.x + across_x * uv_y.x;
    vec3 bitangent = across_y * uv_x.y + across_x * uv_y.y;
    // the max keeps meshes without uv from dividing by zero
    float scale = inversesqrt(max(max(dot(tangent, tangent), dot(bitangent, bitangent)), 1e-30));
    vec3 mapped = texture(normal_texture, texture_uv).xyz * 2.0 - 1.0;
    return normalize(mat3(tangent * scale, bitangent * scale, normal) * mapped);
}

void main() {
    int alpha_mode = material.flags.x;
    bool unlit = material.flags.y == 1;
    // an invisible floor which is only as dark as the shadows on it
    bool shadow_catcher = material.flags.z == 1;
    // the floor can't be seen from below
    if (shadow_catcher && !gl_FrontFacing) {
        discard;
    }
    // vertex and face colors are sRGB so convert them to linear
    vec4 face_color = vec4(1.0);
    if (material.has_textures.y == 1) {
        face_color = texelFetch(face_colors, gl_PrimitiveID);
    }
    vec4 srgb_color = vertex_color * face_color;
    vec4 base_color = material.base_color
        * texture(base_color_texture, texture_uv)
        * vec4(srgb_to_linear(srgb_color.rgb), srgb_color.a);

    float alpha = base_color.a;
    if (alpha_mode == ALPHA_OPAQUE) {
        alpha = 1.0;
    }
    if (alpha_mode == ALPHA_MASK && alpha < material.factors.z) {
        discard;
    }
    // depth along the view axis and which node this is, for depth
    // images and segmentation rather than a picture
    if (frame.write_depth_and_index.x == 1) {
        fragment_color = vec4(-view_position.z, fragment_node_index, 0.0, 1.0);
        return;
    }
    // which no light, shadow, occlusion, exposure or tone mapping changes
    if (unlit) {
        fragment_color = vec4(linear_to_srgb(clamp(base_color.rgb, 0.0, 1.0)) * alpha, alpha);
        return;
    }

    vec2 roughness_metallic = texture(metallic_roughness_texture, texture_uv).gb;
    float roughness = clamp(material.factors.y * roughness_metallic.x, 0.04, 1.0);
    float metallic = clamp(material.factors.x * roughness_metallic.y, 0.0, 1.0);

    vec3 surface_normal;
    // or every triangle flat with its exact face normal
    if (frame.smooth_shading.x == 1) {
        surface_normal = normalize(view_normal);
        // light the back of a triangle as if it were the front
        if (!gl_FrontFacing) {
            surface_normal = -surface_normal;
        }
    } else {
        // the exact triangle normal from how position changes across
        // neighboring pixels, which always faces the camera
        surface_normal = normalize(cross(dFdx(view_position), dFdy(view_position)));
    }
    if (material.has_textures.x == 1) {
        surface_normal = perturb_normal(surface_normal);
    }
    // the camera is at the view space origin, or for an orthographic
    // projection infinitely far away looking down -Z
    vec3 view_direction = normalize(-view_position);
    if (frame.projection[3][3] == 1.0) {
        view_direction = vec3(0.0, 0.0, 1.0);
    }
    float normal_dot_view = max(dot(surface_normal, view_direction), 1e-4);

    // dielectrics reflect 4% head on and metals reflect their color
    vec3 reflectance = mix(vec3(0.04), base_color.rgb, metallic);
    vec3 diffuse_color = base_color.rgb * (1.0 - metallic);
    // light from behind the surface, relative to the side the camera
    // sees, which a thin surface scatters out of the front instead of
    // reflecting: the glTF spec mixes the reflected and transmitted
    // Lambert lobes by this fraction, and only for the dielectric part
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

    // light scattered by the surface, which is as transparent as the
    // surface, and light reflected off it, which isn't: glass is clear
    // but still shows reflections
    vec3 diffuse_light = vec3(0.0);
    vec3 specular_light = vec3(0.0);
    // how much directional light reaches a white diffuse surface here
    float received = 0.0;
    for (int index = 0; index < frame.light_count.x; index++) {
        Light light = frame.lights[index];
        // a directional light is a unit vector away with W zero
        vec3 to_light = light.position.xyz - view_position * light.position.w;
        float distance_squared = dot(to_light, to_light);
        vec3 light_direction = to_light * inversesqrt(distance_squared);
        // the `KHR_lights_punctual` range window, one without a range
        float range_ratio = distance_squared * light.falloff.x * light.falloff.x;
        float window = clamp(1.0 - range_ratio * range_ratio, 0.0, 1.0);
        // the `KHR_lights_punctual` spot cone, one without a cone
        float cone = clamp(
            dot(light.direction.xyz, -light_direction) * light.falloff.y + light.falloff.z,
            0.0, 1.0);
        // a light reaches the front or the back of the surface, and the
        // shadow is looked up off whichever side faces it so light behind
        // a leaf is blocked by what is between it and the leaf but not by
        // the leaf itself
        float normal_dot_light = dot(surface_normal, light_direction);
        vec3 light_side = surface_normal;
        if (normal_dot_light < 0.0) {
            light_side = -surface_normal;
        }
        float visibility = shadow_visibility(
            light.shadow, view_position, light_side, sqrt(distance_squared));
        vec3 radiance = light.color.rgb * window * window * cone * cone / distance_squared
            * visibility;

        // transmitted light is scattered as if from the light mirrored
        // through the surface, which is where its fresnel term comes from
        float behind_dot_light = max(-normal_dot_light, 0.0);
        vec3 mirrored = light_direction + 2.0 * behind_dot_light * surface_normal;
        float mirrored_dot_half = clamp(
            dot(view_direction, normalize(mirrored + view_direction)), 0.0, 1.0);
        normal_dot_light = max(normal_dot_light, 0.0);
        vec3 half_vector = normalize(light_direction + view_direction);
        // unit vectors can dot to just over one, where the `pow` in the
        // fresnel term of one minus this would be NaN, so clamp both ends
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
        // directional lights are the ones `Studio.ground_light` counts
        received += (1.0 - light.position.w) * dot(radiance, LUMINANCE) * normal_dot_light / PI;
    }

    // the environment: diffuse surfaces see it averaged over the cosine
    // lobe around the normal, and shiny ones see the reflection blurred
    // by the GGX lobe, scaled by Karis's fit of the environment BRDF
    // https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
    vec4 fit = roughness * vec4(-1.0, -0.0275, -0.572, 0.022) + vec4(1.0, 0.0425, 1.04, -0.04);
    float fit_bias = min(fit.x * fit.x, exp2(-9.28 * normal_dot_view)) * fit.x + fit.y;
    vec2 scale_bias = vec2(-1.04, 1.04) * fit_bias + fit.zw;
    vec3 reflected = reflect(-view_direction, surface_normal);
    float log_roughness = log(roughness);
    float reflection_blur = exp(((SPECULAR_SHARPNESS.x * log_roughness + SPECULAR_SHARPNESS.y)
        * log_roughness + SPECULAR_SHARPNESS.z) * log_roughness + SPECULAR_SHARPNESS.w);
    vec3 environment_specular = environment_light(reflected, reflection_blur)
        * (reflectance * scale_bias.x + scale_bias.y);
    vec3 environment_diffuse = environment_light(surface_normal, DIFFUSE_SHARPNESS);
    // occlusion only darkens environment light as glTF specifies
    float occlusion = texture(occlusion_texture, texture_uv).r
        * texture(ambient_occlusion, gl_FragCoord.xy / frame.viewport_size.xy).r;
    // and thin surfaces let through the environment behind them
    diffuse_light += (diffuse_color * environment_diffuse
        + transmission_color * environment_light(-surface_normal, DIFFUSE_SHARPNESS))
        * occlusion;
    diffuse_light += material.emissive.rgb * texture(emissive_texture, texture_uv).rgb;
    specular_light += environment_specular * occlusion;

    if (shadow_catcher) {
        // black, and as opaque as the fraction of the floor's light which
        // shadows and nearby surfaces keep from reaching it
        float reaching = received + dot(environment_diffuse, LUMINANCE) * occlusion;
        float shadow = 1.0 - reaching / max(frame.ground_light.x, 1e-6);
        fragment_color = vec4(0.0, 0.0, 0.0, clamp(shadow, 0.0, 1.0));
        return;
    }

    // the framebuffer blends what the display shows as `color + (1 - alpha)
    // * behind`, so the surface's own light `D` covers `alpha` of the display
    // with `to_display(D)`, and its reflections `S` add what they add over
    // the light the surface lets through, `to_display(alpha * D + S) -
    // to_display(alpha * D)`. Opaque that is `to_display(D + S)`, with nothing
    // reflected `alpha * to_display(D)`, and clear glass `to_display(S)`, and
    // never negative as `to_display` only increases
    vec3 covered = diffuse_light * alpha;
    vec3 color = alpha * to_display(diffuse_light)
        + to_display(covered + specular_light) - to_display(covered);
    fragment_color = vec4(color, alpha);
}
