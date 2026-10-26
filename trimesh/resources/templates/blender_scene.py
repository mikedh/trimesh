"""
blender_scene.py
------------------

Build a Blender scene from a GLB exported by trimesh, then render it to
numbered PNG frames or save it as a `.blend`. Run by
`trimesh.interfaces.blender` in Blender 4.3 or later as:
`blender --background --python blender_scene.py -- settings.json`

The GLB carries the geometry, materials, lights, cameras and animation
which Blender's glTF importer reads, with every frame keyed. The settings
JSON adds what a GLB can't say about how trimesh draws a scene and where
Blender reads glTF differently:
- the environment, shadow floor and ambient occlusion of the viewer
- lines and points as many pixels across as the viewer draws them
- `KHR_materials_diffuse_transmission`, which the importer ignores
- glTF spot cones, and lights reaching as far as glTF says
- the camera's exact fields of view
"""

import json
import math
import sys

import bpy

# the light and halfway vector `line.frag.glsl` shades points with, in
# view space looking down -Z
POINT_LIGHT = (-0.3, 0.4, 0.87)


def unit(vector):
    """A vector scaled to length one, as the viewer reads directions."""
    length = math.sqrt(sum(value * value for value in vector))
    return [value / length for value in vector]


def to_blender(vector):
    """glTF's +Y up to Blender's +Z up, the same swap the importer does."""
    return (vector[0], -vector[2], vector[1])


class Graph:
    """
    Adds nodes to a node tree one call at a time, linking sockets and
    setting constants so the math reads like the GLSL it copies.
    """

    def __init__(self, tree):
        self.tree = tree

    def node(self, kind, inputs=None, **properties):
        """One node of `kind` with `properties` set and then `inputs`."""
        node = self.tree.nodes.new(kind)
        # properties like `operation` change which inputs exist
        for name, value in properties.items():
            setattr(node, name, value)
        for key, value in (inputs or {}).items():
            self.set(node.inputs[key], value)
        return node

    def set(self, socket, value):
        """Link another socket into `socket` or set it to a constant."""
        if isinstance(value, bpy.types.NodeSocket):
            self.tree.links.new(value, socket)
        else:
            socket.default_value = value

    def math(self, operation, *values):
        """A scalar math node's output."""
        return self.node(
            "ShaderNodeMath", dict(enumerate(values)), operation=operation
        ).outputs[0]

    def vector(self, operation, *values):
        """A vector math node's vector output."""
        return self.node(
            "ShaderNodeVectorMath", dict(enumerate(values)), operation=operation
        ).outputs["Vector"]

    def dot(self, a, b):
        """The dot product of two vectors."""
        return self.node(
            "ShaderNodeVectorMath", {0: a, 1: b}, operation="DOT_PRODUCT"
        ).outputs["Value"]

    def scale(self, vector, factor):
        """A vector or color times a number."""
        return self.node(
            "ShaderNodeVectorMath", {0: vector, "Scale": factor}, operation="SCALE"
        ).outputs["Vector"]


def match_lights(settings):
    """
    Make Blender's lights glTF's. glTF spot lights square a ramp from the
    outer to the inner cone's cosine, and Blender's are a smoothstep of a
    ramp from the outer cone's cosine `(1 - cos(outer)) * spot_blend` long,
    which is closest to the glTF curve when `spot_blend_scale` times as
    steep. The importer sets the blend to `1 - inner / outer`, which is
    undone to get the inner cone back. glTF lights are points with no
    size, which reach as far as their `range` or everything without one,
    where Blender would cut a dim light off.
    """
    for light in bpy.data.lights:
        light.use_custom_distance = True
        light.cutoff_distance = settings["ranges"].get(light.name, settings["cutoff"])
        if light.type == "SPOT":
            outer = light.spot_size / 2.0
            inner = outer * (1.0 - light.spot_blend)
            ramp = (math.cos(inner) - math.cos(outer)) / (1.0 - math.cos(outer))
            light.spot_blend = min(ramp / settings["spot_blend_scale"], 1.0)
        light.shadow_soft_size = 0.0


def look_through(scene, settings):
    """
    Look through the camera trimesh renders with, named after its node,
    with its exact fields of view: a horizontal one which doesn't match
    the image's shape is pixels which aren't square.
    """
    scene.camera = bpy.data.objects[settings["camera"]]
    scene.render.resolution_x = settings["width"]
    scene.render.resolution_y = settings["height"]
    scene.render.resolution_percentage = 100
    if scene.camera.data.type == "PERSP":
        fov_x, fov_y = (math.radians(value) for value in settings["fov"])
        scene.camera.data.sensor_fit = "VERTICAL"
        scene.camera.data.angle_y = fov_y
        stretch = (math.tan(fov_x / 2.0) / math.tan(fov_y / 2.0)) / (
            settings["width"] / settings["height"]
        )
        scene.render.pixel_aspect_x = max(stretch, 1.0)
        scene.render.pixel_aspect_y = max(1.0 / stretch, 1.0)
    # glTF's planes and orthographic width are in the units of the camera's
    # node, which a scale changes, where Blender's are in world units
    scale = sum(scene.camera.matrix_world.to_scale()) / 3.0
    scene.camera.data.clip_start *= scale
    scene.camera.data.clip_end *= scale
    scene.camera.data.ortho_scale *= scale
    # the importer puts time zero on frame zero
    scene.frame_start = 0
    scene.frame_end = settings["count"] - 1


def use_eevee(scene, settings):
    """EEVEE with the viewer's ambient occlusion and tone mapping."""
    # EEVEE is `BLENDER_EEVEE_NEXT` until Blender 5.0
    engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    if "BLENDER_EEVEE_NEXT" in engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    else:
        scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = settings["samples"]
    scene.eevee.use_fast_gi = True
    scene.eevee.fast_gi_method = "AMBIENT_OCCLUSION_ONLY"
    scene.eevee.fast_gi_distance = settings["occlusion_radius"]
    # the background is the environment or composited by trimesh
    scene.render.film_transparent = not settings["show_environment"]
    scene.view_settings.view_transform = "Khronos PBR Neutral"
    # which multiplies light before tone mapping in stops, as the viewer does,
    # though Blender also exposes and tone maps the `KHR_materials_unlit`
    # materials its importer makes, which the viewer leaves their exact color
    scene.view_settings.exposure = math.log2(settings["exposure"])
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"


def load_image(path, colorspace):
    """A PNG trimesh wrote as a Blender image packed into the file."""
    image = bpy.data.images.load(path)
    image.colorspace_settings.name = colorspace
    image.pack()
    return image


def diffuse_transmission(material, extension):
    """
    `KHR_materials_diffuse_transmission`: the fraction transmitted is
    taken from the base color and added back as a translucent BSDF lit
    from behind, which leaves reflections unchanged as the glTF spec does,
    and the importer's alpha is moved after the sum so cutouts stay cut.
    """
    graph = Graph(material.node_tree)
    nodes = material.node_tree.nodes
    principled = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    output = next(node for node in nodes if node.type == "OUTPUT_MATERIAL")

    fraction = extension["diffuseTransmissionFactor"]
    if extension["diffuseTransmissionTexture"] is not None:
        image = load_image(extension["diffuseTransmissionTexture"], "Non-Color")
        alpha = graph.node("ShaderNodeTexImage", image=image).outputs["Alpha"]
        fraction = graph.math("MULTIPLY", alpha, fraction)
    color = extension["diffuseTransmissionColorFactor"]
    if color is None:
        color = [1.0, 1.0, 1.0]
    if extension["diffuseTransmissionColorTexture"] is not None:
        image = load_image(extension["diffuseTransmissionColorTexture"], "sRGB")
        texture = graph.node("ShaderNodeTexImage", image=image).outputs["Color"]
        color = graph.vector("MULTIPLY", texture, color)

    base = principled.inputs["Base Color"]
    base_color = list(base.default_value)[:3]
    for link in base.links:
        base_color = link.from_socket
    graph.set(base, graph.scale(base_color, graph.math("SUBTRACT", 1.0, fraction)))
    # EEVEE's translucency through a surface of zero thickness is the
    # square of its color, so it gets the square root of the color
    root = graph.node(
        "ShaderNodeGamma", {"Color": graph.scale(color, fraction), "Gamma": 0.5}
    )
    translucent = graph.node("ShaderNodeBsdfTranslucent", {"Color": root.outputs[0]})
    for link in principled.inputs["Normal"].links:
        graph.set(translucent.inputs["Normal"], link.from_socket)
    surface = graph.node(
        "ShaderNodeAddShader",
        {0: principled.outputs["BSDF"], 1: translucent.outputs["BSDF"]},
    ).outputs[0]

    alpha = principled.inputs["Alpha"]
    if alpha.is_linked or alpha.default_value < 1.0:
        coverage = alpha.default_value
        for link in alpha.links:
            coverage = link.from_socket
            material.node_tree.links.remove(link)
        alpha.default_value = 1.0
        transparent = graph.node("ShaderNodeBsdfTransparent").outputs[0]
        surface = graph.node(
            "ShaderNodeMixShader", {0: coverage, 1: transparent, 2: surface}
        ).outputs[0]
    graph.set(output.inputs["Surface"], surface)
    # a thin slab lets light from behind through rather than the surface
    # shadowing itself
    material.thickness_mode = "SLAB"


def emission_material(name):
    """An empty material and its graph, for lines and points."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.node_tree.nodes.clear()
    return material, Graph(material.node_tree)


def emit(graph, color, alpha):
    """
    Show `color` as the viewer does, which writes display values that
    are linear light in Blender close to a 2.2 gamma, covering `alpha`.
    """
    linear = graph.node("ShaderNodeGamma", {"Color": color, "Gamma": 2.2})
    emission = graph.node("ShaderNodeEmission", {"Color": linear.outputs[0]})
    transparent = graph.node("ShaderNodeBsdfTransparent")
    surface = graph.node(
        "ShaderNodeMixShader",
        {0: alpha, 1: transparent.outputs[0], 2: emission.outputs[0]},
    )
    graph.node("ShaderNodeOutputMaterial", {"Surface": surface.outputs[0]})


def line_material():
    """The vertex colors of lines with no lighting, as the viewer draws them."""
    material, graph = emission_material("lines")
    color = graph.node("ShaderNodeAttribute", attribute_name="Color")
    emit(graph, color.outputs["Color"], color.outputs["Alpha"])
    return material


def point_material():
    """
    Each point's color on a disc shaded as a sphere facing the screen,
    which is `line.frag.glsl` for points.
    """
    material, graph = emission_material("points")
    color = graph.node(
        "ShaderNodeAttribute", attribute_name="Color", attribute_type="INSTANCER"
    )
    # where on its disc a pixel is, with the edge at one
    across = graph.node("ShaderNodeAttribute", attribute_name="across")
    flat = graph.vector("MULTIPLY", across.outputs["Vector"], (1.0, 1.0, 0.0))
    squared = graph.dot(flat, flat)
    height = graph.math(
        "SQRT", graph.math("MAXIMUM", graph.math("SUBTRACT", 1.0, squared), 0.0)
    )
    normal = graph.vector("ADD", flat, graph.scale((0.0, 0.0, 1.0), height))
    light = unit(POINT_LIGHT)
    halfway = unit([light[0], light[1], light[2] + 1.0])
    lambert = graph.math("MAXIMUM", graph.dot(normal, light), 0.0)
    highlight = graph.math(
        "POWER", graph.math("MAXIMUM", graph.dot(normal, halfway), 0.0), 40.0
    )
    shaded = graph.vector(
        "ADD",
        graph.scale(
            color.outputs["Color"], graph.math("MULTIPLY_ADD", lambert, 0.65, 0.35)
        ),
        graph.scale((0.4, 0.4, 0.4), highlight),
    )
    inside = graph.math("LESS_THAN", squared, 1.0)
    emit(graph, shaded, graph.math("MULTIPLY", color.outputs["Alpha"], inside))
    return material


def segment_group(name, camera, pixels, pixel):
    """
    A geometry node group, its graph, its input geometry, the camera's
    pose relative to the object, and a field of the world radius which
    is `pixels` across at each point's depth in front of the camera, in
    the object's own units.
    """
    group = bpy.data.node_groups.new(name, "GeometryNodeTree")
    for direction in ("INPUT", "OUTPUT"):
        group.interface.new_socket(
            "Geometry", in_out=direction, socket_type="NodeSocketGeometry"
        )
    graph = Graph(group)
    relative = graph.node(
        "GeometryNodeObjectInfo", {"Object": camera}, transform_space="RELATIVE"
    )
    inverse = graph.node(
        "FunctionNodeInvertMatrix", {"Matrix": relative.outputs["Transform"]}
    )
    in_camera = graph.node(
        "FunctionNodeTransformPoint",
        {
            "Vector": graph.node("GeometryNodeInputPosition").outputs[0],
            "Transform": inverse.outputs["Matrix"],
        },
    )
    # the camera looks down its -Z
    depth = graph.node("ShaderNodeSeparateXYZ", {0: in_camera.outputs[0]}).outputs["Z"]
    own = graph.node(
        "GeometryNodeObjectInfo",
        {"Object": graph.node("GeometryNodeSelfObject").outputs[0]},
    )
    own_scale = graph.node("ShaderNodeSeparateXYZ", {0: own.outputs["Scale"]})
    radius = graph.math(
        "DIVIDE",
        graph.math("MULTIPLY", depth, -0.5 * pixels * pixel),
        own_scale.outputs["X"],
    )
    geometry = graph.node("NodeGroupInput").outputs[0]
    return group, graph, geometry, relative, radius


def finish_group(graph, geometry, material):
    """Give the group's geometry a material and make it the output."""
    surface = graph.node(
        "GeometryNodeSetMaterial", {"Geometry": geometry, "Material": material}
    )
    graph.set(graph.node("NodeGroupOutput").inputs[0], surface.outputs[0])


def points_group(camera, pixels, pixel):
    """
    A square facing the screen at every vertex, which is how the viewer
    draws a point, rather than EEVEE's point clouds which are diamonds,
    left as instances which EEVEE draws quickly.
    """
    group, graph, geometry, relative, radius = segment_group(
        "points", camera, pixels, pixel
    )
    square = graph.node(
        "GeometryNodeMeshGrid",
        {"Size X": 2.0, "Size Y": 2.0, "Vertices X": 2, "Vertices Y": 2},
    )
    # where on the square each corner is, for the material
    corners = graph.node(
        "GeometryNodeStoreNamedAttribute",
        {
            "Geometry": square.outputs["Mesh"],
            "Name": "across",
            "Value": graph.node("GeometryNodeInputPosition").outputs[0],
        },
        data_type="FLOAT_VECTOR",
        domain="POINT",
    )
    squares = graph.node(
        "GeometryNodeInstanceOnPoints",
        {
            "Points": geometry,
            "Instance": corners.outputs[0],
            # turned with the camera so they lie in the image plane
            "Rotation": relative.outputs["Rotation"],
            "Scale": radius,
        },
    )
    finish_group(graph, squares.outputs["Instances"], point_material())
    return group


def lines_group(camera, pixels, pixel):
    """A round tube along every edge, which is how the viewer draws a line."""
    group, graph, geometry, _, radius = segment_group("lines", camera, pixels, pixel)
    curves = graph.node("GeometryNodeMeshToCurve", {"Mesh": geometry})
    # the profile is scaled by the curve's radius before Blender 4.5
    curves = graph.node(
        "GeometryNodeSetCurveRadius", {"Curve": curves.outputs[0], "Radius": radius}
    )
    profile = graph.node("GeometryNodeCurvePrimitiveCircle", {"Resolution": 8})
    tubes = graph.node(
        "GeometryNodeCurveToMesh",
        {
            "Curve": curves.outputs[0],
            "Profile Curve": profile.outputs["Curve"],
            "Fill Caps": True,
        },
    )
    # and by `Scale` from Blender 4.5, which ignores the radius
    if "Scale" in tubes.inputs:
        graph.set(tubes.inputs["Scale"], radius)
    finish_group(graph, tubes.outputs["Mesh"], line_material())
    return group


def draw_segments(scene, settings):
    """
    Lines and points, which Blender imports as edges and vertices it
    doesn't draw, drawn as many pixels across as the viewer draws them.
    """
    bare = [
        item
        for item in scene.objects
        if item.type == "MESH" and len(item.data.polygons) == 0
    ]
    for objects, group, pixels in (
        (
            [item for item in bare if len(item.data.edges) == 0],
            points_group,
            settings["point_size"],
        ),
        (
            [item for item in bare if len(item.data.edges) > 0],
            lines_group,
            settings["line_width"],
        ),
    ):
        if len(objects) == 0:
            continue
        nodes = group(scene.camera, pixels, settings["pixel"])
        for item in objects:
            mesh = item.data
            if "Color" not in mesh.color_attributes:
                # the viewer draws lines and points without colors in black
                black = mesh.color_attributes.new("Color", "FLOAT_COLOR", "POINT")
                black.data.foreach_set("color", [0.0, 0.0, 0.0, 1.0] * len(mesh.vertices))
            # the viewer's lines and points don't cast shadows
            item.visible_shadow = False
            item.modifiers.new(nodes.name, "NODES").node_group = nodes


def environment_world(environment):
    """
    The environment: a gradient linear in height plus spherical Gaussian
    softboxes, the same function the viewer evaluates.
    """
    world = bpy.data.worlds.new("environment")
    world.use_nodes = True
    # zero turns off moving the brightest part of the environment into a
    # sun, which the viewer doesn't do
    world.sun_threshold = 0.0
    graph = Graph(world.node_tree)
    direction = graph.vector(
        "NORMALIZE", graph.node("ShaderNodeTexCoord").outputs["Generated"]
    )
    height = graph.dot(direction, to_blender(unit(environment["up"])))
    sky, ground = environment["sky"], environment["ground"]
    light = graph.vector(
        "MULTIPLY_ADD",
        [a - b for a, b in zip(sky, ground)],
        graph.math("MULTIPLY_ADD", height, 0.5, 0.5),
        ground,
    )
    for towards, sharpness, color in zip(
        environment["directions"], environment["sharpness"], environment["colors"]
    ):
        cosine = graph.dot(direction, to_blender(towards))
        falloff = graph.math(
            "EXPONENT", graph.math("MULTIPLY_ADD", cosine, sharpness, -sharpness)
        )
        light = graph.vector("MULTIPLY_ADD", color, falloff, light)
    graph.set(world.node_tree.nodes["Background"].inputs["Color"], light)
    return world


def shadow_floor(scene, corners, ground_light):
    """
    A floor which is black and as opaque as the light it doesn't get,
    relative to `ground_light` which it gets with nothing in the way.
    Elsewhere it is a holdout rather than transparent, as EEVEE ignores
    transparency mixed with a shader turned into a color, and blending
    instead renders differently every time, so unlike the viewer it hides
    the parts of lines and points which reach below it.
    """
    mesh = bpy.data.meshes.new("ground")
    mesh.from_pydata([to_blender(point) for point in corners], [], [(0, 1, 2, 3)])
    floor = bpy.data.objects.new("ground", mesh)
    scene.collection.objects.link(floor)
    floor.visible_shadow = False
    floor.visible_diffuse = False
    floor.visible_glossy = False

    material = bpy.data.materials.new("ground")
    material.use_nodes = True
    material.use_backface_culling = True
    material.surface_render_method = "DITHERED"
    material.node_tree.nodes.clear()
    graph = Graph(material.node_tree)
    diffuse = graph.node("ShaderNodeBsdfDiffuse", {"Color": (1.0, 1.0, 1.0, 1.0)})
    received = graph.node("ShaderNodeShaderToRGB", {"Shader": diffuse.outputs[0]})
    brightness = graph.node("ShaderNodeRGBToBW", {"Color": received.outputs["Color"]})
    shadow = graph.math(
        "SUBTRACT", 1.0, graph.math("DIVIDE", brightness.outputs[0], ground_light)
    )
    shadow.node.use_clamp = True
    holdout = graph.node("ShaderNodeHoldout")
    black = graph.node("ShaderNodeEmission", {"Color": (0.0, 0.0, 0.0, 1.0)})
    surface = graph.node(
        "ShaderNodeMixShader",
        {0: shadow, 1: holdout.outputs[0], 2: black.outputs[0]},
    )
    graph.node("ShaderNodeOutputMaterial", {"Surface": surface.outputs[0]})
    mesh.materials.append(material)


def main(settings):
    """Build the scene from `settings` then render or save it."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    # set before importing: the importer converts keyframe seconds to frames
    scene.render.fps = settings["fps"]
    scene.render.fps_base = settings["fps_base"]
    # "COMPAT" reads light intensity in the glTF units the viewer uses,
    # rather than dividing by 683 lumens per watt
    bpy.ops.import_scene.gltf(
        filepath=settings["input"], export_import_convert_lighting_mode="COMPAT"
    )
    match_lights(settings)
    look_through(scene, settings)
    use_eevee(scene, settings)
    for name, extension in settings["translucent"].items():
        diffuse_transmission(bpy.data.materials[name], extension)
    draw_segments(scene, settings)
    scene.world = environment_world(settings["environment"])
    if settings["ground"] is not None:
        shadow_floor(scene, settings["ground"], settings["ground_light"])

    if settings["blend"] is None:
        scene.render.filepath = settings["output"]
        bpy.ops.render.render(animation=True)
    else:
        # frames render next to the file, which holds every image
        scene.render.filepath = "//frame_"
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=settings["blend"], compress=True)


if __name__ == "__main__":
    with open(sys.argv[sys.argv.index("--") + 1]) as file_obj:
        main(json.load(file_obj))
