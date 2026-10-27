"""
blender_scene.py
------------------

Render a GLB exported by trimesh to numbered PNG frames, run by
`trimesh.interfaces.blender.build` in Blender 4.3 or
later with JSON settings as the last argument. Blender's glTF importer
reads the scene, and this adds only what a GLB can't say: the camera,
frames, exposure, Blender's own neutral world or a transparent film, and
lines and points as tubes and spheres, which Blender doesn't draw.
"""

import json
import math
import sys

import bpy

# the oldest Blender this runs in, which the error trimesh raises shows
if bpy.app.version < (4, 3, 0):
    raise RuntimeError(
        f"rendering needs Blender 4.3 or later, not {bpy.app.version_string}"
    )


def link(tree, kind, inputs=None):
    """A new node with its inputs set, or linked if they are sockets."""
    node = tree.nodes.new(kind)
    for key, value in (inputs or {}).items():
        if isinstance(value, bpy.types.NodeSocket):
            tree.links.new(value, node.inputs[key])
        else:
            node.inputs[key].default_value = value
    return node


def paint(item):
    """Colors for an object the importer left bare: plastic or emission."""
    material = bpy.data.materials.new(item.name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    if len(item.data.polygons) > 0:
        shader = link(tree, "ShaderNodeBsdfPrincipled", {"Roughness": 0.5})
        socket = "Base Color"
    else:
        shader = link(tree, "ShaderNodeEmission", {"Color": (0.0, 0.0, 0.0, 1.0)})
        socket = "Color"
    for attribute in item.data.color_attributes[:1]:
        color = link(tree, "ShaderNodeVertexColor")
        color.layer_name = attribute.name
        tree.links.new(color.outputs["Color"], shader.inputs[socket])
    link(tree, "ShaderNodeOutputMaterial", {"Surface": shader.outputs[0]})
    return material


def thicken(item, settings):
    """Draw the edges of an object as tubes or its vertices as spheres."""
    group = bpy.data.node_groups.new(item.name, "GeometryNodeTree")
    for side in ("INPUT", "OUTPUT"):
        group.interface.new_socket(
            "Geometry", in_out=side, socket_type="NodeSocketGeometry"
        )
    source = group.nodes.new("NodeGroupInput").outputs[0]
    if len(item.data.edges) > 0:
        radius = 0.5 * settings["line_width"] * settings["pixel"]
        curve = link(group, "GeometryNodeMeshToCurve", {"Mesh": source})
        profile = link(group, "GeometryNodeCurvePrimitiveCircle", {"Radius": radius})
        inputs = {"Curve": curve.outputs[0], "Profile Curve": profile.outputs["Curve"]}
        solid = link(group, "GeometryNodeCurveToMesh", inputs)
    else:
        radius = 0.5 * settings["point_size"] * settings["pixel"]
        ball = link(group, "GeometryNodeMeshIcoSphere", {"Radius": radius})
        inputs = {"Points": source, "Instance": ball.outputs["Mesh"]}
        spheres = link(group, "GeometryNodeInstanceOnPoints", inputs)
        inputs = {"Geometry": spheres.outputs[0]}
        solid = link(group, "GeometryNodeRealizeInstances", inputs)
    inputs = {"Geometry": solid.outputs[0], "Material": item.material_slots[0].material}
    painted = link(group, "GeometryNodeSetMaterial", inputs)
    group.links.new(painted.outputs[0], group.nodes.new("NodeGroupOutput").inputs[0])
    item.modifiers.new(item.name, "NODES").node_group = group


def main(settings):
    """Build the scene from `settings` then render or save it."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    # set before importing as it converts keyframe seconds to frames with
    # an integer `fps` divided by `fps_base`, which is at least one
    scene.render.fps = max(round(settings["fps"]), 1)
    scene.render.fps_base = scene.render.fps / settings["fps"]
    # "COMPAT" reads light intensity in the glTF units the viewer uses
    bpy.ops.import_scene.gltf(
        filepath=settings["input"], export_import_convert_lighting_mode="COMPAT"
    )
    scene.camera = bpy.data.objects[settings["camera"]]
    scene.render.resolution_x, scene.render.resolution_y = settings["resolution"]
    # the importer puts time zero on frame zero
    scene.frame_start, scene.frame_end = 0, settings["count"] - 1
    # EEVEE is `BLENDER_EEVEE_NEXT` from Blender 4.2 until 5.0
    scene.render.engine = "BLENDER_EEVEE"
    if bpy.app.version < (5, 0, 0):
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = settings["samples"]
    scene.render.film_transparent = not settings["show_environment"]
    scene.view_settings.view_transform = "Khronos PBR Neutral"
    scene.view_settings.exposure = math.log2(settings["exposure"])
    scene.render.image_settings.color_mode = "RGBA"

    # lines and points are as wide as pixels are across, which in world
    # units is about the scene's scale over the image's height
    for item in [i for i in scene.objects if i.type == "MESH"]:
        bare = len(item.data.polygons) == 0
        if bare or len(item.material_slots) == 0:
            item.data.materials.clear()
            item.data.materials.append(paint(item))
        if bare:
            thicken(item, settings)
            # the viewer's lines and points don't cast shadows
            item.visible_shadow = False

    # Blender's default world, which is a constant dim gray
    scene.world = bpy.data.worlds.new("world")

    scene.render.filepath = settings["output"]
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main(json.loads(sys.argv[-1]))
