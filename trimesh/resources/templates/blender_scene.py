"""
blender_scene.py
------------------

Build a Blender scene from a GLB exported by trimesh then render it to
numbered PNG frames or save it as a `.blend`, run by
`trimesh.interfaces.blender` in Blender 4.3 or later as:
`blender --background --python blender_scene.py -- settings.json`

Blender's glTF importer reads geometry, materials, cameras, lights and
animation. This adds only what a GLB can't say: the camera, frames,
exposure, Blender's own neutral world or a transparent film, and lines
and points as tubes and spheres, which Blender doesn't draw.
"""

import json
import math
import sys

import bpy


def link(tree, kind, inputs=None, **properties):
    """A new node with properties set then inputs, linked if they are sockets."""
    node = tree.nodes.new(kind)
    for key, value in properties.items():
        setattr(node, key, value)
    for key, value in (inputs or {}).items():
        if isinstance(value, bpy.types.NodeSocket):
            tree.links.new(value, node.inputs[key])
        else:
            node.inputs[key].default_value = value
    return node


def material_tree(name):
    """An empty material and its node tree."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.node_tree.nodes.clear()
    return material, material.node_tree


def paint(item):
    """Colors for an object the importer left bare: plastic or emission."""
    material, tree = material_tree(item.name)
    if len(item.data.polygons) > 0:
        shader = link(tree, "ShaderNodeBsdfPrincipled", {"Roughness": 0.5})
        socket = "Base Color"
    else:
        shader = link(tree, "ShaderNodeEmission", {"Color": (0.0, 0.0, 0.0, 1.0)})
        socket = "Color"
    for attribute in item.data.color_attributes[:1]:
        color = link(tree, "ShaderNodeVertexColor", layer_name=attribute.name)
        tree.links.new(color.outputs["Color"], shader.inputs[socket])
    link(tree, "ShaderNodeOutputMaterial", {"Surface": shader.outputs[0]})
    return material


def thicken(item, radius):
    """Draw the edges of an object as tubes or its vertices as spheres."""
    group = bpy.data.node_groups.new(item.name, "GeometryNodeTree")
    for direction in ("INPUT", "OUTPUT"):
        group.interface.new_socket(
            "Geometry", in_out=direction, socket_type="NodeSocketGeometry"
        )
    source = group.nodes.new("NodeGroupInput").outputs[0]
    if len(item.data.edges) > 0:
        curve = link(group, "GeometryNodeMeshToCurve", {"Mesh": source})
        profile = link(group, "GeometryNodeCurvePrimitiveCircle", {"Radius": radius})
        solid = link(
            group,
            "GeometryNodeCurveToMesh",
            {"Curve": curve.outputs[0], "Profile Curve": profile.outputs["Curve"]},
        )
    else:
        ball = link(group, "GeometryNodeMeshIcoSphere", {"Radius": radius})
        spheres = link(
            group,
            "GeometryNodeInstanceOnPoints",
            {"Points": source, "Instance": ball.outputs["Mesh"]},
        )
        solid = link(
            group, "GeometryNodeRealizeInstances", {"Geometry": spheres.outputs[0]}
        )
    painted = link(
        group,
        "GeometryNodeSetMaterial",
        {"Geometry": solid.outputs[0], "Material": item.material_slots[0].material},
    )
    group.links.new(painted.outputs[0], group.nodes.new("NodeGroupOutput").inputs[0])
    item.modifiers.new(item.name, "NODES").node_group = group


def main(settings):
    """Build the scene from `settings` then render or save it."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    # set before importing: the importer converts keyframe seconds to frames
    scene.render.fps = settings["fps"]
    scene.render.fps_base = settings["fps_base"]
    # "COMPAT" reads light intensity in the glTF units the viewer uses
    bpy.ops.import_scene.gltf(
        filepath=settings["input"], export_import_convert_lighting_mode="COMPAT"
    )
    scene.camera = bpy.data.objects[settings["camera"]]
    scene.render.resolution_x = settings["width"]
    scene.render.resolution_y = settings["height"]
    scene.render.resolution_percentage = 100
    # the importer puts time zero on frame zero
    scene.frame_start = 0
    scene.frame_end = settings["count"] - 1

    # EEVEE is `BLENDER_EEVEE_NEXT` from Blender 4.2 until 5.0
    engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    scene.render.engine = "BLENDER_EEVEE"
    if "BLENDER_EEVEE_NEXT" in engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = settings["samples"]
    scene.render.film_transparent = not settings["show_environment"]
    scene.view_settings.view_transform = "Khronos PBR Neutral"
    scene.view_settings.exposure = math.log2(settings["exposure"])
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    # lines and points are as wide as pixels are across, which in world
    # units is about the scene's scale over the image's height
    for item in [i for i in scene.objects if i.type == "MESH"]:
        bare = len(item.data.polygons) == 0
        if bare or len(item.material_slots) == 0:
            item.data.materials.clear()
            item.data.materials.append(paint(item))
        if bare:
            width = settings["point_size"]
            if len(item.data.edges) > 0:
                width = settings["line_width"]
            thicken(item, 0.5 * width * settings["pixel"])
            # the viewer's lines and points don't cast shadows
            item.visible_shadow = False

    # Blender's default world, which is a constant dim gray
    scene.world = bpy.data.worlds.new("world")

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
