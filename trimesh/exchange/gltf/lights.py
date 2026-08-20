"""
lights.py
--------------

Read and write the `KHR_lights_punctual` GLTF extension.

A file stores its lights once in a document-level array and has each node
reference one by index, so both halves are written and read here together
rather than from opposite ends of the exporter. Every light is named after
the node carrying it, which is what lets `Scene.lights` and the scene
graph refer to the same thing.

This is handled directly rather than through `extensions.py`, whose
registry dispatches handlers for extensions found on a material, texture,
or primitive while it is being parsed. A light is document-level and is
written alongside the camera, which the exporter also handles inline.
"""

import numpy as np

from ...constants import log

# the extension key, and trimesh light class name to GLTF light
# type with the reverse for loading
NAME = "KHR_lights_punctual"
_TYPES = {
    "DirectionalLight": "directional",
    "PointLight": "point",
    "SpotLight": "spot",
}
_CLASSES = {v: k for k, v in _TYPES.items()}


def parse_lights(header, names):
    """
    Collect the `KHR_lights_punctual` lights a header defines.

    A file stores its lights once in a document-level array and has each
    node reference one by index. Walk the nodes rather than the array: a
    light is only in the scene if a node carries it, and two nodes may
    reference one entry. Each is named after the node carrying it, which
    is what lets `Scene.lights` and the scene graph refer to one thing.

    Parameters
    ------------
    header : dict
      Parsed GLTF header.
    names : dict
      Mapping of {node index in the file : node name in the scene}

    Returns
    ----------
    lights : list
      One `trimesh.scene.lighting.Light` per node reference, in file order.
    """
    from ...scene import lighting
    from ...visual import color

    stored = header.get("extensions", {}).get(NAME, {}).get("lights", [])

    lights = []
    for index, node in enumerate(header.get("nodes", [])):
        reference = node.get("extensions", {}).get(NAME, {}).get("light")
        if reference is None:
            continue
        if not isinstance(reference, int) or reference >= len(stored):
            log.warning("node references a missing light!")
            continue

        entry = stored[reference]
        constructor = getattr(lighting, _CLASSES.get(entry.get("type"), ""), None)
        if constructor is None:
            log.warning(f"unsupported light type `{entry.get('type')}`!")
            continue

        kwargs = {
            # name it after the node so `Scene.lights` and the graph
            # agree, which is what a re-export needs
            "name": names[index],
            "intensity": entry.get("intensity", 1.0),
            "radius": entry.get("range"),
        }
        if "color" in entry:
            kwargs["color"] = color.to_rgba(np.array(entry["color"], dtype=np.float64))
        if constructor is lighting.SpotLight:
            spot = entry.get("spot", {})
            # outer first as the inner setter validates against it
            kwargs["outerConeAngle"] = spot.get("outerConeAngle", np.pi / 4.0)
            kwargs["innerConeAngle"] = spot.get("innerConeAngle", 0.0)

        lights.append(constructor(**kwargs))

    return lights


def append_lights(lights, tree, node_index):
    """
    Write the light array and every node reference into a tree in-place.

    A node refers to a light by its position in the document array, which
    is why both halves are written here rather than from opposite ends.

    Parameters
    ------------
    lights : list
      Sequence of `trimesh.scene.lighting.Light`.
    tree : dict
      GLTF header, mutated in place, with `nodes` already populated.
    node_index : dict
      Mapping of {node name : index in `tree["nodes"]`}
    """
    from ...visual import color

    stored = []
    for light in lights:
        entry = {
            "name": light.name,
            "type": _TYPES[type(light).__name__],
            # the extension stores linear RGB in the 0.0 - 1.0 range
            "color": color.to_float(light.color)[:3].tolist(),
            "intensity": float(light.intensity),
        }
        if light.radius is not None:
            entry["range"] = float(light.radius)
        if entry["type"] == "spot":
            # only a spot has a cone, and the extension defaults differ
            # from ours so always write both rather than guessing
            entry["spot"] = {
                "innerConeAngle": float(light.innerConeAngle),
                "outerConeAngle": float(light.outerConeAngle),
            }
        stored.append(entry)

    tree["extensions"] = {**tree.get("extensions", {}), NAME: {"lights": stored}}

    for i, light in enumerate(lights):
        index = node_index.get(light.name)
        if index is None:
            log.warning(f"light `{light.name}` has no node!")
            continue
        node = tree["nodes"][index]
        node["extensions"] = {**node.get("extensions", {}), NAME: {"light": i}}
