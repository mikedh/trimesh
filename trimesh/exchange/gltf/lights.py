"""
lights.py
--------------

Read and write the `KHR_lights_punctual` glTF extension. Every light is
named after the node carrying it, so `Scene.lights` and the scene graph
refer to the same thing. An `EnvironmentLight`, which glTF has no place
for, is stored in the `extras` of its node.
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
_CLASSES = {value: key for key, value in _TYPES.items()}
# the key in a node's `extras` holding an `EnvironmentLight`
ENVIRONMENT = "environment_light"
# `EnvironmentLight` arrays stored by name, beside its softbox rows
_ENVIRONMENT_ARRAYS = ("sky", "ground")


def parse_lights(header, names):
    """
    Collect the `KHR_lights_punctual` lights nodes carry, one per node
    as two nodes may reference one entry of the document's array, and
    the `EnvironmentLight` in the `extras` of a node.

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
        extras = node.get("extras")
        if isinstance(extras, dict) and isinstance(extras.get(ENVIRONMENT), dict):
            entry = extras[ENVIRONMENT]
            lights.append(
                lighting.EnvironmentLight(
                    name=names[index],
                    color=color.to_rgba(
                        np.array(entry.get("color", [1.0, 1.0, 1.0]), dtype=np.float64)
                    ),
                    intensity=entry.get("intensity", 1.0),
                    **{key: entry.get(key) for key in _ENVIRONMENT_ARRAYS},
                    softboxes=[
                        tuple(box[field] for field in lighting.SOFTBOX.names)
                        for box in entry.get("softboxes", [])
                    ],
                )
            )
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
            "name": names[index],
            "intensity": entry.get("intensity", 1.0),
            "radius": entry.get("range"),
            # glTF lights are white unless they say otherwise
            "color": color.to_rgba(
                np.array(entry.get("color", [1.0, 1.0, 1.0]), dtype=np.float64)
            ),
        }
        if constructor is lighting.SpotLight:
            spot = entry.get("spot", {})
            # outer first as the inner setter validates against it
            kwargs["outerConeAngle"] = spot.get("outerConeAngle", np.pi / 4.0)
            kwargs["innerConeAngle"] = spot.get("innerConeAngle", 0.0)

        lights.append(constructor(**kwargs))

    return lights


def append_lights(lights, tree, node_index):
    """
    Write the light array and every node's reference into it into a
    tree in place, and an `EnvironmentLight` into the `extras` of its node.

    Parameters
    ------------
    lights : list
      Sequence of `trimesh.scene.lighting.Light`.
    tree : dict
      GLTF header, mutated in place, with `nodes` already populated.
    node_index : dict
      Mapping of {node name : index in `tree["nodes"]`}
    """
    from ...scene.lighting import SOFTBOX, EnvironmentLight
    from ...visual import color

    stored = []
    for light in lights:
        index = node_index.get(light.name)
        if index is None:
            log.warning(f"light `{light.name}` has no node!")
            continue
        node = tree["nodes"][index]
        entry = {
            "name": light.name,
            # the extension stores linear RGB in the 0.0 - 1.0 range
            "color": color.to_float(light.color)[:3].tolist(),
            "intensity": float(light.intensity),
        }
        if isinstance(light, EnvironmentLight):
            # merged into a copy of the node's extras so the scene isn't changed
            entry.update(
                {key: getattr(light, key).tolist() for key in _ENVIRONMENT_ARRAYS}
            )
            # one object per softbox with a value per `SOFTBOX` field
            entry["softboxes"] = [
                {field: box[field].tolist() for field in SOFTBOX.names}
                for box in light.softboxes
            ]
            node["extras"] = {**node.get("extras", {}), ENVIRONMENT: entry}
            continue
        entry["type"] = _TYPES[type(light).__name__]
        if light.radius is not None:
            entry["range"] = float(light.radius)
        if entry["type"] == "spot":
            # only a spot has a cone, and the extension defaults differ
            # from ours so always write both rather than guessing
            entry["spot"] = {
                "innerConeAngle": float(light.innerConeAngle),
                "outerConeAngle": float(light.outerConeAngle),
            }
        # the node holding the light points at it, merged into a copy of
        # any extensions the node already has so the scene isn't changed
        node["extensions"] = {**node.get("extensions", {}), NAME: {"light": len(stored)}}
        stored.append(entry)

    if len(stored) > 0:
        tree["extensions"] = {**tree.get("extensions", {}), NAME: {"lights": stored}}
