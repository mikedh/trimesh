"""
animation.py
--------------

Read and write glTF keyframed animation and skins, where glTF
orders quaternions `x, y, z, w` and trimesh orders them `w, x, y, z`.
"""

import numpy as np

from ...constants import log, tol
from ...scene.animation import Animation, Interpolation
from ...transformations import sample_channel, trs_from_matrix, trs_matrix
from ...typed import NDArray3D, float64

# the value of each part of a transform a node doesn't store, in glTF order
DEFAULTS = {
    "translation": [0.0, 0.0, 0.0],
    "rotation": [0.0, 0.0, 0.0, 1.0],
    "scale": [1.0, 1.0, 1.0],
}

# the glTF sampler interpolation each trimesh interpolation is written as
INTERPOLATIONS = {
    Interpolation.LINEAR: "LINEAR",
    Interpolation.STEP: "STEP",
    Interpolation.CUBIC: "CUBICSPLINE",
}
# and the one each is loaded as
FROM_GLTF = {value: key for key, value in INTERPOLATIONS.items()}

# quaternion components reordered between glTF `x, y, z, w` and
# trimesh `w, x, y, z`
WXYZ_FROM_XYZW = [3, 0, 1, 2]
XYZW_FROM_WXYZ = [1, 2, 3, 0]


def node_matrices(nodes: list) -> NDArray3D[float64]:
    """
    The transform every glTF node is at rest, all computed at once.

    Parameters
    ------------
    nodes : list
      glTF nodes which each store a `matrix` or any of its parts.

    Returns
    ------------
    matrices : (len(nodes), 4, 4) float
      Homogeneous transform of every node relative to its parent.
    """
    # every node's parts with the default for any it doesn't store
    translation, rotation, scale = (
        np.reshape([node.get(key, value) for node in nodes], (-1, len(value)))
        for key, value in DEFAULTS.items()
    )
    matrices = trs_matrix(translation, rotation[:, WXYZ_FROM_XYZW], scale)
    # a node which stores a `matrix` stores no parts
    explicit = [index for index, node in enumerate(nodes) if "matrix" in node]
    # glTF matrices are column-major
    stored = np.reshape([nodes[index]["matrix"] for index in explicit], (-1, 4, 4))
    matrices[explicit] = stored.transpose((0, 2, 1))
    return matrices


def parse_animations(header: dict, access: list, names: dict) -> list:
    """
    Load each glTF animation as an `Animation` of every node it moves at
    every keyframe of any of its channels, which is exact at those times,
    between which it is a cubic spline if any channel is, held if every
    channel is, and otherwise blended linearly.

    Parameters
    ------------
    header : dict
      The glTF header.
    access : list
      Decoded accessor data.
    names : dict
      {node index : node name} of the nodes in the loaded scene.

    Returns
    ----------
    animations : list
      `trimesh.scene.animation.Animation` objects.
    """
    result = []
    for number, animation in enumerate(header.get("animations", [])):
        channels = []
        for channel in animation.get("channels", []):
            target = channel.get("target", {})
            if target.get("path") == "weights":
                log.warning("morph target `weights` animation is not supported")
            # a node outside the loaded scene isn't there to animate
            if target.get("node") not in names or target.get("path") not in DEFAULTS:
                continue
            try:
                sampler = animation["samplers"][channel["sampler"]]
                keys, values = access[sampler["input"]], access[sampler["output"]]
                mode = FROM_GLTF[sampler.get("interpolation", "LINEAR")]
                channels.append((target["node"], target["path"], keys, values, mode))
            except (KeyError, IndexError, TypeError):
                if tol.strict:
                    raise
                log.warning("unable to load animation channel", exc_info=True)
        if len(channels) == 0:
            continue
        times = np.unique(np.concatenate([channel[2] for channel in channels]))
        nodes = list(dict.fromkeys(channel[0] for channel in channels))
        # every node at rest at every keyframe, then each channel sampled
        rest = trs_from_matrix(node_matrices([header["nodes"][n] for n in nodes]))
        parts = {p: np.tile(part, (len(times), 1, 1)) for p, part in zip(DEFAULTS, rest)}
        for node, path, keys, values, mode in channels:
            # (k, 1 or 3, width) where a cubic spline stores an arriving
            # tangent, value, and leaving tangent per keyframe
            values = values.reshape((len(keys), -1, parts[path].shape[2]))
            if path == "rotation":
                values = values[..., WXYZ_FROM_XYZW]
            parts[path][:, nodes.index(node)] = sample_channel(
                times=keys.reshape(-1),
                values=values[:, values.shape[1] // 2],
                tangent_in=values[:, 0],
                tangent_out=values[:, -1],
                mode=INTERPOLATIONS[mode],
                query=times,
                rotation=path == "rotation",
            )
        # cubic if any channel is, held if every one is, otherwise linear
        modes = {channel[4] for channel in channels}
        interpolation = Interpolation.LINEAR
        if Interpolation.CUBIC in modes:
            interpolation = Interpolation.CUBIC
        elif modes == {Interpolation.STEP}:
            interpolation = Interpolation.STEP
        result.append(
            Animation(
                name=animation.get("name", f"animation_{number}"),
                nodes=[names[node] for node in nodes],
                times=times,
                matrices=trs_matrix(*parts.values()),
                interpolation=interpolation,
            )
        )
    return result


def append_animations(
    tree: dict, buffer_items: dict, animations: list, node_index: dict
) -> None:
    """
    Add animations to a glTF tree in place, those sharing a name as one
    glTF animation as importers play only the first. An animated node's
    `matrix` is rewritten into parts as the spec forbids animating one,
    and only parts which move or aren't where the node rests animate,
    interpolated as the animation is with a spline's derivative as the
    tangents of a glTF cubic spline.

    Parameters
    ------------
    tree : dict
      glTF header, mutated in place.
    buffer_items : dict
      Buffer data, mutated in place.
    animations : list
      `trimesh.scene.animation.Animation` objects.
    node_index : dict
      {node name : index in `tree["nodes"]`}
    """
    # imported here as this module is imported by `gltf/__init__.py`
    from . import _data_append

    # {name : glTF animation} which every animation of the name adds to
    result = {}
    for animation in animations:
        present = [i for i, node in enumerate(animation.nodes) if node in node_index]
        if len(present) < len(animation.nodes):
            if tol.strict:
                raise ValueError(f"`{animation.name}` animates missing nodes")
            log.warning(f"can't export the missing nodes `{animation.name}` animates")
        indexes = [node_index[animation.nodes[i]] for i in present]
        nodes = [tree["nodes"][index] for index in indexes]
        # every node at rest ahead of its keyframes as (1 + k, n, width)
        parts = [
            np.concatenate((rest[None], part[:, present]))
            for rest, part in zip(trs_from_matrix(node_matrices(nodes)), animation.trs)
        ]
        parts[1] = parts[1][..., XYZW_FROM_WXYZ]
        times = np.float32(animation.times)
        # a spline writes its derivative as both tangents of a keyframe
        tangents = [None] * 3
        interpolation = animation.interpolation
        if interpolation == Interpolation.CUBIC and animation.spline is not None:
            derivative = animation.spline(animation.times, 1)[:, present]
            tangents = np.split(np.float32(derivative), [3, 7], axis=2)
            tangents[1] = tangents[1][..., XYZW_FROM_WXYZ]
        elif interpolation == Interpolation.CUBIC:
            interpolation = Interpolation.LINEAR
        sampler = {"interpolation": INTERPOLATIONS[interpolation]}
        for (path, default), part, tangent in zip(DEFAULTS.items(), parts, tangents):
            still, values = np.float32(part[0]), np.float32(part[1:])
            for node, value in zip(nodes, still):
                node.pop("matrix", None)
                if (value != default).any():
                    node[path] = value.tolist()
            # a path moves if it leaves its rest by more than the few float32
            # units in the last place a matrix decomposition leaves behind
            tolerance = 4.0 * np.finfo(np.float32).eps * max(np.abs(part).max(), 1.0)
            moves = (np.abs(part[1:] - part[0]) > tolerance).any(axis=(0, 2))
            if path == "rotation":
                # a quaternion and its negation are the same rotation
                moves &= (np.abs(part[1:] + part[0]) > tolerance).any(axis=(0, 2))
            if tangent is not None:
                # arriving tangent, value, and leaving tangent per keyframe
                values = np.stack((tangent, values, tangent), axis=1).reshape(
                    (-1, *values.shape[1:])
                )
            vector = f"VEC{len(default)}"
            for column in np.flatnonzero(moves):
                # equal accessors are written once by `_data_append`
                inputs, output = (
                    _data_append(
                        acc=tree["accessors"],
                        buff=buffer_items,
                        blob={"componentType": 5126, "type": kind},
                        data=np.ascontiguousarray(data),
                    )
                    for kind, data in (("SCALAR", times), (vector, values[:, column]))
                )
                merged = result.setdefault(
                    animation.name,
                    {"name": animation.name, "samplers": [], "channels": []},
                )
                target = {"node": indexes[column], "path": path}
                merged["channels"].append(
                    {"sampler": len(merged["samplers"]), "target": target}
                )
                merged["samplers"].append({"input": inputs, "output": output, **sampler})
    if len(result) > 0:
        tree["animations"] = list(result.values())


def parse_skins(header: dict, access: list, names: dict) -> list:
    """
    Load every glTF skin, in order so a node's `skin` index finds it.

    Parameters
    ------------
    header : dict
      The glTF header.
    access : list
      Decoded accessor data.
    names : dict
      {node index : node name}

    Returns
    ----------
    skins : list
      `trimesh.scene.animation.Skin` or None if it couldn't be loaded.
    """
    from ...scene.animation import Skin

    result = []
    for number, skin in enumerate(header.get("skins", [])):
        try:
            inverse = None
            if "inverseBindMatrices" in skin:
                # glTF matrices are column-major
                inverse = np.reshape(
                    access[skin["inverseBindMatrices"]], (-1, 4, 4)
                ).transpose((0, 2, 1))
            result.append(
                Skin(
                    joints=[names[index] for index in skin["joints"]],
                    inverse_bind_matrices=inverse,
                )
            )
        except (KeyError, IndexError, ValueError, TypeError):
            if tol.strict:
                raise
            log.warning(f"unable to load skin `{number}`", exc_info=True)
            result.append(None)
    return result


def append_skins(tree: dict, buffer_items: dict, skins: dict, node_index: dict) -> None:
    """
    Add skins to a glTF tree in place, writing each distinct skin once
    however many nodes use it.

    Parameters
    ------------
    tree : dict
      glTF header, mutated in place.
    buffer_items : dict
      Buffer data, mutated in place.
    skins : dict
      {node name : `trimesh.scene.animation.Skin`}
    node_index : dict
      {node name : index in `tree["nodes"]`}
    """
    # imported here as this module is imported by `gltf/__init__.py`
    from . import _data_append

    # equal accessors are written once by `_data_append`, so equal skins are
    # equal entries which every node using one shares
    entries = []
    for node, skin in skins.items():
        try:
            target = tree["nodes"][node_index[node]]
            joints = [node_index[joint] for joint in skin.joints]
        except KeyError:
            if tol.strict:
                raise
            log.warning(f"can't export the skin of `{node}` as a node is missing")
            continue
        accessor = _data_append(
            acc=tree["accessors"],
            buff=buffer_items,
            blob={"componentType": 5126, "type": "MAT4"},
            # glTF matrices are column-major
            data=np.ascontiguousarray(
                skin.inverse_bind_matrices.transpose((0, 2, 1)), dtype=np.float32
            ),
        )
        entry = {"joints": joints, "inverseBindMatrices": accessor}
        if entry not in entries:
            entries.append(entry)
        target["skin"] = entries.index(entry)
    if len(entries) > 0:
        tree["skins"] = entries
