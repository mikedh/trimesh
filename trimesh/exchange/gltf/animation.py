"""
animation.py
--------------

Read and write glTF animations and skins.
"""

from functools import partial

import numpy as np

from ... import util
from ...constants import log, tol
from ...path.curve import evaluate_bezier
from ...scene.animation import Animation, Skin, interpolate_keyframes
from ...transformations import trs_from_matrix, trs_matrix
from ...typed import NDArray, NDArray2D, NDArray3D, float64

# default node transform parts in glTF order
DEFAULTS = {
    "translation": [0.0, 0.0, 0.0],
    "rotation": [0.0, 0.0, 0.0, 1.0],
    "scale": [1.0, 1.0, 1.0],
}
# columns of `Animation.trs` for each glTF path, `xyzw` from `wxyz`
COLUMNS = {"translation": [0, 1, 2], "rotation": [4, 5, 6, 3], "scale": [7, 8, 9]}
GLTF_ORDER = np.concatenate(list(COLUMNS.values()))

TO_GLTF = {"step": "STEP", "linear": "LINEAR", "cubic": "CUBICSPLINE"}
FROM_GLTF = {value: key for key, value in TO_GLTF.items()}

# glTF quaternions are `xyzw` and trimesh's are `wxyz`
WXYZ_FROM_XYZW = [3, 0, 1, 2]
XYZW_FROM_WXYZ = [1, 2, 3, 0]


def node_matrices(nodes: list) -> tuple[NDArray3D[float64], NDArray2D[float64]]:
    """
    Rest transforms of glTF nodes.

    Parameters
    ------------
    nodes : list
      glTF nodes with a `matrix` or TRS parts.

    Returns
    ------------
    matrices : (len(nodes), 4, 4) float
      Parent-to-node transforms.
    trs : (len(nodes), 10) float
      Translation, `wxyz` quaternion and scale of each.
    """
    trs = np.zeros((len(nodes), 10))
    trs[:, GLTF_ORDER] = [
        [v for k, d in DEFAULTS.items() for v in n.get(k, d)] for n in nodes
    ]
    matrices = trs_matrix(*np.split(trs, [3, 7], axis=1))
    # a stored matrix is column-major, exact, and may not decompose
    whole = [index for index, node in enumerate(nodes) if "matrix" in node]
    if len(whole) > 0:
        stored = np.reshape([nodes[i]["matrix"] for i in whole], (-1, 4, 4))
        matrices[whole] = stored.transpose((0, 2, 1))
        trs[whole] = np.concatenate(trs_from_matrix(matrices[whole]), axis=1)
    return matrices, trs


def parse_channel(
    animation: dict, channel: dict, access: list, times: NDArray | None = None
):
    """
    Keyframes of one glTF channel with STEP held as linear, or its samples.

    Parameters
    ------------
    animation : dict
      glTF animation holding the channel.
    channel : dict
      glTF channel.
    access : list
      Decoded accessor data.
    times : None or (m,) float
      Times to sample at.

    Returns
    ------------
    parsed : tuple or (m, width) float
      (node, path, keys, mode), or the samples at `times`.
    """
    path = channel["target"]["path"]
    sampler = animation["samplers"][channel["sampler"]]
    keys = np.float64(access[sampler["input"]]).reshape(-1)
    # cubic stores in-tangent, value and out-tangent per keyframe
    values = np.float64(access[sampler["output"]])
    values = values.reshape((len(keys), -1, len(DEFAULTS[path])))
    mode = FROM_GLTF[sampler.get("interpolation", "LINEAR")]
    if mode == "step":
        # hold each value until the float32 before the next keyframe,
        # which stays strictly increasing when exported
        before = np.nextafter(np.float32(keys[1:]), np.float32(-np.inf))
        keys = np.append(np.column_stack((keys[:-1], before)).ravel(), keys[-1:])
        values, mode = np.repeat(values, 2, axis=0)[:-1], "linear"
    if times is None:
        return channel["target"]["node"], path, keys, mode
    if mode == "linear":
        return interpolate_keyframes(
            keys, values[:, 0], times, rotation=path == "rotation"
        )
    # a cubic Hermite segment is the Bezier curve of its values and a third
    # of each tangent across it
    upper = np.clip(np.searchsorted(keys, times), 1, len(keys) - 1)
    lower = upper - 1
    width = np.maximum(keys[upper] - keys[lower], tol.zero)
    reach = width[:, None] / 3.0
    controls = (
        values[lower, 1],
        values[lower, 1] + reach * values[lower, 2],
        values[upper, 1] - reach * values[upper, 0],
        values[upper, 1],
    )
    fraction = np.clip((times - keys[lower]) / width, 0.0, 1.0)
    sampled = evaluate_bezier(np.stack(controls, axis=1), fraction)
    if path == "rotation":
        return util.unitize(sampled)
    return sampled


def parse_skins(header: dict, access: list, names: dict, skinned: dict) -> dict:
    """
    Load the glTF skins used by nodes of the scene.

    Parameters
    ------------
    header : dict
      The glTF header.
    access : list
      Decoded accessor data.
    names : dict
      {node index : node name}
    skinned : dict
      {node name : index of the glTF skin it uses}

    Returns
    ----------
    skins : dict
      {node name : `trimesh.scene.animation.Skin`}
    """
    loaded = {}
    for index in set(skinned.values()):
        try:
            skin = header["skins"][index]
            inverse = None
            if "inverseBindMatrices" in skin:
                # glTF matrices are column-major
                inverse = np.reshape(access[skin["inverseBindMatrices"]], (-1, 4, 4))
                inverse = inverse.transpose((0, 2, 1))
            joints = [names[joint] for joint in skin["joints"]]
            loaded[index] = Skin(joints=joints, inverse_bind_matrices=inverse)
        except (KeyError, IndexError, ValueError, TypeError):
            if tol.strict:
                raise
            log.warning(f"unable to load skin `{index}`", exc_info=True)
    return {node: loaded[index] for node, index in skinned.items() if index in loaded}


def parse_animations(header: dict, access: list, names: dict) -> list:
    """
    Load each glTF animation as a linear and a cubic `Animation`.

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
    animations : list
      `trimesh.scene.animation.Animation` objects.
    """
    result = []
    for number, animation in enumerate(header.get("animations", [])):
        stored = animation.get("channels", [])
        channels = [
            c
            for c in stored
            if c.get("target", {}).get("node") in names
            and c["target"].get("path") in DEFAULTS
        ]
        if any(c.get("target", {}).get("path") == "weights" for c in stored):
            log.warning("morph target `weights` animation is not supported")
        try:
            parsed = [parse_channel(animation, c, access) for c in channels]
        except (KeyError, IndexError, TypeError, ValueError):
            if tol.strict:
                raise
            log.warning(f"unable to load animation `{number}`", exc_info=True)
            continue
        # a node is cubic if any of its channels is, and linear otherwise
        cubic = {node for node, *_, mode in parsed if mode == "cubic"}
        for interpolation in ("linear", "cubic"):
            group = [
                (channel, node, path, keys)
                for channel, (node, path, keys, _) in zip(channels, parsed)
                if (node in cubic) == (interpolation == "cubic")
            ]
            if len(group) == 0:
                continue
            times = np.unique(np.concatenate([item[3] for item in group]))
            nodes = list(dict.fromkeys(item[1] for item in group))
            column = {node: index for index, node in enumerate(nodes)}
            # rest parts at every time, overwritten by each channel's samples
            rest = node_matrices([header["nodes"][n] for n in nodes])[1]
            trs = np.tile(rest, (len(times), 1, 1))
            for channel, node, path, _ in group:
                trs[:, column[node], COLUMNS[path]] = parse_channel(
                    animation, channel, access, times
                )
            result.append(
                Animation(
                    name=animation.get("name", f"animation_{number}"),
                    nodes=[names[node] for node in nodes],
                    times=times,
                    matrices=trs_matrix(*np.split(trs, [3, 7], axis=2)),
                    interpolation=interpolation,
                )
            )
    return result


def append_animations(tree: dict, buffer_items: dict, scene, node_index: dict) -> None:
    """
    Add the animations and skins of a scene to a glTF tree in place.

    Parameters
    ------------
    tree : dict
      glTF header, mutated in place.
    buffer_items : dict
      Buffer data, mutated in place.
    scene : trimesh.Scene
      Scene to export.
    node_index : dict
      {node name : index in `tree["nodes"]`}
    """
    # avoid a circular import
    from . import _data_append

    append = partial(_data_append, tree["accessors"], buffer_items)
    # {name : glTF animation}, merged as importers play only the first
    result = {}
    for animation in scene.animations:
        present = [i for i, node in enumerate(animation.nodes) if node in node_index]
        if len(present) < len(animation.nodes):
            if tol.strict:
                raise ValueError(f"`{animation.name}` animates missing nodes")
            log.warning(f"can't export the missing nodes `{animation.name}` animates")
        indexes = [node_index[animation.nodes[i]] for i in present]
        nodes = [tree["nodes"][index] for index in indexes]
        rest = node_matrices(nodes)[1]
        # animated nodes are stored as parts
        for node, row in zip(nodes, np.float32(rest[:, GLTF_ORDER])):
            node.pop("matrix", None)
            parts = zip(DEFAULTS.items(), np.split(row, [3, 7]))
            node.update({p: v.tolist() for (p, d), v in parts if (v != d).any()})
        trs = animation.trs[:, present]
        # ignore float32 noise from decomposition, and `q` is `-q`
        tolerance = 4.0 * np.finfo(np.float32).eps * max(np.abs(trs).max(), 1.0)
        moves = (np.abs(trs - rest) > tolerance).any(axis=0)
        moves[:, 3:7] &= (np.abs(trs[..., 3:7] + rest[:, 3:7]) > tolerance).any(axis=0)
        keyframes = trs
        if animation.interpolation == "cubic":
            # in-tangent, value and out-tangent per keyframe
            tangents = animation.tangents[:, present]
            keyframes = np.stack((tangents, trs, tangents), axis=1)
        keyframes = np.float32(keyframes).reshape((-1, *trs.shape[1:]))
        # (path, columns, node) of every channel in glTF path order
        moving = [
            (path, columns, node)
            for path, columns in COLUMNS.items()
            for node in np.flatnonzero(moves[:, columns].any(axis=1))
        ]
        if len(moving) == 0:
            continue
        merged = result.setdefault(
            animation.name, {"name": animation.name, "samplers": [], "channels": []}
        )
        times = append(
            {"componentType": 5126, "type": "SCALAR"}, np.float32(animation.times)
        )
        start = len(merged["samplers"])
        merged["channels"].extend(
            {"sampler": start + i, "target": {"node": indexes[node], "path": path}}
            for i, (path, _, node) in enumerate(moving)
        )
        merged["samplers"].extend(
            {
                "input": times,
                "output": append(
                    {"componentType": 5126, "type": f"VEC{len(columns)}"},
                    keyframes[:, node, columns],
                ),
                "interpolation": TO_GLTF[animation.interpolation],
            }
            for _, columns, node in moving
        )
    if len(result) > 0:
        tree["animations"] = list(result.values())

    # write each distinct skin once
    entries = []
    for node, skin in scene.skins.items():
        try:
            target = tree["nodes"][node_index[node]]
            joints = [node_index[joint] for joint in skin.joints]
        except KeyError:
            if tol.strict:
                raise
            log.warning(f"can't export the skin of `{node}` as a node is missing")
            continue
        # glTF matrices are column-major
        inverse = append(
            {"componentType": 5126, "type": "MAT4"},
            np.float32(skin.inverse_bind_matrices.transpose((0, 2, 1))),
        )
        entry = {"joints": joints, "inverseBindMatrices": inverse}
        if entry not in entries:
            entries.append(entry)
        target["skin"] = entries.index(entry)
    if len(entries) > 0:
        tree["skins"] = entries
