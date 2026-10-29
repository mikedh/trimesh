"""
animation.py
--------------

Read and write glTF animations and skins.
"""

from functools import partial

import numpy as np

from ... import util
from ...constants import log, tol
from ...scene.animation import INTERPOLATIONS, Animation, Skin, interpolate_keyframes
from ...transformations import trs_from_matrix, trs_matrix
from ...typed import NDArray3D, float64

# default node transform parts in glTF order
DEFAULTS = {
    "translation": [0.0, 0.0, 0.0],
    "rotation": [0.0, 0.0, 0.0, 1.0],
    "scale": [1.0, 1.0, 1.0],
}

TO_GLTF = {"step": "STEP", "linear": "LINEAR", "cubic": "CUBICSPLINE"}
FROM_GLTF = {value: key for key, value in TO_GLTF.items()}

# glTF quaternions are `xyzw` and trimesh's are `wxyz`
WXYZ_FROM_XYZW = [3, 0, 1, 2]
XYZW_FROM_WXYZ = [1, 2, 3, 0]


def node_matrices(nodes: list) -> NDArray3D[float64]:
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
    """
    matrices = np.tile(np.eye(4), (len(nodes), 1, 1))
    # most nodes are only a matrix or nothing, which needs no composing
    parts = [
        index for index, node in enumerate(nodes) if not DEFAULTS.keys().isdisjoint(node)
    ]
    if len(parts) > 0:
        translation, rotation, scale = (
            np.reshape([nodes[i].get(key, value) for i in parts], (-1, len(value)))
            for key, value in DEFAULTS.items()
        )
        matrices[parts] = trs_matrix(translation, rotation[:, WXYZ_FROM_XYZW], scale)
    for index, node in enumerate(nodes):
        # column-major
        if "matrix" in node:
            matrices[index] = np.reshape(node["matrix"], (4, 4)).T
    return matrices


def parse_animations(
    header: dict, access: list, names: dict, skinned: dict, matrices: NDArray3D[float64]
) -> tuple[list, dict]:
    """
    Load glTF animations and skins.

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
    matrices : (len(header["nodes"]), 4, 4) float
      Rest transform of every node from `node_matrices`.

    Returns
    ----------
    animations : list
      `trimesh.scene.animation.Animation` objects.
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
                inverse = np.reshape(
                    access[skin["inverseBindMatrices"]], (-1, 4, 4)
                ).transpose((0, 2, 1))
            joints = [names[joint] for joint in skin["joints"]]
            loaded[index] = Skin(joints=joints, inverse_bind_matrices=inverse)
        except (KeyError, IndexError, ValueError, TypeError):
            if tol.strict:
                raise
            log.warning(f"unable to load skin `{index}`", exc_info=True)
    skins = {node: loaded[index] for node, index in skinned.items() if index in loaded}

    result = []
    for number, animation in enumerate(header.get("animations", [])):
        channels = []
        for channel in animation.get("channels", []):
            target = channel.get("target", {})
            if target.get("path") == "weights":
                log.warning("morph target `weights` animation is not supported")
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
        # one `Animation` per smoothest channel interpolation, same name
        smoothest = {}
        for node, *_, mode in channels:
            smoothest[node] = max(
                smoothest.get(node, mode), mode, key=INTERPOLATIONS.index
            )
        for interpolation in INTERPOLATIONS:
            group = [c for c in channels if smoothest[c[0]] == interpolation]
            if len(group) == 0:
                continue
            times = np.unique(np.concatenate([channel[2] for channel in group]))
            nodes = list(dict.fromkeys(channel[0] for channel in group))
            # rest parts as stored, only decomposing nodes stored as a matrix
            stored = [header["nodes"][n] for n in nodes]
            rest = [
                np.reshape([node.get(key, value) for node in stored], (-1, len(value)))
                for key, value in DEFAULTS.items()
            ]
            rest[1] = rest[1][:, WXYZ_FROM_XYZW]
            whole = [index for index, node in enumerate(stored) if "matrix" in node]
            if len(whole) > 0:
                decomposed = trs_from_matrix(matrices[[nodes[i] for i in whole]])
                for part, values in zip(rest, decomposed):
                    part[whole] = values
            parts = {
                p: np.tile(part, (len(times), 1, 1)) for p, part in zip(DEFAULTS, rest)
            }
            column = {node: index for index, node in enumerate(nodes)}
            for node, path, keys, values, mode in group:
                # (k, 1 or 3, width), cubic stores in-tangent, value, out-tangent
                keys = keys.reshape(-1)
                values = values.reshape((len(keys), -1, parts[path].shape[2]))
                if path == "rotation":
                    values = values[..., WXYZ_FROM_XYZW]
                if len(keys) == len(times) and (keys == times).all():
                    # on its own keyframes every interpolation is the value
                    sampled = values[:, len(values[0]) // 2]
                elif mode == "cubic":
                    # cubic Hermite spline per the glTF spec
                    upper = np.clip(np.searchsorted(keys, times), 1, len(keys) - 1)
                    lower = np.maximum(upper - 1, 0)
                    span = keys[upper] - keys[lower]
                    step = np.divide(
                        times - keys[lower],
                        span,
                        out=np.zeros_like(times),
                        where=span > 0,
                    )
                    step, span = np.clip(step, 0.0, 1.0)[:, None], span[:, None]
                    sampled = (
                        (2.0 * step**3 - 3.0 * step**2 + 1.0) * values[lower, 1]
                        + span * (step**3 - 2.0 * step**2 + step) * values[lower, 2]
                        + (-2.0 * step**3 + 3.0 * step**2) * values[upper, 1]
                        + span * (step**3 - step**2) * values[upper, 0]
                    )
                    if path == "rotation":
                        sampled = util.unitize(sampled)
                else:
                    sampled = interpolate_keyframes(
                        keys, values[:, 0], times, mode == "step", path == "rotation"
                    )
                parts[path][:, column[node]] = sampled
            result.append(
                Animation(
                    name=animation.get("name", f"animation_{number}"),
                    nodes=[names[node] for node in nodes],
                    times=times,
                    matrices=trs_matrix(*parts.values()),
                    interpolation=interpolation,
                )
            )
    return result, skins


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

    # `trs` columns in glTF order
    order = np.r_[0:3, np.add(XYZW_FROM_WXYZ, 3), 7:10]
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
        # rest pose then keyframes as (1 + k, n, 10)
        rest = np.concatenate(trs_from_matrix(node_matrices(nodes)), axis=1)
        parts = np.concatenate((rest[None], animation.trs[:, present]))[..., order]
        keyframes = parts[1:]
        interpolation = animation.interpolation
        if interpolation == "cubic" and animation.spline is not None:
            tangent = animation.spline(animation.times, 1)[:, present][..., order]
            keyframes = np.stack((tangent, keyframes, tangent), axis=1)
            keyframes = keyframes.reshape((-1, *parts.shape[1:]))
        elif interpolation == "cubic":
            interpolation = "linear"
        sampler = {"interpolation": TO_GLTF[interpolation]}
        times = np.float32(animation.times)
        merged = {"name": animation.name, "samplers": [], "channels": []}
        merged = result.setdefault(animation.name, merged)
        for (path, default), part, values in zip(
            DEFAULTS.items(),
            np.split(parts, [3, 7], axis=2),
            np.split(np.float32(keyframes), [3, 7], axis=2),
        ):
            for node, value in zip(nodes, np.float32(part[0])):
                node.pop("matrix", None)
                if (value != default).any():
                    node[path] = value.tolist()
            # ignore float32 noise from decomposition
            tolerance = 4.0 * np.finfo(np.float32).eps * max(np.abs(part).max(), 1.0)
            moves = (np.abs(part[1:] - part[0]) > tolerance).any(axis=(0, 2))
            if path == "rotation":
                # `q` and `-q` are the same rotation
                moves &= (np.abs(part[1:] + part[0]) > tolerance).any(axis=(0, 2))
            for column in np.flatnonzero(moves):
                target = {"node": indexes[column], "path": path}
                merged["channels"].append(
                    {"sampler": len(merged["samplers"]), "target": target}
                )
                inputs = append({"componentType": 5126, "type": "SCALAR"}, times)
                vector = {"componentType": 5126, "type": f"VEC{len(default)}"}
                output = append(vector, values[:, column])
                merged["samplers"].append({"input": inputs, "output": output, **sampler})
    animations = [animation for animation in result.values() if animation["channels"]]
    if len(animations) > 0:
        tree["animations"] = animations

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
