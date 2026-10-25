"""
animation.py
--------------

Read and write glTF keyframed animation and skins.

glTF animates the `translation`, `rotation`, and `scale` of nodes with
samplers, each an accessor of times and one of values. trimesh stores
an `Animation` of scene graph edge matrices on one timeline, so this is
where channels become matrices and back. glTF orders quaternions
`x, y, z, w` where trimesh orders them `w, x, y, z`.
"""

from collections import defaultdict

import numpy as np

from ... import util
from ...constants import log, tol
from ...transformations import (
    keyframe_interval,
    quaternion_slerp,
    trs_from_matrix,
    trs_matrix,
)
from ...typed import ArrayLike, NDArray, NDArray1D, NDArray3D, float64

# the value of each channel a node doesn't store, in glTF order
DEFAULTS = {
    "translation": [0.0, 0.0, 0.0],
    "rotation": [0.0, 0.0, 0.0, 1.0],
    "scale": [1.0, 1.0, 1.0],
}

# samples per second taken of a cubic spline when it is loaded
SPLINE_RATE = 60.0


def trs_from_node(
    node: dict,
) -> tuple[NDArray1D[float64], NDArray1D[float64], NDArray1D[float64]]:
    """
    The translation, rotation, and scale a glTF node is at rest.

    Parameters
    ------------
    node : dict
      A glTF node which stores a `matrix` or any of its channels.

    Returns
    ------------
    translation : (3,) float
      Translation of the node.
    quaternion : (4,) float
      Rotation ordered `w, x, y, z`.
    scale : (3,) float
      Scale of the node.
    """
    if "matrix" in node:
        # glTF matrices are column-major
        return trs_from_matrix(np.reshape(node["matrix"], (4, 4)).T)
    translation, rotation, scale = (
        np.array(node.get(key, value), dtype=np.float64)
        for key, value in DEFAULTS.items()
    )
    return translation, rotation[[3, 0, 1, 2]], scale


def node_matrices(nodes: list) -> NDArray3D[float64]:
    """
    The transform every glTF node is at rest, all computed at once.

    Parameters
    ------------
    nodes : list
      glTF nodes which each store a `matrix` or any of its channels.

    Returns
    ------------
    matrices : (len(nodes), 4, 4) float
      Homogeneous transform of every node relative to its parent.
    """
    # every node's channels with the default for any it doesn't store
    translation, rotation, scale = (
        np.array([node.get(key, value) for node in nodes], dtype=np.float64).reshape(
            (-1, len(value))
        )
        for key, value in DEFAULTS.items()
    )
    matrices = trs_matrix(translation, rotation[:, [3, 0, 1, 2]], scale)
    # a node which stores a `matrix` stores no channels
    explicit = [index for index, node in enumerate(nodes) if "matrix" in node]
    # glTF matrices are column-major
    matrices[explicit] = (
        np.array([nodes[index]["matrix"] for index in explicit], dtype=np.float64)
        .reshape((-1, 4, 4))
        .transpose((0, 2, 1))
    )
    return matrices


def sample_channel(
    times: ArrayLike,
    values: ArrayLike,
    tangent_in: ArrayLike | None,
    tangent_out: ArrayLike | None,
    mode: str,
    query: ArrayLike,
    rotation: bool,
) -> NDArray[float64]:
    """
    Sample glTF channels which share keyframe times at any times,
    holding their first and last keyframe outside of them.

    Parameters
    ------------
    times : (k,) float
      Keyframe times in seconds.
    values : (k, d) or (k, c, d) float
      Value at each keyframe of one or `c` channels, quaternions
      ordered `w, x, y, z`.
    tangent_in : (k, d) or (k, c, d) float or None
      Arriving cubic tangents, only used by "CUBICSPLINE".
    tangent_out : (k, d) or (k, c, d) float or None
      Leaving cubic tangents, only used by "CUBICSPLINE".
    mode : str
      glTF interpolation: "LINEAR", "STEP", or "CUBICSPLINE".
    query : (m,) float
      Times to sample at.
    rotation : bool
      If the values are quaternions.

    Returns
    ------------
    sampled : (m, d) or (m, c, d) float
      The channels at each time.
    """
    values = np.asanyarray(values, dtype=np.float64)
    lower, upper, width, fraction = keyframe_interval(times, query)
    # one weight per query broadcast across every channel and component
    column = (-1,) + (1,) * (values.ndim - 1)
    if mode == "STEP":
        return values[np.where(fraction >= 1.0, upper, lower)]
    if mode == "CUBICSPLINE":
        # glTF's cubic Hermite spline, with tangents per second
        step, span = fraction.reshape(column), width.reshape(column)
        squared, cubed = step**2, step**3
        sampled = (
            (2.0 * cubed - 3.0 * squared + 1.0) * values[lower]
            + span * (cubed - 2.0 * squared + step) * tangent_out[lower]
            + (-2.0 * cubed + 3.0 * squared) * values[upper]
            + span * (cubed - squared) * tangent_in[upper]
        )
        if rotation:
            # the spec blends quaternion components and then normalizes
            sampled = util.unitize(sampled.reshape((-1, 4))).reshape(sampled.shape)
        return sampled
    if rotation:
        # every channel's quaternions slerp at the same fractions
        count = values[0].size // 4
        return quaternion_slerp(
            values[lower].reshape((-1, 4)),
            values[upper].reshape((-1, 4)),
            np.repeat(fraction, count),
        ).reshape(values[lower].shape)
    return values[lower] + (values[upper] - values[lower]) * fraction.reshape(column)


def parse_animations(header: dict, access: list, names: dict, edges: dict) -> list:
    """
    Load each glTF animation as one `Animation` of scene graph edges.

    Channels which are all "LINEAR" or all "STEP" load exactly onto
    the union of their keyframe times. A cubic spline or mixed modes
    are sampled onto a `SPLINE_RATE` timeline and blended linearly.

    Parameters
    ------------
    header : dict
      The glTF header.
    access : list
      Decoded accessor data.
    names : dict
      {node index : node name}
    edges : dict
      {node name : parent node name}

    Returns
    ----------
    animations : list
      `trimesh.scene.animation.Animation` objects.
    """
    from ...scene.animation import Animation

    nodes = header.get("nodes", [])
    result = []
    for number, animation in enumerate(header.get("animations", [])):
        name = animation.get("name", f"animation_{number}")
        # {node index : {channel : (times, values, in, out, mode)}}
        curves = defaultdict(dict)
        for channel in animation.get("channels", []):
            node = channel.get("target", {}).get("node")
            path = channel.get("target", {}).get("path")
            if path == "weights":
                log.warning("morph target `weights` animation is not supported")
            if node not in names or path not in DEFAULTS:
                continue
            try:
                sampler = animation["samplers"][channel["sampler"]]
                times = np.asarray(access[sampler["input"]], dtype=np.float64).reshape(-1)
                # (keyframe, 1 or 3, width): a cubic spline stores an
                # arriving tangent, value, and leaving tangent per keyframe
                values = np.asarray(access[sampler["output"]], dtype=np.float64).reshape(
                    (len(times), -1, len(DEFAULTS[path]))
                )
            except (KeyError, IndexError, ValueError, TypeError):
                if tol.strict:
                    raise
                log.warning(f"unable to load animation `{name}`", exc_info=True)
                continue
            if path == "rotation":
                values = values[..., [3, 0, 1, 2]]
            # the value is in the middle, which is all there is unless cubic
            middle = values.shape[1] // 2
            curves[node][path] = (
                times,
                values[:, middle],
                values[:, 0],
                values[:, -1],
                sampler.get("interpolation", "LINEAR"),
            )
        if len(curves) == 0:
            continue

        channels = [curve for node in curves.values() for curve in node.values()]
        modes = {curve[4] for curve in channels}
        grid = np.unique(np.concatenate([curve[0] for curve in channels]))
        interpolation = "linear"
        if modes == {"STEP"}:
            interpolation = "step"
        elif modes != {"LINEAR"}:
            # a spline or mixed modes are sampled densely and blended linearly
            dense = np.arange(grid[0], grid[-1], 1.0 / SPLINE_RATE)
            grid = np.union1d(grid, dense)

        # (keyframe, edge, width) of every channel of every animated node
        shape = (len(grid), len(curves))
        parts = {
            path: np.zeros((*shape, len(default))) for path, default in DEFAULTS.items()
        }
        # {(path, mode, keyframe times) : [(column, curve)]}
        groups = defaultdict(list)
        for column, (node, curve) in enumerate(curves.items()):
            # a channel the file doesn't animate holds the rest pose
            for part, rest in zip(parts.values(), trs_from_node(nodes[node])):
                part[:, column] = rest
            for path, channel in curve.items():
                groups[(path, channel[4], channel[0].tobytes())].append((column, channel))
        # channels with the same keyframes and interpolation sample at once
        for (path, mode, _), members in groups.items():
            columns = [column for column, _ in members]
            # the keyframe times every member shares
            times = members[0][1][0]
            # stacked (keyframe, channel, width) values and tangents
            stacked = [
                np.stack([channel[which] for _, channel in members], axis=1)
                for which in (1, 2, 3)
            ]
            parts[path][:, columns] = sample_channel(
                times,
                *stacked,
                mode=mode,
                query=grid,
                rotation=path == "rotation",
            )
        parts = parts.values()
        matrices = trs_matrix(*(part.reshape((-1, part.shape[-1])) for part in parts))
        result.append(
            Animation(
                edges=[(edges.get(names[node]), names[node]) for node in curves],
                times=grid,
                matrices=matrices.reshape((*shape, 4, 4)),
                name=name,
                interpolation=interpolation,
            )
        )
    return result


def append_animations(
    tree: dict,
    buffer_items: dict,
    animations: list,
    node_index: dict,
    graph,
    spline: bool = False,
) -> None:
    """
    Add animations to a glTF tree in place, with every animation of a
    name written as one glTF animation, as importers play only the first
    of several sharing a name. Every animated node is rewritten from a
    `matrix` into channels as the spec forbids animating a matrix.

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
    graph : trimesh.scene.transforms.SceneGraph
      The exported graph, for the parent of each node.
    spline : bool
      Write "CUBICSPLINE" samplers fit through the keyframes with
      `scipy.interpolate.CubicSpline` rather than "LINEAR" ones.
    """
    # imported here as this module is imported by `gltf/__init__.py`
    from . import _data_append, float32

    parents = graph.transforms.parents
    # {name : glTF animation} which every animation of the name adds to
    result = {}
    animated = set()
    for animation in animations:
        parts = dict(zip(DEFAULTS, animation.trs))
        mode = animation.interpolation.upper()
        times = None
        samplers, channels = [], []
        for column, (frame_from, frame_to) in enumerate(animation.edges):
            index = node_index.get(frame_to)
            # falsy frames like `0` or `""` are valid node names
            if frame_from is None:
                frame_from = graph.base_frame
            if index is None or parents.get(frame_to) != frame_from:
                # glTF animates a node relative to its parent node
                log.warning(f"can't export animation of `{frame_from} -> {frame_to}`")
                continue
            rest = trs_from_node(tree["nodes"][index])
            for (path, values), resting in zip(parts.items(), rest):
                values = values[:, column]
                # skip a channel which never leaves the rest pose in the
                # float32 that is written, where a quaternion and its
                # negation are the same rotation
                written = values.astype(float32)
                still = (written == resting.astype(float32)).all()
                if path == "rotation":
                    still = still or (written == -resting.astype(float32)).all()
                    values = values[:, [1, 2, 3, 0]]
                if still:
                    continue
                if times is None:
                    times = _data_append(
                        acc=tree["accessors"],
                        buff=buffer_items,
                        blob={"componentType": 5126, "type": "SCALAR"},
                        data=animation.times.astype(float32),
                    )
                interpolation = mode
                if spline and mode == "LINEAR":
                    from scipy.interpolate import CubicSpline

                    # a spline through every keyframe, written with its
                    # derivative as both the arriving and leaving tangent
                    tangent = CubicSpline(animation.times, values, axis=0)(
                        animation.times, 1
                    )
                    values = np.stack((tangent, values, tangent), axis=1).reshape(
                        (-1, values.shape[1])
                    )
                    interpolation = "CUBICSPLINE"
                samplers.append(
                    {
                        "input": times,
                        "output": _data_append(
                            acc=tree["accessors"],
                            buff=buffer_items,
                            blob={"componentType": 5126, "type": f"VEC{values.shape[1]}"},
                            data=np.ascontiguousarray(values, dtype=float32),
                        ),
                        "interpolation": interpolation,
                    }
                )
                channels.append(
                    {
                        "sampler": len(samplers) - 1,
                        "target": {"node": index, "path": path},
                    }
                )
                animated.add(index)
        if len(channels) > 0:
            merged = result.setdefault(
                animation.name,
                {"name": animation.name, "samplers": [], "channels": []},
            )
            # samplers are referenced by index within their animation
            offset = len(merged["samplers"])
            merged["samplers"].extend(samplers)
            merged["channels"].extend(
                {**channel, "sampler": channel["sampler"] + offset}
                for channel in channels
            )

    if len(result) == 0:
        return
    for index in animated:
        node = tree["nodes"][index]
        if "matrix" not in node:
            continue
        translation, quaternion, scale = trs_from_node(node)
        node.pop("matrix")
        for (path, default), value in zip(
            DEFAULTS.items(), (translation, quaternion[[1, 2, 3, 0]], scale)
        ):
            if not np.allclose(value, default):
                node[path] = value.tolist()
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
    from . import _data_append, float32

    written = {}
    for node, skin in skins.items():
        try:
            target = tree["nodes"][node_index[node]]
            joints = [node_index[joint] for joint in skin.joints]
        except KeyError:
            log.warning(f"can't export the skin of `{node}` as a node is missing")
            continue
        key = (tuple(joints), hash(skin.inverse_bind_matrices.tobytes()))
        if key not in written:
            accessor = _data_append(
                acc=tree["accessors"],
                buff=buffer_items,
                blob={"componentType": 5126, "type": "MAT4"},
                # glTF matrices are column-major
                data=np.ascontiguousarray(
                    skin.inverse_bind_matrices.transpose((0, 2, 1)), dtype=float32
                ),
            )
            skins_list = tree.setdefault("skins", [])
            skins_list.append({"joints": joints, "inverseBindMatrices": accessor})
            written[key] = len(skins_list) - 1
        target["skin"] = written[key]
