"""
animation.py
--------------

Read and write glTF animations and skins.
"""

from functools import partial
from itertools import groupby
from operator import attrgetter
from typing import NamedTuple

import numpy as np

from ...constants import log, tol
from ...scene.animation import Animation, Skin, interpolate_keyframes
from ...transformations import trs_from_matrix, trs_matrix
from ...typed import ArrayLike, NDArray, NDArray2D, NDArray3D, float64

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
    trs[:, GLTF_ORDER] = np.reshape(
        [[v for k, d in DEFAULTS.items() for v in n.get(k, d)] for n in nodes], (-1, 10)
    )
    matrices = trs_matrix(*np.split(trs, [3, 7], axis=1)).reshape((-1, 4, 4))
    # a stored matrix is column-major, exact, and may not decompose
    whole = np.array(["matrix" in node for node in nodes], dtype=bool)
    stored = np.reshape([n["matrix"] for n in nodes if "matrix" in n], (-1, 4, 4))
    matrices[whole] = stored.transpose((0, 2, 1))
    trs[whole] = np.concatenate(trs_from_matrix(matrices[whole]), axis=1)
    return matrices, trs


class Channel(NamedTuple):
    """
    One glTF channel's keyframes, with STEP held as linear.
    """

    # index of the glTF node and key of `DEFAULTS` it drives
    node: int
    path: str
    # (k,) increasing times, (k, width) values and (k, 2, width) in and out
    # tangents: authored for cubic ones, segment slopes for the rest
    keys: NDArray
    values: NDArray
    tangents: NDArray
    cubic: bool
    # the input accessor and interpolation, which channels with the same keys share
    timeline: tuple


def chords(keys: ArrayLike, values: ArrayLike) -> NDArray:
    """
    In and out tangents of keyframes joined by lines, as a cubic plays them.

    Parameters
    ------------
    keys : (k,) float
      Increasing keyframe times.
    values : (k, width) float
      Value at each keyframe.

    Returns
    ------------
    tangents : (k, 2, width) float
      Slope of the segment ending and starting at each, zero past the ends.
    """
    slope = np.diff(values, axis=0) / np.diff(keys)[:, None]
    padded = np.concatenate((np.zeros_like(values[:1]), slope, np.zeros_like(values[:1])))
    return np.stack((padded[:-1], padded[1:]), axis=1)


def hold(keys: ArrayLike, values: ArrayLike) -> tuple[NDArray, NDArray]:
    """
    STEP keyframes as linear ones which hold each value until the float32
    before the next keyframe, so they stay strictly increasing when exported.

    Parameters
    ------------
    keys : (k,) float
      Increasing keyframe times.
    values : (k, width) float
      Value at each keyframe.

    Returns
    ------------
    keys : (2k - 1,) float
      Each keyframe and the float32 before the next.
    values : (2k - 1, width) float
      Each value repeated.
    """
    before = np.nextafter(np.float32(keys[1:]), np.float32(-np.inf))
    keys = np.append(np.column_stack((keys[:-1], before)).ravel(), keys[-1:])
    return keys, np.repeat(values, 2, axis=0)[:-1]


def parse_channel(node: int, path: str, sampler: dict, access: list) -> Channel:
    """
    Keyframes of one glTF channel.

    Parameters
    ------------
    node : int
      Index of the glTF node it drives.
    path : str
      Key of `DEFAULTS` it drives.
    sampler : dict
      Its glTF animation sampler.
    access : list
      Decoded accessor data.

    Returns
    ------------
    channel : Channel
      Keyframes with STEP held as linear.
    """
    keys = np.float64(access[sampler["input"]]).reshape(-1)
    # cubic stores in-tangent, value and out-tangent per keyframe
    stored = np.float64(access[sampler["output"]])
    stored = stored.reshape((len(keys), -1, len(DEFAULTS[path])))
    values = stored[:, len(stored[0]) // 2]
    mode = sampler.get("interpolation", "LINEAR")
    if mode == "STEP":
        keys, values = hold(keys, values)
    tangents = chords(keys, values)
    if mode == "CUBICSPLINE":
        tangents = stored[:, [0, 2]]
    return Channel(
        node,
        path,
        keys,
        values,
        tangents,
        mode == "CUBICSPLINE",
        (sampler["input"], mode),
    )


def channel_tangents(
    keys: ArrayLike, values: ArrayLike, tangents: ArrayLike, times: ArrayLike
) -> NDArray:
    """
    In and out derivatives of a channel at times, from the segments
    before and after each, exactly the stored ones at keyframes.

    Parameters
    ------------
    keys : (k,) float
      Increasing keyframe times.
    values : (k, width) float
      Value at each keyframe.
    tangents : (k, 2, width) float
      In and out tangents at each keyframe.
    times : (m,) float
      Times to differentiate at.

    Returns
    ------------
    derivatives : (m, 2, width) float
      In and out derivative at each time.
    """
    if len(keys) < 2:
        return np.zeros((len(times), 2, values.shape[1]))
    # the segment ending at each time for its in derivative, starting there for out
    upper = np.clip(
        [np.searchsorted(keys, times, side=side) for side in ("left", "right")],
        1,
        len(keys) - 1,
    ).T
    lower = upper - 1
    width = (keys[upper] - keys[lower])[..., None]
    s = np.clip((times[:, None, None] - keys[lower][..., None]) / width, 0.0, 1.0)
    # the cubic Hermite derivative, with the end tangents' weights exactly 1
    return (
        (3 * s**2 - 4 * s + 1) * tangents[lower, 1]
        + (3 * s**2 - 2 * s) * tangents[upper, 0]
        + 6 * (s - s**2) * (values[upper] - values[lower]) / width
    )


def stack_channels(
    nodes: list, names: dict, name: str, channels: list, interpolation: str
) -> Animation:
    """
    One `Animation` of glTF channels on every keyframe of any of them.

    Parameters
    ------------
    nodes : list
      glTF nodes.
    names : dict
      {node index : node name}
    name : str
      Name of the animation.
    channels : list
      `Channel` of every animated node.
    interpolation : str
      `linear`, or `cubic` playing every channel's tangents.

    Returns
    ------------
    animation : trimesh.scene.animation.Animation
      Every channel stacked on one timeline.
    """
    times = np.unique(np.concatenate([channel.keys for channel in channels]))
    column = {n: i for i, n in enumerate(dict.fromkeys(c.node for c in channels))}
    # rest parts at every time, overwritten by each channel's samples
    rest = node_matrices([nodes[node] for node in column])[1]
    trs = np.tile(rest, (len(times), 1, 1))
    tangents = np.zeros((len(times), 2, *rest.shape))
    # channels of a path sharing keyframes are sampled side by side at once
    shared = attrgetter("path", "timeline")
    for (path, _), batch in groupby(sorted(channels, key=shared), key=shared):
        batch = list(batch)
        keys, rows = batch[0].keys, np.array([column[c.node] for c in batch])[:, None]
        values = np.concatenate([channel.values for channel in batch], axis=1)
        authored = np.concatenate([channel.tangents for channel in batch], axis=2)
        shape = (len(batch), len(COLUMNS[path]))
        # linear slerps rotations and cubic plays every channel's tangents
        sampled = interpolate_keyframes(
            keys,
            values,
            times,
            rotation=path == "rotation",
            tangents={"cubic": authored}.get(interpolation),
        )
        trs[:, rows, COLUMNS[path]] = sampled.reshape((len(times), *shape))
        derivatives = channel_tangents(keys, values, authored, times)
        tangents[:, :, rows, COLUMNS[path]] = derivatives.reshape((-1, 2, *shape))
        # keyframes exactly as stored, which slerp would unitize
        stored = np.searchsorted(times, keys)[:, None, None]
        trs[stored, rows, COLUMNS[path]] = values.reshape((len(keys), *shape))
    return Animation(
        name=name,
        nodes=[names[node] for node in column],
        times=times,
        interpolation=interpolation,
        trs=trs,
        tangents=tangents,
    )


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
    try:
        skins = {index: header["skins"][index] for index in set(skinned.values())}
        # glTF matrices are column-major, and a skin without them binds at rest
        binds = {
            index: np.reshape(access[skin["inverseBindMatrices"]], (-1, 4, 4))
            for index, skin in skins.items()
            if "inverseBindMatrices" in skin
        }
        loaded = {
            index: Skin(
                [names[joint] for joint in skin["joints"]],
                binds.get(index, np.eye(4)[None]).transpose((0, 2, 1)),
            )
            for index, skin in skins.items()
        }
    except (KeyError, IndexError, ValueError, TypeError):
        if tol.strict:
            raise
        log.warning("unable to load skins", exc_info=True)
        return {}
    return {node: loaded[index] for node, index in skinned.items()}


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
        try:
            # {(node, path) : sampler} as the spec targets each once
            samplers = animation["samplers"]
            targets = {
                (c["target"].get("node"), c["target"]["path"]): samplers[c["sampler"]]
                for c in animation.get("channels", [])
            }
            channels = [
                parse_channel(node, path, sampler, access)
                for (node, path), sampler in targets.items()
                if node in names and path in DEFAULTS
            ]
        except (KeyError, IndexError, TypeError, ValueError):
            if tol.strict:
                raise
            log.warning(f"unable to load animation `{number}`", exc_info=True)
            continue
        if any(path == "weights" for _, path in targets):
            log.warning("morph target `weights` animation is not supported")
        # a node is cubic if any of its channels is, and linear otherwise
        cubic = {channel.node for channel in channels if channel.cubic}
        groups = {
            "linear": [channel for channel in channels if channel.node not in cubic],
            "cubic": [channel for channel in channels if channel.node in cubic],
        }
        name = animation.get("name", f"animation_{number}")
        result.extend(
            stack_channels(header["nodes"], names, name, group, interpolation)
            for interpolation, group in groups.items()
            if len(group) > 0
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
    missing = [a.name for a in scene.animations if not node_index.keys() >= set(a.nodes)]
    if len(missing) > 0:
        if tol.strict:
            raise ValueError(f"{missing} animate missing nodes")
        log.warning(f"can't export the missing nodes {missing} animate")
    # {name : glTF animation}, merged as importers play only the first
    result = {}
    for animation in scene.animations:
        present = [i for i, node in enumerate(animation.nodes) if node in node_index]
        indexes = [node_index[animation.nodes[i]] for i in present]
        nodes = [tree["nodes"][index] for index in indexes]
        rest = np.float32(node_matrices(nodes)[1])
        # animated nodes are stored as parts
        for node, row in zip(nodes, rest[:, GLTF_ORDER]):
            node.pop("matrix", None)
            parts = zip(DEFAULTS.items(), np.split(row, [3, 7]))
            node.update({p: v.tolist() for (p, d), v in parts if (v != d).any()})
        # a part moves unless every keyframe stores its rest, where `q` is `-q`
        trs = np.float32(animation.trs[:, present])
        moves = (trs != rest).any(axis=0)
        moves[:, 3:7] &= (trs[..., 3:7] != -rest[:, 3:7]).any(axis=0)
        keyframes = trs
        if animation.interpolation == "cubic":
            # in-tangent, value and out-tangent per keyframe
            tangents = np.float32(animation.tangents[:, :, present])
            keyframes = np.stack((tangents[:, 0], trs, tangents[:, 1]), axis=1)
            keyframes = keyframes.reshape((-1, *trs.shape[1:]))
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

    # skins whose node and joints are all exported
    placed = {
        n: skin
        for n, skin in scene.skins.items()
        if node_index.keys() >= {n, *skin.joints}
    }
    if len(placed) < len(scene.skins):
        message = f"can't export skins of {scene.skins.keys() - placed.keys()}"
        if tol.strict:
            raise ValueError(message)
        log.warning(message)
    # {(joints, inverse bind accessor) : index} writes each distinct skin once
    skins = {}
    for node, skin in placed.items():
        # glTF matrices are column-major
        inverse = append(
            {"componentType": 5126, "type": "MAT4"},
            np.float32(skin.inverse_bind_matrices.transpose((0, 2, 1))),
        )
        key = (tuple(node_index[joint] for joint in skin.joints), inverse)
        tree["nodes"][node_index[node]]["skin"] = skins.setdefault(key, len(skins))
    if len(skins) > 0:
        tree["skins"] = [{"joints": list(j), "inverseBindMatrices": i} for j, i in skins]
