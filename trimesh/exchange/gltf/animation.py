"""
animation.py
--------------

Read and write GLTF keyframed animation.

GLTF stores an animation as samplers holding a time accessor and a value
accessor, and channels pointing each sampler at one node and one of the
`translation`, `rotation`, or `scale` paths. Trimesh stores the same
motion as a `RigidAnimation` driving one *edge* of the scene graph, so
this is where a node target and its implied parent become an edge.

GLTF also orders quaternions `XYZW` where trimesh orders them `WXYZ`,
and flattens matrices column-major.
"""

from collections import OrderedDict, defaultdict

import numpy as np

from ...constants import log
from ...transformations import tqs_from_matrix

# the node transform keys GLTF animates, in the order a TQS tuple holds them
_PATHS = ("translation", "rotation", "scale")
# and the matching field on a trimesh keyframe
_FIELDS = ("translation", "quaternion", "scale")

# trimesh interpolation mode : GLTF sampler interpolation, and the reverse
_INTERPOLATION = {"linear": "LINEAR", "step": "STEP", "cubic": "CUBICSPLINE"}
_INTERPOLATION_LOAD = {v: k for k, v in _INTERPOLATION.items()}

# GLTF orders quaternions `XYZW` where trimesh orders them `WXYZ`
_Q_TO_GLTF = [1, 2, 3, 0]
_Q_FROM_GLTF = [3, 0, 1, 2]

# the GLTF default for each of the TRS keys, in `_PATHS` order
_TRS_DEFAULT = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0])


def trs_from_node(node: dict) -> tuple:
    """
    Read a node's TRS keys as `(translation, WXYZ quaternion, scale)`,
    filling in the GLTF default for any which are absent.
    """
    translation, quaternion, scale = (
        np.array(node.get(path, default), dtype=np.float64)
        for path, default in zip(_PATHS, _TRS_DEFAULT)
    )
    # a node stores `XYZW` but only if it stored a rotation at all
    if "rotation" in node:
        quaternion = quaternion[_Q_FROM_GLTF]
    return translation, quaternion, scale


def node_from_trs(trs, node: dict) -> None:
    """
    Store a `(translation, WXYZ quaternion, scale)` on a node in-place,
    omitting any component which is already the GLTF default.
    """
    for path, default, value in zip(_PATHS, _TRS_DEFAULT, trs):
        value = np.asanyarray(value, dtype=np.float64)
        if np.allclose(value, default, atol=1e-12):
            continue
        node[path] = (value[_Q_TO_GLTF] if path == "rotation" else value).tolist()


def append_animations(tree, buffer_items, animations, node_index):
    """
    Append keyframed animations to a GLTF tree, mutating it in-place.

    Animations which share a `name` are combined into a single GLTF
    animation. Any node targeted by a channel is rewritten from a
    `matrix` into TRS, as the spec forbids a matrix on animated nodes.

    Parameters
    ------------
    tree : dict
      GLTF header, will be mutated in-place.
    buffer_items
      Collection of buffer bytes, will be mutated in-place.
    animations : list
      Sequence of `trimesh.scene.animation.RigidAnimation`.
    node_index : dict
      Mapping of {node name : index in `tree["nodes"]`}
    """
    # imported here rather than at module level as `gltf/__init__.py`
    # imports this module, so at import time it isn't finished yet
    from . import _data_append, float32

    grouped = OrderedDict()
    for animation in animations:
        # animations sharing a name become one GLTF animation
        grouped.setdefault(animation.name, []).append(animation)

    result = []
    # nodes which need rewriting from `matrix` into TRS
    animated = set()

    for name, group in grouped.items():
        samplers = []
        channels = []

        for animation in group:
            index = node_index.get(animation.frame_to)
            if index is None:
                log.warning(f"animation targets missing node `{animation.frame_to}`!")
                continue

            keyframes = animation.keyframes
            cubic = animation.interpolation == "cubic"
            node = tree["nodes"][index]
            # what the node looks like where a channel isn't animated, in
            # trimesh's `wxyz` so it compares against the keyframes directly
            static = (
                tqs_from_matrix(np.reshape(node["matrix"], (4, 4)).T)
                if "matrix" in node
                else trs_from_node(node)
            )
            # the time accessor is shared by every channel of this node but is
            # only written once a channel survives the static check below, as
            # an accessor nothing references would be stranded in the file
            sampler_input = None

            for path, field, rest in zip(_PATHS, _FIELDS, static):
                values = keyframes[field]
                tangents = (keyframes[field + "_in"], keyframes[field + "_out"])

                # a channel which never moves and already matches the node's
                # static pose doesn't need storing. note a quaternion and its
                # negation are the same rotation
                flat = len(values) == 1 or np.ptp(values, axis=0).max() < 1e-12
                if cubic:
                    # tangents bend a curve which has identical endpoints
                    flat = flat and max(np.abs(t).max() for t in tangents) < 1e-12
                if flat and (
                    np.allclose(values[0], rest)
                    or (path == "rotation" and np.allclose(values[0], -rest))
                ):
                    continue

                if path == "rotation":
                    # trimesh stores `wxyz` and GLTF wants `xyzw`
                    values = values[:, _Q_TO_GLTF]
                    tangents = tuple(t[:, _Q_TO_GLTF] for t in tangents)
                    if not cubic:
                        # keep adjacent keyframes in the same hemisphere or a
                        # viewer interpolates the long way and visibly jerks.
                        # a spline is left alone as flipping a keyframe without
                        # flipping its tangents would break the curve
                        signs = np.cumprod(
                            np.where(
                                np.sum(values[1:] * values[:-1], axis=1) < 0, -1.0, 1.0
                            )
                        )
                        values = values.copy()
                        values[1:] *= signs.reshape((-1, 1))

                if sampler_input is None:
                    # the first channel to survive pays for the time accessor
                    # note `_data_append` fills in the min/max the spec requires
                    sampler_input = _data_append(
                        acc=tree["accessors"],
                        buff=buffer_items,
                        blob={"componentType": 5126, "type": "SCALAR"},
                        data=animation.times.astype(float32),
                    )

                if cubic:
                    # the spec interleaves each keyframe as in-tangent, value,
                    # and out-tangent so the accessor is three times as long
                    values = np.stack([tangents[0], values, tangents[1]], axis=1).reshape(
                        (-1, values.shape[1])
                    )

                samplers.append(
                    {
                        "input": sampler_input,
                        "output": _data_append(
                            acc=tree["accessors"],
                            buff=buffer_items,
                            blob={
                                "componentType": 5126,
                                "type": "VEC4" if path == "rotation" else "VEC3",
                            },
                            data=np.ascontiguousarray(values, dtype=float32),
                        ),
                        "interpolation": _INTERPOLATION[animation.interpolation],
                    }
                )
                channels.append(
                    {
                        "sampler": len(samplers) - 1,
                        "target": {"node": index, "path": path},
                    }
                )
                # only a node which is actually targeted has to be rewritten
                animated.add(index)

        if len(channels) > 0:
            result.append({"name": name, "samplers": samplers, "channels": channels})

    if len(result) == 0:
        return

    # the spec forbids a `matrix` on any node targeted by an animation so
    # replace it with the equivalent TRS, decomposing every animated node in
    # one batched call rather than an eigendecomposition each
    ordered = [i for i in sorted(animated) if "matrix" in tree["nodes"][i]]
    if len(ordered) > 0:
        flat = np.reshape([tree["nodes"][i].pop("matrix") for i in ordered], (-1, 4, 4))
        # a GLTF matrix is column-major so transpose each one
        trs = tqs_from_matrix(flat.transpose(0, 2, 1))
        for j, index in enumerate(ordered):
            # `node_from_trs` omits any component already at the GLTF default
            node_from_trs([part[j] for part in trs], tree["nodes"][index])

    tree["animations"] = result


def parse_animations(header, access, names, edges):
    """
    Convert GLTF animations into trimesh animation objects.

    A GLTF animation holds channels targeting multiple nodes, which is
    flattened here into one `RigidAnimation` per node sharing a name.
    GLTF targets a node and leaves the parent implied by its node tree
    where trimesh drives an edge, so this is where that node target is
    resolved into the `frame_from -> frame_to` pair it actually means.

    Parameters
    ------------
    header : dict
      GLTF header.
    access : list
      Decoded accessor data.
    names : dict
      Mapping of {node index : node name}
    edges : dict
      Mapping of {node name : parent node name}

    Returns
    ----------
    animations : list
      Sequence of `RigidAnimation` objects.
    """
    # imported here rather than at module level to keep the dependency
    # one-way: `trimesh.scene.scene` imports this package to export, so a
    # module-level import back into `trimesh.scene` closes a cycle which
    # only survives today because of the order the modules happen to load
    from ...scene.animation import KEYFRAME, RigidAnimation, keyframes_from_matrix

    result = []
    nodes = header.get("nodes", [])

    for index, current in enumerate(header.get("animations", [])):
        name = current.get("name", f"animation_{index}")
        samplers = current.get("samplers", [])
        # {node index : {path : (times, values, in, out, mode)}}
        collected = defaultdict(dict)

        for channel in current.get("channels", []):
            target = channel.get("target", {})
            node, path = target.get("node"), target.get("path")
            if node not in names or path not in _PATHS:
                if path == "weights":
                    log.warning("morph target `weights` animation is not supported!")
                continue

            try:
                sampler = samplers[channel["sampler"]]
                times = np.asanyarray(access[sampler["input"]], dtype=np.float64).reshape(
                    -1
                )
                values = np.asanyarray(access[sampler["output"]], dtype=np.float64)
                values = values.reshape((len(values), -1))
            except BaseException:
                log.warning("unable to load animation sampler!", exc_info=True)
                continue

            stored = sampler.get("interpolation", "LINEAR")
            mode = _INTERPOLATION_LOAD.get(stored)
            if mode is None:
                log.warning(f"unsupported interpolation `{stored}`, using LINEAR")
                mode = "linear"

            if mode == "cubic":
                # stored interleaved as in-tangent, value, out-tangent
                if len(values) != len(times) * 3:
                    log.warning(f"animation `{name}` bad CUBICSPLINE length, skipping")
                    continue
                split = values.reshape((len(times), 3, -1))
                incoming, values, outgoing = split[:, 0], split[:, 1], split[:, 2]
            else:
                incoming = outgoing = np.zeros_like(values)

            if len(times) != len(values):
                log.warning(f"animation `{name}` sampler length mismatch, skipping")
                continue

            if path == "rotation":
                # GLTF stores `xyzw` and trimesh stores `wxyz`
                values, incoming, outgoing = (
                    v[:, _Q_FROM_GLTF] for v in (values, incoming, outgoing)
                )

            collected[node][path] = (times, values, incoming, outgoing, mode)

        for node, sampled in collected.items():
            # every channel of this node has to land on a shared time base,
            # which in the common case they already reference
            bases = [c[0] for c in sampled.values()]
            shared = all(np.array_equal(b, bases[0]) for b in bases[1:])
            times = bases[0] if shared else np.unique(np.concatenate(bases))

            modes = {c[4] for c in sampled.values()}
            # channels have to agree on how to blend, as marking a mixed
            # animation STEP would make its smooth channels blocky and marking
            # it cubic would need tangents the others lack. a resampled spline
            # is still followed along its curve but lands on keyframes which
            # have no tangents of their own, so it can't stay a spline either
            interpolation = modes.pop() if len(modes) == 1 and shared else "linear"

            # what this node looks like where a channel isn't animated, which
            # holds that static value across the whole timeline
            static = nodes[node]
            keyframes = np.zeros(len(times), dtype=KEYFRAME)
            keyframes["time"] = times
            (
                keyframes["translation"],
                keyframes["quaternion"],
                keyframes["scale"],
            ) = (
                tqs_from_matrix(np.reshape(static["matrix"], (4, 4)).T)
                if "matrix" in static
                else trs_from_node(static)
            )

            for path, field in zip(_PATHS, _FIELDS):
                if path not in sampled:
                    continue
                channel_times, values, incoming, outgoing, mode = sampled[path]

                if not shared:
                    # land this channel on the shared base by sampling its own
                    # curve, so a spline is followed along its real shape
                    # before being flattened onto the new keyframe times
                    alone = keyframes_from_matrix(channel_times)
                    alone[field] = values
                    alone[field + "_in"] = incoming
                    alone[field + "_out"] = outgoing
                    values = tqs_from_matrix(
                        RigidAnimation(
                            frame_to=path, keyframes=alone, interpolation=mode
                        ).at(times)
                    )[_FIELDS.index(field)]

                keyframes[field] = values
                # tangents are meaningless outside a cubic spline, and this
                # only reaches cubic when every channel shared a time base
                if interpolation == "cubic":
                    keyframes[field + "_in"] = incoming
                    keyframes[field + "_out"] = outgoing

            result.append(
                RigidAnimation(
                    frame_to=names[node],
                    frame_from=edges.get(names[node]),
                    keyframes=keyframes,
                    name=name,
                    interpolation=interpolation,
                )
            )

    return result
