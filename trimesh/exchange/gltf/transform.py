"""
transform.py
--------------

Translate between GLTF's conventions and trimesh's.

GLTF orders quaternions `XYZW` where trimesh orders them `WXYZ`, and
flattens matrices column-major. A node transform is stored as a
translation, rotation, and scale, which is the `(t, q, s)` tuple
`trimesh.transformations.tqs_from_matrix` already returns.
"""

import numpy as np

from ...transformations import tqs_from_matrix
from ...typed import NDArray1D, NDArray2D, float64

# the animation channel paths, in the order a TRS tuple holds them
PATHS = ("translation", "rotation", "scale")

# reorder a quaternion between the two conventions
_TO_GLTF = [1, 2, 3, 0]
_FROM_GLTF = [3, 0, 1, 2]

# a `WXYZ` quaternion which doesn't rotate
_NO_ROTATION = [1.0, 0.0, 0.0, 0.0]


def quaternion_to_gltf(quaternion) -> NDArray2D[float64]:
    """Reorder quaternions from trimesh's `WXYZ` into GLTF's `XYZW`."""
    return np.asanyarray(quaternion, dtype=np.float64).reshape((-1, 4))[:, _TO_GLTF]


def quaternion_from_gltf(quaternion) -> NDArray2D[float64]:
    """Reorder quaternions from GLTF's `XYZW` into trimesh's `WXYZ`."""
    return np.asanyarray(quaternion, dtype=np.float64).reshape((-1, 4))[:, _FROM_GLTF]


def matrix_from_gltf(values) -> NDArray2D[float64]:
    """Unflatten a (16,) GLTF matrix, which is stored column-major."""
    return np.array(values, dtype=np.float64).reshape((4, 4)).T


def trs_from_node(node: dict) -> tuple:
    """Collect a node's TRS keys as `(translation, WXYZ quaternion, scale)`,
    filling in the GLTF default for any which are absent."""
    return (
        np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64),
        quaternion_from_gltf(node.get("rotation", [0.0, 0.0, 0.0, 1.0]))[0],
        np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64),
    )


def node_from_trs(trs, node: dict) -> None:
    """Store a `(translation, WXYZ quaternion, scale)` on a node in-place,
    omitting any component which is already the GLTF default."""
    translation, quaternion, scale = trs

    if not np.allclose(translation, 0.0, atol=1e-12):
        node["translation"] = np.asanyarray(translation, dtype=np.float64).tolist()
    if not np.allclose(quaternion, _NO_ROTATION, atol=1e-12):
        node["rotation"] = quaternion_to_gltf(quaternion)[0].tolist()
    if not np.allclose(scale, 1.0, atol=1e-12):
        node["scale"] = np.asanyarray(scale, dtype=np.float64).tolist()


def trs_from_gltf_matrices(values) -> tuple:
    """Decompose `(n, 16)` column-major GLTF matrices into stacked
    translations, `WXYZ` quaternions, and scales."""
    flat = np.array(values, dtype=np.float64).reshape((-1, 4, 4))
    # a GLTF matrix is column-major so transpose each one
    return tqs_from_matrix(flat.transpose(0, 2, 1))


def unwind(quaternion) -> NDArray1D[float64]:
    """
    Flip signs so adjacent `(n, 4)` quaternions share a hemisphere.

    A quaternion and its negation are the same rotation, but a viewer
    interpolating linearly between two which are in opposite hemispheres
    will take the long way around and the motion will visibly jerk.
    """
    quaternion = np.asanyarray(quaternion, dtype=np.float64).reshape((-1, 4))

    # the sign flip needed to keep each pair within 90 degrees, accumulated
    # so a flip carries through every keyframe which follows it
    signs = np.ones(len(quaternion))
    signs[1:] = np.cumprod(
        np.where(np.sum(quaternion[1:] * quaternion[:-1], axis=1) < 0, -1.0, 1.0)
    )

    return quaternion * signs.reshape((-1, 1))
