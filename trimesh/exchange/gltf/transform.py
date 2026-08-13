"""
transform.py
--------------

Pack the translation-rotation-scale form GLTF stores node transforms
in, on top of `trimesh.transformations.tqs_from_matrix`.

This module speaks GLTF's conventions natively: quaternions ordered
`XYZW` and matrices flattened column-major. Everything is vectorized,
and TRS is packed into a single `(n, 10)` array rather than three, as
that is the shape a GLTF node dict and animation sampler want.
"""

import numpy as np

from ...transformations import tqs_from_matrix, tqs_matrix
from ...typed import NDArray1D, NDArray2D, NDArray3D, float64

# column layout of a packed TRS array
TRANSLATION = slice(0, 3)
# note GLTF stores quaternions XYZW and trimesh stores them WXYZ
ROTATION = slice(3, 7)
SCALE = slice(7, 10)

# how many values a packed TRS holds: 3 + 4 + 3
TRS = 10

# a TRS which leaves a transform unchanged
_IDENTITY = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
_IDENTITY.flags.writeable = False

# the animation channel paths which map onto a packed TRS
PATHS = ("translation", "rotation", "scale")

# reorder a quaternion between the two conventions
_TO_GLTF = [1, 2, 3, 0]
_FROM_GLTF = [3, 0, 1, 2]


def trs_from_matrix(matrices) -> NDArray2D[float64]:
    """
    Decompose homogeneous transforms into packed TRS.

    A matrix can't be interpolated or stored by GLTF directly, and
    has to be split into a translation, a rotation, and a scale.

    Parameters
    ------------
    matrices : (4, 4) or (n, 4, 4) float
      Homogeneous transformation matrices.

    Returns
    ----------
    trs : (n, 10) float
      Translation, `XYZW` quaternion, and scale for each matrix.
    """
    # note this must not mutate the input as callers pass read-only arrays
    matrices = np.asanyarray(matrices, dtype=np.float64).reshape((-1, 4, 4))

    translation, quaternion, scale = tqs_from_matrix(matrices)

    trs = np.zeros((len(matrices), TRS))
    trs[:, TRANSLATION] = translation
    trs[:, ROTATION] = quaternion.reshape((-1, 4))[:, _TO_GLTF]
    trs[:, SCALE] = scale

    return trs


def matrix_from_trs(trs) -> NDArray3D[float64]:
    """
    Compose homogeneous transforms from packed TRS.

    The inverse of `trs_from_matrix`, applying in GLTF's order
    of translation, then rotation, then scale.

    Parameters
    ------------
    trs : (10,) or (n, 10) float
      Translation, `XYZW` quaternion, and scale.

    Returns
    ----------
    matrices : (n, 4, 4) float
      Homogeneous transformation matrix for each row.
    """
    trs = np.asanyarray(trs, dtype=np.float64).reshape((-1, TRS))

    return tqs_matrix(
        trs[:, TRANSLATION], trs[:, ROTATION][:, _FROM_GLTF], trs[:, SCALE]
    ).reshape((-1, 4, 4))


def trs_from_node(node: dict) -> NDArray1D[float64]:
    """
    Collect the TRS keys of a GLTF node, filling in any defaults.

    Parameters
    ------------
    node : dict
      A GLTF node which may have `translation`, `rotation`, or `scale`.

    Returns
    ----------
    trs : (10,) float
      Translation, `XYZW` quaternion, and scale.
    """
    trs = _IDENTITY.copy()
    for key, index in zip(PATHS, (TRANSLATION, ROTATION, SCALE)):
        if key in node:
            trs[index] = np.reshape(node[key], -1)
    return trs


def node_from_trs(trs, node: dict) -> None:
    """
    Store a packed TRS on a GLTF node, mutating it in-place.

    Any component which is already the GLTF default is omitted
    rather than being written out.

    Parameters
    ------------
    trs : (10,) float
      Translation, `XYZW` quaternion, and scale.
    node : dict
      GLTF node to store the values on.
    """
    trs = np.asanyarray(trs, dtype=np.float64).reshape(TRS)
    for key, index in zip(PATHS, (TRANSLATION, ROTATION, SCALE)):
        if not np.allclose(trs[index], _IDENTITY[index], atol=1e-12):
            node[key] = trs[index].tolist()


def quaternion_to_gltf(quaternion) -> NDArray2D[float64]:
    """
    Reorder quaternions from trimesh's `WXYZ` into GLTF's `XYZW`.

    Parameters
    ------------
    quaternion : (n, 4) float
      Quaternions ordered `WXYZ`.

    Returns
    ----------
    reordered : (n, 4) float
      The same rotations ordered `XYZW`.
    """
    return np.asanyarray(quaternion, dtype=np.float64).reshape((-1, 4))[:, _TO_GLTF]


def quaternion_from_gltf(quaternion) -> NDArray2D[float64]:
    """
    Reorder quaternions from GLTF's `XYZW` into trimesh's `WXYZ`.

    Parameters
    ------------
    quaternion : (n, 4) float
      Quaternions ordered `XYZW`.

    Returns
    ----------
    reordered : (n, 4) float
      The same rotations ordered `WXYZ`.
    """
    return np.asanyarray(quaternion, dtype=np.float64).reshape((-1, 4))[:, _FROM_GLTF]


def matrix_from_gltf(values) -> NDArray2D[float64]:
    """
    Unflatten a GLTF matrix, which is stored column-major.

    Parameters
    ------------
    values : (16,) float
      A GLTF `matrix` value.

    Returns
    ----------
    matrix : (4, 4) float
      Homogeneous transformation matrix.
    """
    return np.array(values, dtype=np.float64).reshape((4, 4)).T


def matrix_to_gltf(matrix) -> list:
    """
    Flatten a matrix into the column-major order GLTF stores.

    Parameters
    ------------
    matrix : (4, 4) float
      Homogeneous transformation matrix.

    Returns
    ----------
    values : list
      16 floats which can be stored as a GLTF `matrix`.
    """
    return np.asanyarray(matrix, dtype=np.float64).T.reshape(-1).tolist()


def unwind(quaternion) -> NDArray2D[float64]:
    """
    Flip quaternion signs so adjacent keyframes share a hemisphere.

    A quaternion and its negation are the same rotation, but a viewer
    interpolating linearly between two which are in opposite hemispheres
    will take the long way around and the motion will visibly jerk.

    Parameters
    ------------
    quaternion : (n, 4) float
      Quaternions, in any consistent ordering.

    Returns
    ----------
    unwound : (n, 4) float
      The same rotations with adjacent pairs in the same hemisphere.
    """
    quaternion = np.asanyarray(quaternion, dtype=np.float64).reshape((-1, 4))

    # the sign flip needed to keep each pair within 90 degrees, accumulated
    # so a flip carries through every keyframe which follows it
    signs = np.ones(len(quaternion))
    signs[1:] = np.cumprod(
        np.where(np.sum(quaternion[1:] * quaternion[:-1], axis=1) < 0, -1.0, 1.0)
    )

    return quaternion * signs.reshape((-1, 1))
