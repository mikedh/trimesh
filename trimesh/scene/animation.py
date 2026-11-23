"""
animation.py
--------------

Keyframed animation of scene graph nodes and mesh skins.
"""

import numpy as np

from .. import caching
from ..constants import tol
from ..path.curve import evaluate_bezier
from ..transformations import quaternion_slerp, trs_from_matrix, trs_matrix
from ..typed import (
    ArrayLike,
    Floating,
    Iterable,
    NDArray,
    NDArray1D,
    NDArray2D,
    NDArray3D,
    NDArray4D,
    float64,
)

# how to sample between keyframes
INTERPOLATIONS = ("step", "linear", "cubic")


def interpolate_keyframes(
    times: ArrayLike,
    values: ArrayLike,
    query: Floating | ArrayLike,
    step: bool = False,
    rotation: bool = False,
    tangents: ArrayLike | None = None,
) -> NDArray:
    """
    Sample keyframed values, clamped outside the keyframe range.

    Parameters
    ------------
    times : (k,) float
      Increasing keyframe times in seconds.
    values : (k, ..., d) float
      Value at each keyframe, `wxyz` quaternions if `rotation`.
    query : float or (m,) float
      Times to sample at.
    step : bool
      Hold each keyframe instead of blending.
    rotation : bool
      Slerp quaternions.
    tangents : None or (k, 2, ..., d) float
      In and out derivatives at each keyframe to blend as cubic Hermite.

    Returns
    ------------
    sampled : (..., d) or (m, ..., d) float
      Value at each query time.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    upper = np.clip(np.searchsorted(times, query), 1, len(times) - 1)
    lower = np.maximum(upper - 1, 0)
    width = times[upper] - times[lower]
    # a single keyframe has zero width
    fraction = np.clip(
        np.divide(query - times[lower], width, out=np.zeros_like(query), where=width > 0),
        0.0,
        1.0,
    )
    # broadcast against the values of one keyframe
    stretch = (1,) * (values.ndim - 1)
    if tangents is not None:
        # a cubic Hermite segment is the Bezier curve of its values and a
        # third of each tangent across it
        tangents = np.asarray(tangents, dtype=np.float64)
        reach = (width / 3.0).reshape(width.shape + stretch)
        controls = (
            values[lower],
            values[lower] + reach * tangents[lower, 1],
            values[upper] - reach * tangents[upper, 0],
            values[upper],
        )
        return evaluate_bezier(
            np.stack(controls, axis=-2), fraction.reshape(width.shape + stretch[1:])
        )
    if step:
        return values[np.where(fraction >= 1.0, upper, lower)]
    fraction = fraction.reshape(fraction.shape + stretch)
    if rotation:
        blended = quaternion_slerp(
            values[lower].reshape((-1, 4)),
            values[upper].reshape((-1, 4)),
            np.repeat(fraction, values[0].size // 4),
        ).reshape(values[lower].shape)
        return blended
    return values[lower] + (values[upper] - values[lower]) * fraction


def monotone_tangents(times: ArrayLike, values: ArrayLike) -> NDArray:
    """
    Monotone cubic derivatives of keyframed values with respect to time.

    Parameters
    ------------
    times : (k,) float
      Increasing keyframe times in seconds.
    values : (k, ..., d) float
      Value at each keyframe.

    Returns
    ------------
    tangents : (k, 2, ..., d) float
      In and out derivatives, the harmonic mean of the slopes on either
      side and zero at extrema (Fritsch-Butland 1984), so a held or linear
      part stays held or linear and nothing overshoots.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return np.zeros((len(values), 2, *values.shape[1:]))
    change = np.diff(values, axis=0)
    width = np.diff(times).reshape((-1,) + (1,) * (values.ndim - 1))
    # repeated keyframes have no slope
    slope = np.divide(change, width, out=np.zeros_like(change), where=width > 0)
    before, after = slope[:-1], slope[1:]
    middle = np.divide(
        2.0 * before * after,
        before + after,
        out=np.zeros_like(before),
        where=before * after > 0.0,
    )
    tangents = np.concatenate((slope[:1], middle, slope[-1:]))
    return np.stack((tangents, tangents), axis=1)


class Animation:
    """
    Keyframed translation, rotation and scale of scene graph nodes on one
    timeline.
    """

    def __init__(
        self,
        name: str,
        nodes: Iterable,
        times: ArrayLike,
        matrices: ArrayLike | None = None,
        interpolation: str = "linear",
        trs: ArrayLike | None = None,
        tangents: ArrayLike | None = None,
    ):
        """
        Create an animation of scene graph nodes.

        Parameters
        ------------
        name : str
          Name of the animation.
        nodes : (n,) sequence of hashable
          Scene graph nodes to animate.
        times : (k,) float
          Increasing keyframe times in seconds.
        matrices : None or (k, n, 4, 4) float
          Parent-to-node transforms, or (k, 4, 4) for one node, stored as
          `trs` with neighboring quaternions in one hemisphere.
        interpolation : str
          One of `INTERPOLATIONS`.
        trs : None or (k, n, 10) float
          Translation, `wxyz` quaternion and scale instead of `matrices`.
        tangents : None or (k, 2, n, 10) float
          In and out derivatives of `trs` for cubic interpolation, or
          `monotone_tangents` of it.
        """
        if interpolation not in INTERPOLATIONS:
            raise ValueError(f"interpolation must be one of {INTERPOLATIONS}!")
        self.interpolation = interpolation
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if len(times) == 0 or (np.diff(times) < 0.0).any():
            raise ValueError("times must be increasing and not empty!")
        self.name = name
        self.nodes = list(nodes)
        shape = (len(times), len(self.nodes))
        if trs is None:
            matrices = np.asarray(matrices, dtype=np.float64).reshape((*shape, 4, 4))
            trs = np.concatenate(trs_from_matrix(matrices), axis=2)
            # flip quaternions so neighbors blend the short way around
            flips = (trs[1:, :, 3:7] * trs[:-1, :, 3:7]).sum(axis=2) < 0.0
            trs[1:, :, 3:7] *= np.cumprod(np.where(flips, -1.0, 1.0), axis=0)[..., None]
            if tol.strict and not np.allclose(
                trs_matrix(*np.split(trs, [3, 7], 2)), matrices
            ):
                raise ValueError("matrices must be translation, rotation and scale!")
        trs = np.asarray(trs, dtype=np.float64).reshape((*shape, 10))
        if tangents is None:
            tangents = monotone_tangents(times, trs)
        self._data = caching.DataStore()
        self._cache = caching.Cache(id_function=self._data.__hash__)
        self._data["times"] = times
        self._data["trs"] = trs
        self._data["tangents"] = np.asarray(tangents, dtype=np.float64).reshape(
            (shape[0], 2, *shape[1:], 10)
        )

    @property
    def times(self) -> NDArray1D[float64]:
        """
        Keyframe times.

        Returns
        ------------
        times : (k,) float
          Increasing keyframe times in seconds.
        """
        return self._data["times"]

    @property
    def trs(self) -> NDArray3D[float64]:
        """
        Every node at every keyframe.

        Returns
        ------------
        trs : (k, n, 10) float
          Translation, `wxyz` quaternion and scale.
        """
        return self._data["trs"]

    @property
    def tangents(self) -> NDArray4D[float64]:
        """
        Derivatives of `trs` with respect to time for cubic interpolation.

        Returns
        ------------
        tangents : (k, 2, n, 10) float
          In and out derivative of every node at every keyframe.
        """
        return self._data["tangents"]

    @caching.cache_decorator
    def matrices(self) -> NDArray4D[float64]:
        """
        Transform of every node at every keyframe.

        Returns
        ------------
        matrices : (k, n, 4, 4) float
          Parent-to-node transforms, edited through `trs`.
        """
        matrices = trs_matrix(*np.split(self.trs, [3, 7], axis=2))
        matrices.flags.writeable = False
        return matrices

    @property
    def duration(self) -> float:
        """
        Time of the last keyframe.

        Returns
        ------------
        duration : float
          Seconds.
        """
        return float(self.times[-1])

    def interpolate(self, time: Floating) -> NDArray3D[float64]:
        """
        Transform of every node at a time.

        Parameters
        ------------
        time : float
          Seconds.

        Returns
        ------------
        matrices : (n, 4, 4) float
          Parent-to-node transforms in `nodes` order.
        """
        times, trs = self.times, self.trs
        if self.interpolation == "cubic":
            sampled = interpolate_keyframes(times, trs, time, tangents=self.tangents)
        else:
            step = self.interpolation == "step"
            sampled = interpolate_keyframes(times, trs, time, step)
            sampled[:, 3:7] = interpolate_keyframes(
                times, trs[..., 3:7], time, step, rotation=True
            )
        # `trs_matrix` normalizes quaternions as glTF specifies
        return trs_matrix(sampled[:, :3], sampled[:, 3:7], sampled[:, 7:])


class Skin:
    """
    Joints which deform a mesh by linear blend skinning.
    """

    def __init__(self, joints: Iterable, inverse_bind_matrices: ArrayLike | None = None):
        """
        Create a skin.

        Parameters
        ------------
        joints : (j,) sequence of hashable
          Scene graph nodes referenced by joint index.
        inverse_bind_matrices : None or (j, 4, 4) float
          Inverse bind matrices, default identity.
        """
        self.joints = list(joints)
        if inverse_bind_matrices is None:
            inverse_bind_matrices = np.eye(4)
        shape = (len(self.joints), 4, 4)
        self.inverse_bind_matrices = np.broadcast_to(inverse_bind_matrices, shape) * 1.0

    def deform(self, mesh, graph) -> NDArray2D[float64]:
        """
        Vertices of a skinned mesh posed by its joints.

        Parameters
        ------------
        mesh : trimesh.Trimesh
          Mesh with `vertex_attributes["joints"]` and `["weights"]`.
        graph : trimesh.scene.transforms.SceneGraph
          Scene graph containing every joint.

        Returns
        ------------
        vertices : (n, 3) float
          Posed vertices in world space.
        """
        world = np.reshape([graph.get(joint)[0] for joint in self.joints], (-1, 4, 4))
        rows = (world @ self.inverse_bind_matrices)[:, :3].reshape((-1, 12))
        joints = mesh.vertex_attributes["joints"]
        weights = np.asarray(mesh.vertex_attributes["weights"], dtype=np.float64)
        blended = (weights[:, :, None] * rows[joints]).sum(axis=1).reshape((-1, 3, 4))
        vertices = np.asarray(mesh.vertices, dtype=np.float64)[:, :, None]
        return np.matmul(blended[:, :, :3], vertices)[:, :, 0] + blended[:, :, 3]

    def __hash__(self) -> int:
        """
        Hash of the skin.

        Returns
        ------------
        hashed : int
          Hash of the joints and inverse bind matrices.
        """
        matrices = caching.hash_fast(self.inverse_bind_matrices.tobytes())
        return hash((tuple(self.joints), matrices))
