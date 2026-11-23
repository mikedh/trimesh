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
) -> NDArray:
    """
    Sample keyframed values, clamped outside the keyframe range.

    Parameters
    ------------
    times : (k,) float
      Increasing keyframe times in seconds.
    values : (k, ...) float
      Value at each keyframe, `wxyz` quaternions if `rotation`.
    query : float or (m,) float
      Times to sample at.
    step : bool
      Hold each keyframe instead of blending.
    rotation : bool
      Slerp quaternions.

    Returns
    ------------
    sampled : (...) or (m, ...) float
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
    if step:
        return values[np.where(fraction >= 1.0, upper, lower)]
    if rotation:
        return quaternion_slerp(
            values[lower].reshape((-1, 4)),
            values[upper].reshape((-1, 4)),
            np.repeat(fraction, values[0].size // 4),
        ).reshape(values[lower].shape)
    fraction = fraction.reshape(fraction.shape + (1,) * (values.ndim - 1))
    return values[lower] + (values[upper] - values[lower]) * fraction


class Animation:
    """
    Keyframed transforms of scene graph nodes on one timeline.
    """

    def __init__(
        self,
        name: str,
        nodes: Iterable,
        times: ArrayLike,
        matrices: ArrayLike,
        interpolation: str = "linear",
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
        matrices : (k, n, 4, 4) float
          Parent-to-node transforms, or (k, 4, 4) for one node.
        interpolation : str
          One of `INTERPOLATIONS`.
        """
        if interpolation not in INTERPOLATIONS:
            raise ValueError(f"interpolation must be one of {INTERPOLATIONS}!")
        self.interpolation = interpolation
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if len(times) == 0 or (np.diff(times) < 0.0).any():
            raise ValueError("times must be increasing and not empty!")
        self.name = name
        self.nodes = list(nodes)
        self._data = caching.DataStore()
        self._cache = caching.Cache(id_function=self._data.__hash__)
        self._data["times"] = times
        shape = (len(times), len(self.nodes), 4, 4)
        self._data["matrices"] = np.asarray(matrices, dtype=np.float64).reshape(shape)

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
    def matrices(self) -> NDArray4D[float64]:
        """
        Transform of every node at every keyframe.

        Returns
        ------------
        matrices : (k, n, 4, 4) float
          Parent-to-node transforms.
        """
        return self._data["matrices"]

    @caching.cache_decorator
    def trs(self) -> NDArray3D[float64]:
        """
        Matrices decomposed into translation, rotation and scale.

        Returns
        ------------
        trs : (k, n, 10) float
          Translation, `wxyz` quaternion and scale, hemisphere-aligned.
        """
        trs = np.concatenate(trs_from_matrix(self.matrices), axis=2)
        # flip quaternions so neighbors blend the short way around
        flips = (trs[1:, :, 3:7] * trs[:-1, :, 3:7]).sum(axis=2) < 0.0
        trs[1:, :, 3:7] *= np.cumprod(np.where(flips, -1.0, 1.0), axis=0)[..., None]
        return trs

    @caching.cache_decorator
    def tangents(self) -> NDArray3D[float64]:
        """
        Monotone cubic derivatives of `trs` with respect to time.

        Returns
        ------------
        tangents : (k, n, 10) float
          Harmonic mean of the slopes on either side of each keyframe and
          zero at extrema (Fritsch-Butland 1984), so a held or linear part
          stays held or linear and nothing overshoots.
        """
        times = self.times
        width = np.diff(times)[:, None, None]
        # a segment no wider than float32 time resolution is a jump, such as
        # a held glTF STEP, whose keys take the slope on their other side
        jump = width <= np.spacing(np.float32(times[1:]))[:, None, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(jump, np.nan, np.diff(self.trs, axis=0) / width)
        if len(slope) == 0:
            return np.zeros_like(self.trs)
        before = np.where(np.isnan(slope[:-1]), slope[1:], slope[:-1])
        after = np.where(np.isnan(slope[1:]), slope[:-1], slope[1:])
        middle = np.divide(
            2.0 * before * after,
            before + after,
            out=np.zeros_like(before),
            where=before * after > 0.0,
        )
        slope = np.nan_to_num(slope)
        return np.concatenate((slope[:1], middle, slope[-1:]))

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
        times = np.asarray(self.times)
        if self.interpolation == "cubic" and len(times) > 1:
            # the cubic Hermite segment around `time` as a Bezier curve
            upper = np.clip(np.searchsorted(times, time), 1, len(times) - 1)
            width = times[upper] - times[upper - 1]
            fraction = np.clip((time - times[upper - 1]) / max(width, tol.zero), 0, 1)
            trs, tangents = self.trs[upper - 1 : upper + 1], self.tangents
            reach = tangents[upper - 1 : upper + 1] * width / 3.0
            controls = np.stack((trs[0], trs[0] + reach[0], trs[1] - reach[1], trs[1]))
            sampled = evaluate_bezier(controls.transpose((1, 0, 2)), fraction)
        else:
            step = self.interpolation == "step"
            sampled = interpolate_keyframes(times, self.trs, time, step)
            sampled[:, 3:7] = interpolate_keyframes(
                times, self.trs[..., 3:7], time, step, rotation=True
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
