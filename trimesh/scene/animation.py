"""
animation.py
--------------

Keyframed animation of scene graph nodes as stacked matrices, and the
skins which let scene graph nodes move the vertices of a mesh.
"""

from enum import Enum

import numpy as np

from .. import caching
from ..constants import log
from ..transformations import sample_channel, skin_points, trs_from_matrix, trs_matrix
from ..typed import (
    ArrayLike,
    Callable,
    Floating,
    Iterable,
    NDArray1D,
    NDArray2D,
    NDArray3D,
    NDArray4D,
    float64,
)


class Interpolation(str, Enum):
    """
    How an animation is sampled between keyframes: blending the
    translation, rotation, and scale linearly, holding each keyframe,
    or along a cubic spline through every keyframe. A `str` mixin as
    `StrEnum` needs Python 3.11 and trimesh supports 3.10.
    """

    LINEAR = "linear"
    STEP = "step"
    CUBIC = "cubic"


class Animation:
    """
    Motion of scene graph nodes on one timeline, stored as the transform
    of every node from its parent at every keyframe and sampled between
    them by its `Interpolation`.
    """

    def __init__(
        self,
        name: str,
        nodes: Iterable,
        times: ArrayLike,
        matrices: ArrayLike,
        interpolation: Interpolation | str = Interpolation.LINEAR,
    ):
        """
        Create an animation of scene graph nodes.

        Parameters
        ------------
        name : str
          Animations sharing a name export as one glTF animation.
        nodes : (n,) sequence of hashable
          The scene graph nodes this moves relative to their parents.
        times : (k,) float
          Increasing keyframe times in seconds.
        matrices : (k, n, 4, 4) float
          Homogeneous transform of every node from its parent at every
          keyframe, which may be (k, 4, 4) for a single node.
        interpolation : Interpolation or str
          How to sample between keyframes, where a value which isn't
          one raises a ValueError.
        """
        self.interpolation = Interpolation(interpolation)
        times = caching.tracked_array(times, dtype=np.float64).reshape(-1)
        if len(times) == 0 or (np.diff(times) < 0.0).any():
            raise ValueError("times must be increasing and not empty!")
        self.name = name
        self.nodes = list(nodes)
        self._data = caching.DataStore()
        self._cache = caching.Cache(id_function=self._data.__hash__)
        self._data["times"] = times
        # raises a ValueError if there isn't a matrix per node per keyframe
        shape = (len(times), len(self.nodes), 4, 4)
        self._data["matrices"] = np.asarray(matrices, dtype=np.float64).reshape(shape)

    @property
    def times(self) -> NDArray1D[float64]:
        """
        When each keyframe occurs, which may be edited in place.

        Returns
        ------------
        times : (k,) float
          Increasing keyframe times in seconds.
        """
        return self._data["times"]

    @property
    def matrices(self) -> NDArray4D[float64]:
        """
        The transform of every node at every keyframe, which may be
        edited in place.

        Returns
        ------------
        matrices : (k, n, 4, 4) float
          Homogeneous transform of every node from its parent.
        """
        return self._data["matrices"]

    @caching.cache_decorator
    def trs(self) -> tuple[NDArray3D[float64], NDArray3D[float64], NDArray3D[float64]]:
        """
        The matrices split into translation, rotation, and scale once
        so sampling between keyframes is only a blend.

        Returns
        ------------
        translation : (k, n, 3) float
          Translation of every node at every keyframe.
        quaternion : (k, n, 4) float
          Rotation ordered `w, x, y, z`, with neighboring keyframes in
          the same hemisphere so a blend takes the shorter way around.
        scale : (k, n, 3) float
          Scale of every node at every keyframe.
        """
        # plain arrays as every operation on a tracked one is tracked
        matrices = np.asarray(self.matrices)
        translation, quaternion, scale = (
            part.reshape((*matrices.shape[:2], -1))
            for part in trs_from_matrix(matrices.reshape((-1, 4, 4)))
        )
        # a quaternion and its negation are the same rotation
        flips = (quaternion[1:] * quaternion[:-1]).sum(axis=2) < 0.0
        quaternion[1:] *= np.cumprod(np.where(flips, -1.0, 1.0), axis=0)[..., None]
        return translation, quaternion, scale

    @caching.cache_decorator
    def spline(self) -> Callable | None:
        """
        A cubic spline through every keyframe of `trs`, which is what a
        CUBIC animation samples and exports the derivative of.

        Returns
        ------------
        spline : None or scipy.interpolate.CubicSpline
          Of the translation, quaternion, and scale stacked into (k, n, 10),
          or None with one keyframe or if scipy isn't installed.
        """
        if len(self.times) < 2:
            return None
        try:
            from scipy.interpolate import CubicSpline
        except ImportError:
            log.warning(f"`{self.name}` is sampled linearly without scipy")
            return None
        return CubicSpline(np.asarray(self.times), np.concatenate(self.trs, axis=2))

    @property
    def duration(self) -> float:
        """
        The time of the last keyframe, which is where a player loops.

        Returns
        ------------
        duration : float
          Seconds.
        """
        return float(self.times[-1])

    def at(self, time: Floating) -> NDArray3D[float64]:
        """
        The transform of every node from its parent at a time, holding
        the first and last keyframe outside of the animation.

        Parameters
        ------------
        time : float
          Seconds.

        Returns
        ------------
        matrices : (n, 4, 4) float
          Homogeneous transform of every node in `nodes` order.
        """
        times = np.asarray(self.times)
        if self.interpolation == Interpolation.CUBIC and self.spline is not None:
            # where `trs_matrix` normalizes quaternions as glTF specifies
            sampled = self.spline(np.clip(time, times[0], times[-1]))
            return trs_matrix(sampled[:, :3], sampled[:, 3:7], sampled[:, 7:])
        # the keyframes on either side of the time are all it reads
        bracket = np.clip(np.searchsorted(times, time) + [-1, 0], 0, len(times) - 1)
        # held or blended between them, without cubic tangents
        mode, query = Interpolation.LINEAR, [time]
        if self.interpolation == Interpolation.STEP:
            mode = Interpolation.STEP
        return trs_matrix(
            *(
                sample_channel(
                    times[bracket], part[bracket], None, None, mode, query, rotation
                )[0]
                for part, rotation in zip(self.trs, (False, True, False))
            )
        )


class Skin:
    """
    Scene graph nodes which move the vertices of a mesh, which glTF
    calls a skin: every vertex moves by a blend of its joints given by
    `mesh.vertex_attributes["joints"]` and `["weights"]`.
    """

    def __init__(self, joints: Iterable, inverse_bind_matrices: ArrayLike | None = None):
        """
        Create a skin.

        Parameters
        ------------
        joints : (j,) sequence of hashable
          Scene graph nodes in the order a vertex's joint indices use.
        inverse_bind_matrices : None or (j, 4, 4) float
          The inverse of each joint's world transform when the mesh
          was attached to it, which is the identity if not passed.
        """
        self.joints = list(joints)
        if inverse_bind_matrices is None:
            inverse_bind_matrices = np.tile(np.eye(4), (len(self.joints), 1, 1))
        self.inverse_bind_matrices = np.array(
            inverse_bind_matrices, dtype=np.float64
        ).reshape((len(self.joints), 4, 4))

    def deform(self, mesh, graph) -> NDArray2D[float64]:
        """
        The vertices of a skinned mesh where its joints put them,
        which is in world space as the node holding the mesh doesn't
        move it, as glTF specifies.

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
        # each joint's world transform times its inverse bind matrix
        world = np.reshape([graph.get(joint)[0] for joint in self.joints], (-1, 4, 4))
        return skin_points(
            mesh.vertices,
            mesh.vertex_attributes["joints"],
            mesh.vertex_attributes["weights"],
            np.matmul(world, self.inverse_bind_matrices),
        )

    def __hash__(self) -> int:
        """
        A hash which changes when the joints or matrices do.

        Returns
        ------------
        hashed : int
          Hash of the joints and inverse bind matrices.
        """
        matrices = caching.hash_fast(self.inverse_bind_matrices.tobytes())
        return hash((tuple(self.joints), matrices))
