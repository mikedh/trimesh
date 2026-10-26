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
    or along a cubic spline through every keyframe.
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
    def trs(self) -> NDArray3D[float64]:
        """
        The matrices split into translation, rotation, and scale once
        so sampling between keyframes is only a blend.

        Returns
        ------------
        trs : (k, n, 10) float
          The translation, rotation quaternion ordered `w, x, y, z`, and
          scale of every node at every keyframe, with the quaternions of
          neighboring keyframes in the same hemisphere so a blend takes
          the shorter way around.
        """
        trs = np.concatenate(trs_from_matrix(self.matrices), axis=2)
        # a quaternion and its negation are the same rotation
        flips = (trs[1:, :, 3:7] * trs[:-1, :, 3:7]).sum(axis=2) < 0.0
        trs[1:, :, 3:7] *= np.cumprod(np.where(flips, -1.0, 1.0), axis=0)[..., None]
        return trs

    @caching.cache_decorator
    def spline(self) -> Callable | None:
        """
        A cubic spline through every keyframe of `trs`, which is what a
        CUBIC animation samples and exports the derivative of.

        Returns
        ------------
        spline : None or scipy.interpolate.CubicSpline
          Of `trs`, or None with one keyframe or if scipy isn't installed.
        """
        if len(self.times) < 2:
            return None
        try:
            from scipy.interpolate import CubicSpline
        except ImportError:
            log.warning(f"`{self.name}` is sampled linearly without scipy")
            return None
        return CubicSpline(np.asarray(self.times), self.trs)

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

    def interpolate(self, time: Floating) -> NDArray3D[float64]:
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
            sampled = self.spline(np.clip(time, times[0], times[-1]))
        else:
            # held or blended between keyframes, without cubic tangents
            mode = Interpolation.LINEAR
            if self.interpolation == Interpolation.STEP:
                mode = Interpolation.STEP
            sampled = sample_channel(times, self.trs, time, mode)
            sampled[:, 3:7] = sample_channel(times, self.trs[..., 3:7], time, mode, True)
        # where `trs_matrix` normalizes quaternions as glTF specifies
        return trs_matrix(sampled[:, :3], sampled[:, 3:7], sampled[:, 7:])


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
            inverse_bind_matrices = np.eye(4)
        shape = (len(self.joints), 4, 4)
        self.inverse_bind_matrices = np.broadcast_to(inverse_bind_matrices, shape) * 1.0

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
        attributes = mesh.vertex_attributes
        joints, weights = attributes["joints"], attributes["weights"]
        return skin_points(
            mesh.vertices, joints, weights, world @ self.inverse_bind_matrices
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
