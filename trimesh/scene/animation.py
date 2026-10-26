"""
animation.py
--------------

Keyframed animation of a scene graph as homogeneous matrices, and the
skins which let scene graph nodes move the vertices of a mesh.
"""

import numpy as np

from .. import caching
from ..transformations import (
    sample_channel,
    skin_points,
    trs_from_matrix,
    trs_matrix,
)
from ..typed import (
    ArrayLike,
    Floating,
    Iterable,
    NDArray1D,
    NDArray2D,
    NDArray3D,
    NDArray4D,
    float64,
)

# how a transform between keyframes is found
INTERPOLATIONS = ("linear", "step")


class Animation:
    """
    Rigid motion of scene graph edges on one timeline, stored as the
    `frame_from -> frame_to` matrix of every edge at every keyframe.
    """

    def __init__(
        self,
        edges: Iterable,
        times: ArrayLike,
        matrices: ArrayLike,
        name: str = "animation",
        interpolation: str = "linear",
    ):
        """
        Create an animation of scene graph edges.

        Parameters
        ------------
        edges : (e,) sequence of (frame_from, frame_to)
          The scene graph edges this drives, where a `frame_from`
          of None is the base frame.
        times : (k,) float
          Increasing keyframe times in seconds.
        matrices : (k, e, 4, 4) float
          The transform of every edge at every keyframe, which
          may be (k, 4, 4) for a single edge.
        name : str
          Animations sharing a name export as one glTF animation.
        interpolation : str
          "linear" blends between keyframes and "step" holds each one.
        """
        self._data = caching.DataStore()
        self._cache = caching.Cache(id_function=self._data.__hash__)
        self.edges = [tuple(edge) for edge in edges]
        self.times = times
        self.matrices = np.reshape(matrices, (len(self.times), len(self.edges), 4, 4))
        self.name = name
        if interpolation not in INTERPOLATIONS:
            raise ValueError(f"interpolation must be one of {INTERPOLATIONS}")
        self.interpolation = interpolation

    @property
    def times(self) -> NDArray1D[float64]:
        """
        When each keyframe occurs.

        Returns
        ------------
        times : (k,) float
          Increasing keyframe times in seconds.
        """
        return self._data["times"]

    @times.setter
    def times(self, values: ArrayLike) -> None:
        """
        Set the keyframe times.

        Parameters
        ------------
        values : (k,) float
          Increasing keyframe times in seconds.
        """
        values = caching.tracked_array(values, dtype=np.float64).reshape(-1)
        if len(values) == 0 or (np.diff(values) < 0.0).any():
            raise ValueError("times must be increasing and not empty!")
        self._data["times"] = values

    @property
    def matrices(self) -> NDArray4D[float64]:
        """
        The transform of every edge at every keyframe, which may be
        edited in place.

        Returns
        ------------
        matrices : (k, e, 4, 4) float
          Homogeneous `frame_from -> frame_to` transforms.
        """
        return self._data["matrices"]

    @matrices.setter
    def matrices(self, values: ArrayLike) -> None:
        """
        Set the transform of every edge at every keyframe.

        Parameters
        ------------
        values : (k, e, 4, 4) float
          Homogeneous `frame_from -> frame_to` transforms.
        """
        values = caching.tracked_array(values, dtype=np.float64)
        if values.shape != (len(self.times), len(self.edges), 4, 4):
            raise ValueError(
                f"matrices must be (keyframes, edges, 4, 4) not {values.shape}"
            )
        self._data["matrices"] = values

    @caching.cache_decorator
    def trs(self) -> tuple[NDArray3D[float64], NDArray3D[float64], NDArray3D[float64]]:
        """
        The matrices split into translation, rotation, and scale once
        so sampling between keyframes is only a blend.

        Returns
        ------------
        translation : (k, e, 3) float
          Translation of every edge at every keyframe.
        quaternion : (k, e, 4) float
          Rotation ordered `w, x, y, z`, with neighboring keyframes in
          the same hemisphere so a blend takes the shorter way around.
        scale : (k, e, 3) float
          Scale of every edge at every keyframe.
        """
        shape = self.matrices.shape[:2]
        translation, quaternion, scale = trs_from_matrix(
            self.matrices.reshape((-1, 4, 4))
        )
        quaternion = quaternion.reshape((*shape, 4))
        # a quaternion and its negation are the same rotation
        flips = (quaternion[1:] * quaternion[:-1]).sum(axis=2) < 0.0
        quaternion[1:] *= np.cumprod(np.where(flips, -1.0, 1.0), axis=0)[..., None]
        return translation.reshape((*shape, 3)), quaternion, scale.reshape((*shape, 3))

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

    def at(self, time: Floating | ArrayLike) -> NDArray3D[float64] | NDArray4D[float64]:
        """
        The transform of every edge at one or more times, holding the
        first and last keyframe outside of the animation.

        Parameters
        ------------
        time : float or (m,) float
          Seconds.

        Returns
        ------------
        matrices : (e, 4, 4) or (m, e, 4, 4) float
          Transform of every edge at each time.
        """
        query = np.asanyarray(time, dtype=np.float64)
        shared = {"times": self.times, "query": query.reshape(-1)}
        if self.interpolation == "step":
            # hold each keyframe until the next one is reached
            sampled = sample_channel(
                values=self.matrices,
                tangent_in=None,
                tangent_out=None,
                mode="step",
                rotation=False,
                **shared,
            )
        else:
            translation, quaternion, scale = (
                sample_channel(
                    values=values,
                    tangent_in=None,
                    tangent_out=None,
                    mode="linear",
                    rotation=rotation,
                    **shared,
                )
                for values, rotation in zip(self.trs, (False, True, False))
            )
            count = len(self.edges)
            sampled = trs_matrix(
                translation.reshape((-1, 3)),
                quaternion.reshape((-1, 4)),
                scale.reshape((-1, 3)),
            ).reshape((-1, count, 4, 4))

        if query.ndim == 0:
            return sampled[0]
        return sampled

    def copy(self) -> "Animation":
        """
        A copy of this animation which shares no data with it.

        Returns
        ------------
        copied : Animation
          The same edges, times, and matrices.
        """
        return Animation(
            edges=list(self.edges),
            times=np.array(self.times),
            matrices=np.array(self.matrices),
            name=self.name,
            interpolation=self.interpolation,
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
        self._data = caching.DataStore()
        self.joints = list(joints)
        if inverse_bind_matrices is None:
            inverse_bind_matrices = np.tile(np.eye(4), (len(self.joints), 1, 1))
        self.inverse_bind_matrices = inverse_bind_matrices

    @property
    def inverse_bind_matrices(self) -> NDArray3D[float64]:
        """
        The inverse of each joint's world transform when the mesh was
        attached to it, which may be edited in place.

        Returns
        ------------
        inverse_bind_matrices : (j, 4, 4) float
          One homogeneous transform per joint.
        """
        return self._data["inverse_bind_matrices"]

    @inverse_bind_matrices.setter
    def inverse_bind_matrices(self, values: ArrayLike) -> None:
        """
        Set the inverse bind matrix of every joint.

        Parameters
        ------------
        values : (j, 4, 4) float
          One homogeneous transform per joint.
        """
        values = caching.tracked_array(values, dtype=np.float64)
        if values.shape != (len(self.joints), 4, 4):
            raise ValueError(
                f"inverse bind matrices must be (joints, 4, 4) not {values.shape}"
            )
        self._data["inverse_bind_matrices"] = values

    def matrices(self, graph) -> NDArray3D[float64]:
        """
        Where every joint moves vertices to right now.

        Parameters
        ------------
        graph : trimesh.scene.transforms.SceneGraph
          Scene graph containing every joint.

        Returns
        ------------
        matrices : (j, 4, 4) float
          Each joint's world transform times its inverse bind matrix.
        """
        world = np.reshape([graph.get(joint)[0] for joint in self.joints], (-1, 4, 4))
        return np.matmul(world, self.inverse_bind_matrices)

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
        return skin_points(
            mesh.vertices,
            mesh.vertex_attributes["joints"],
            mesh.vertex_attributes["weights"],
            self.matrices(graph),
        )

    def copy(self) -> "Skin":
        """
        A copy of this skin which shares no data with it.

        Returns
        ------------
        copied : Skin
          The same joints and inverse bind matrices.
        """
        return Skin(
            joints=list(self.joints),
            inverse_bind_matrices=np.array(self.inverse_bind_matrices),
        )

    def __hash__(self) -> int:
        """
        A hash which changes when the joints or matrices do.

        Returns
        ------------
        hashed : int
          Hash of the joints and inverse bind matrices.
        """
        return hash((tuple(self.joints), self._data.__hash__()))
