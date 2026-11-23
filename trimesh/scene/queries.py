"""
scene/queries.py
-----------------

`trimkit` queries against every node of a `Scene` where it currently
is: `trimkit.RayScene` and `trimkit.CollisionScene` with the scene's
instances passed in, and otherwise exactly trimkit's API.
"""

import numpy as np
from trimkit import CollisionScene, RayScene

from .. import util
from ..typed import NDArray


def trimkit_meshes(geometry: dict) -> list:
    """
    Every geometry as the `(vertices, faces)` trimkit builds, in order,
    with anything that isn't a `Trimesh` as an empty mesh that nothing
    ever touches or hits so instance indexes line up with `geometry`.

    Parameters
    ------------
    geometry
      Keyed `{name: geometry}`, i.e. `Scene.geometry`.

    Returns
    ---------
    meshes : list of ((n, 3) float, (m, 3) int)
      One mesh per geometry.
    """
    empty = (np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64))
    meshes = dict.fromkeys(geometry, empty)
    meshes.update(
        {
            name: (value.vertices.view(np.ndarray), value.faces.view(np.ndarray))
            for name, value in geometry.items()
            if util.is_instance_named(value, "Trimesh")
        }
    )
    return list(meshes.values())


class SceneInstances:
    """
    Every node of a `Scene` as the instances trimkit queries take.
    To query other instances, such as many poses of a robot, call
    the trimkit base method with your own `instance_matrices`.
    """

    @property
    def instances(self) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Where every node in `nodes` currently places its geometry.

        Returns
        ---------
        instance_matrices : (k, 4, 4) float
          Each node's transform from the base frame.
        instance_index : (k,) int
          The geometry each node places.
        """
        return self.scene.graph.instance_matrices(self.geometry)

    @property
    def nodes(self) -> NDArray:
        """
        The node of every instance, as `nodes[instances]` names them.

        Returns
        ---------
        nodes : (k,) object
          The scene graph's `nodes_geometry`.
        """
        nodes = np.zeros(len(self.scene.graph.nodes_geometry), dtype=object)
        nodes[:] = self.scene.graph.nodes_geometry
        return nodes

    def boxes(self):
        """
        `boxes` of every node where it currently is.

        Returns
        ---------
        boxes : trimkit.Boxes
          The world axis aligned box of each instance.
        """
        return super().boxes(*self.instances)

    def __reduce__(self):
        # rebuilt from the scene rather than from trimkit's own state
        return (type(self), (self.scene,))


class SceneRay(SceneInstances, RayScene):
    """
    A `trimkit.RayScene` of a `Scene`'s geometry that casts rays at
    every node of the scene where it currently is.
    """

    def __init__(self, scene):
        """
        Build every geometry once: moving nodes never rebuilds it.

        Parameters
        ----------
        scene : trimesh.Scene
          Scene to cast rays at.
        """
        super().__init__(trimkit_meshes(scene.geometry))
        self.scene = scene
        # the order of the meshes built, which instance indexes refer to
        self.geometry = tuple(scene.geometry)

    def intersect(self, origins, directions, **kwargs):
        """
        `trimkit.RayScene.intersect` of every node.

        Parameters
        ----------
        origins : (n, 3) float
          Where each ray starts.
        directions : (n, 3) float
          Each ray's direction, of any length.
        **kwargs
          Passed to `trimkit.RayScene.intersect`, i.e. `maximum_hits`.

        Returns
        ---------
        rays, instances, faces, distances
          As `trimkit.RayScene.intersect`, with `nodes[instances]`
          naming the node of each hit.
        """
        return super().intersect(origins, directions, *self.instances, **kwargs)

    def intersect_any(self, origins, directions):
        """
        `trimkit.RayScene.intersect_any` of every node.

        Parameters
        ----------
        origins : (n, 3) float
          Where each ray starts.
        directions : (n, 3) float
          Each ray's direction, of any length.

        Returns
        ---------
        hit : (n,) bool
          Whether each ray hits any node.
        """
        return super().intersect_any(origins, directions, *self.instances)


class SceneCollision(SceneInstances, CollisionScene):
    """
    A `trimkit.CollisionScene` of a `Scene`'s geometry that queries
    every node of the scene where it currently is.
    """

    def __init__(self, scene):
        """
        Build every geometry once: moving nodes never rebuilds it.

        Parameters
        ----------
        scene : trimesh.Scene
          Scene to query, a convex mesh measured as a solid.
        """
        super().__init__(
            trimkit_meshes(scene.geometry),
            convex=np.array(
                [
                    util.is_instance_named(value, "Trimesh") and value.is_convex
                    for value in scene.geometry.values()
                ],
                dtype=bool,
            ),
        )
        self.scene = scene
        # the order of the meshes built, which instance indexes refer to
        self.geometry = tuple(scene.geometry)

    def collides(self, **kwargs):
        """
        `trimkit.CollisionScene.collides` of every node.

        Parameters
        ----------
        **kwargs
          Passed to `trimkit.CollisionScene.collides`, i.e. `pairs`.

        Returns
        ---------
        result : trimkit.CollisionResult
          Rows of instance pairs, which `nodes` names.
        """
        return super().collides(*self.instances, **kwargs)

    def distances(self, **kwargs):
        """
        `trimkit.CollisionScene.distances` of every node.

        Parameters
        ----------
        **kwargs
          Passed to `trimkit.CollisionScene.distances`, i.e. `pairs`.

        Returns
        ---------
        result : trimkit.CollisionResult
          Rows of instance pairs, which `nodes` names.
        """
        return super().distances(*self.instances, **kwargs)
