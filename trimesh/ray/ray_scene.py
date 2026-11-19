"""
ray/ray_scene.py
----------------

Ray queries against every node of a scene with whichever engine
is installed, reporting which node each hit is on.
"""

import numpy as np

from .. import caching, util
from ..transformations import transform_points
from ..typed import ArrayLike, Integer, NDArray
from .parent import RayParent


class SceneRay(RayParent):
    """
    Ray queries against every node of a `Scene` where it currently
    is, with the same interface as `Trimesh.ray`. With `return_instance`
    each hit also reports its position in `nodes`.
    """

    def __init__(self, scene):
        """
        Parameters
        -------------
        scene : trimesh.Scene
          Scene to cast rays at.
        """
        self.scene = scene
        # trimkit places instances itself so only rebuilds for new geometry
        self._geometry_cache = caching.Cache(id_function=scene.geometry_hash)
        # every other engine casts at the meshes where the nodes put them
        self._placed_cache = caching.Cache(id_function=scene.__hash__)

    @property
    def nodes(self) -> NDArray:
        """
        The node of every instance, as `nodes[index_instance]` names them.

        Returns
        ---------
        nodes : (k,) object
          The scene graph's `nodes_geometry`.
        """
        nodes = np.zeros(len(self.scene.graph.nodes_geometry), dtype=object)
        nodes[:] = self.scene.graph.nodes_geometry
        return nodes

    @property
    def bounds(self) -> NDArray[np.float64]:
        """
        The axis aligned bounds of the scene.

        Returns
        ----------
        bounds : (2, 3) float
          Minimum and maximum corner.
        """
        return self.scene.bounds

    @property
    def instanced(self) -> tuple:
        """
        Every geometry built once into a `trimkit.RayScene`, which
        moving nodes never rebuilds.

        Returns
        ---------
        scene : trimkit.RayScene
          One mesh per geometry, empty for anything not a `Trimesh`.
        geometry : list
          The geometry name of each mesh.
        """
        cached = self._geometry_cache["instanced"]
        if cached is None:
            from trimkit import RayScene

            from ..scene.scene import trimkit_meshes

            cached = (
                RayScene(trimkit_meshes(self.scene.geometry)),
                list(self.scene.geometry),
            )
            self._geometry_cache["instanced"] = cached
        return cached

    @property
    def placed(self) -> tuple:
        """
        Every mesh node's mesh where the node puts it, as one
        intersector which reports the mesh each hit is on.

        Returns
        ---------
        intersector : RayParent or None
          Of every placed mesh concatenated, or None for no faces.
        instances : (p,) int
          The position in `nodes` of each placed mesh.
        offsets : (p,) int
          Where each placed mesh's faces start in the concatenation.
        """
        cached = self._placed_cache["placed"]
        if cached is not None:
            return cached
        from ..base import Trimesh

        pieces, instances = [], []
        for index, node in enumerate(self.scene.graph.nodes_geometry):
            matrix, name = self.scene.graph[node]
            geometry = self.scene.geometry[name]
            if util.is_instance_named(geometry, "Trimesh") and len(geometry.faces) > 0:
                vertices = transform_points(geometry.vertices, matrix)
                pieces.append((vertices, geometry.faces.view(np.ndarray)))
                instances.append(index)
        instances = np.array(instances, dtype=np.int64)
        offsets = np.cumsum([0] + [len(f) for _, f in pieces[:-1]], dtype=np.int64)
        intersector = None
        if len(pieces) > 0:
            shift = np.cumsum([0] + [len(v) for v, _ in pieces[:-1]])
            concatenated = Trimesh(
                vertices=np.vstack([v for v, _ in pieces]),
                faces=np.vstack([f + s for (_, f), s in zip(pieces, shift)]),
                process=False,
            )
            # embree reports which of `pieces` natively as its geometry
            if util.has_module("embreex"):
                from .ray_pyembree import RayMeshIntersector
            else:
                from .ray_triangle import RayMeshIntersector
            intersector = RayMeshIntersector(concatenated, meshes=pieces)
        cached = (intersector, instances, offsets)
        self._placed_cache["placed"] = cached
        return cached

    def intersects_id(
        self,
        ray_origins: ArrayLike,
        ray_directions: ArrayLike,
        multiple_hits: bool = True,
        return_locations: bool = False,
        return_instance: bool = False,
        max_hits: Integer | None = None,
        **kwargs,
    ):
        """
        Find the triangles hit by each ray, across every node.

        Parameters
        ----------
        ray_origins : (n, 3) float
          Origins of rays
        ray_directions : (n, 3) float
          Direction (vector) of rays
        multiple_hits : bool
          Every hit along each ray, or only its first
        return_locations : bool
          Also return where each hit is
        return_instance : bool
          Also return the node of each hit, as a position in `nodes`
        max_hits : None or int
          The most hits per ray, or every hit

        Returns
        ---------
        index_tri : (m,) int
          Indexes of the faces of the hit node's geometry
        index_ray : (m,) int
          Indexes of ray
        locations : (m, 3) float
          Intersection points, only returned if `return_locations`
        index_instance : (m,) int
          Position in `nodes` of each hit, only returned if `return_instance`
        """
        origins = np.asarray(ray_origins, dtype=np.float64)
        directions = np.asarray(ray_directions, dtype=np.float64)
        if util.has_module("trimkit"):
            from .tinybvh import hits_result

            if not multiple_hits:
                max_hits = 1
            scene, geometry = self.instanced
            hits = scene.intersect(
                origins,
                directions,
                *self.scene.graph.instance_matrices(geometry),
                maximum_hits=max_hits,
            )
            return hits_result(
                origins,
                directions,
                hits,
                return_locations=return_locations,
                return_instance=return_instance,
            )

        intersector, instances, offsets = self.placed
        if intersector is None:
            # nothing in the scene has faces to hit
            empty = np.zeros(0, dtype=np.int64)
            result = [empty, empty]
            if return_locations:
                result.append(np.zeros((0, 3), dtype=np.float64))
            if return_instance:
                result.append(empty)
            return tuple(result)
        if max_hits is not None:
            kwargs["max_hits"] = max_hits
        index_tri, index_ray, locations, piece = intersector.intersects_id(
            origins,
            directions,
            multiple_hits=multiple_hits,
            return_locations=True,
            return_instance=True,
            **kwargs,
        )
        # the face in the node's own geometry
        result = [index_tri - offsets[piece], index_ray]
        if return_locations:
            result.append(locations)
        if return_instance:
            result.append(instances[piece])
        return tuple(result)

    def intersects_any(self, ray_origins: ArrayLike, ray_directions: ArrayLike):
        """
        Check if each ray hits any node.

        Parameters
        -----------
        ray_origins : (n, 3) float
          Origins of rays
        ray_directions : (n, 3) float
          Direction (vector) of rays

        Returns
        ----------
        hit : (n,) bool
          Did each ray hit anything
        """
        origins = np.asarray(ray_origins, dtype=np.float64)
        directions = np.asarray(ray_directions, dtype=np.float64)
        if util.has_module("trimkit"):
            scene, geometry = self.instanced
            return scene.intersect_any(
                origins, directions, *self.scene.graph.instance_matrices(geometry)
            )
        intersector = self.placed[0]
        if intersector is None:
            return np.zeros(len(origins), dtype=bool)
        return intersector.intersects_any(origins, directions)

    def __getstate__(self):
        # everything else rebuilds from the scene
        return {"scene": self.scene}

    def __setstate__(self, state):
        self.__init__(state["scene"])

    def __copy__(self, *args):
        return type(self)(self.scene)
