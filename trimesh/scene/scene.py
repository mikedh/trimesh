import numpy as np
import collections

from .. import util
from .. import units
from .. import convex
from .. import caching
from .. import grouping
from .. import transformations

from .. import bounds as bounds_module

from ..io import gltf
from .transforms import TransformForest


class Scene:
    """
    A simple scene graph which can be rendered directly via
    pyglet/openGL or through other endpoints such as a
    raytracer. Meshes are added by name, which can then be
    moved by updating transform in the transform tree.
    """

    def __init__(self,
                 geometry=None,
                 base_frame='world',
                 metadata={},
                 graph=None):
        """
        Create a new Scene object.

        Parameters
        -------------
        geometry:   Trimesh, Path2D, Path3D object, or list of same
        base_frame: str, name of base frame
        metadata:   dict, any metadata about the scene
        graph:      TransformForest, graph to use
                    None: create a new TransformForest
        """
        # mesh name : Trimesh object
        self.geometry = collections.OrderedDict()

        # graph structure of instances
        if graph is None:
            # create a new graph
            self.graph = TransformForest(base_frame=base_frame)
        else:
            # if we've been passed a graph use it
            self.graph = graph

        # create our cache
        self._cache = caching.Cache(id_function=self.md5)

        # add passed geometry to scene
        self.add_geometry(geometry)

        # hold metadata about the scene
        self.metadata = {}
        self.metadata.update(metadata)

    def add_geometry(self,
                     geometry,
                     node_name=None):
        """
        Add a geometry to the scene.

        If the mesh has multiple transforms defined in its
        metadata, they will all be copied into the
        TransformForest of the current scene automatically.

        Parameters
        ----------
        geometry: Trimesh, Path3D, or list of same
        node_name: name in the scene graph

        Returns
        ----------
        node_name: str, name of node in self.graph
        """

        if geometry is None:
            return
        elif util.is_sequence(geometry):
            # if passed a sequence add all elements
            return [self.add_geometry(i) for i in geometry]
        elif isinstance(geometry, dict):
            # if someone passed us a dict of geometry
            return self.geometry.update(geometry)

        # if object has metadata indicating different
        # information use those values
        if 'name' in geometry.metadata:
            name = geometry.metadata['name']
        else:
            # try to create a simple name
            name = 'geometry_' + str(len(self.geometry))
            # if its already taken make a not- simple name
            if name in self.geometry:
                name = 'geometry_' + util.unique_id().upper()

        # save the geometry reference
        self.geometry[name] = geometry

        # create a unique node name if not passed
        if node_name is None:
            node_name = name + util.unique_id().upper()

        # create an identity transform from world
        transform = np.eye(4)
        self.graph.update(frame_to=node_name,
                          matrix=transform,
                          geometry=name,
                          geometry_flags={'visible': True})
        return node_name

    def md5(self):
        """
        MD5 of scene which will change when meshes or
        transforms are changed

        Returns
        --------
        hashed: str, MD5 hash of scene
        """
        # start with transforms hash
        hashes = [self.graph.md5()]
        for g in self.geometry.values():
            if hasattr(g, 'md5'):
                hashes.append(g.md5())
            elif hasattr(g, 'tostring'):
                hashes.append(str(hash(g.tostring())))
            else:
                # try to just straight up hash
                # this may raise errors
                hashes.append(str(hash(g)))

        md5 = util.md5_object(''.join(hashes))

        return md5

    @property
    def is_empty(self):
        """
        Does the scene have anything in it.

        Returns
        ----------
        is_empty: bool, True if nothing is in the scene
        """

        is_empty = len(self.geometry) == 0
        return is_empty

    @caching.cache_decorator
    def bounds_corners(self):
        """
        A list of points that represent the corners of the
        AABB of every geometry in the scene.

        This can be useful if you want to take the AABB in
        a specific frame.

        Returns
        -----------
        corners: (n, 3) float, points in space
        """
        # the saved corners of each instance
        corners_inst = []
        # (n, 3) float corners of each geometry
        corners_geom = {k: bounds_module.corners(v.bounds)
                        for k, v in self.geometry.items()}

        for node_name in self.graph.nodes_geometry:
            # access the transform and geometry name from node
            transform, geometry_name = self.graph[node_name]
            # not all nodes have associated geometry
            if geometry_name is None:
                continue
            # transform geometry corners into where
            # the instance of the geometry is located
            corners_inst.extend(
                transformations.transform_points(
                    corners_geom[geometry_name],
                    transform))
        # make corners numpy array
        corners_inst = np.array(corners_inst,
                                dtype=np.float64)
        return corners_inst

    @caching.cache_decorator
    def bounds(self):
        """
        Return the overall bounding box of the scene.

        Returns
        --------
        bounds: (2,3) float points for min, max corner
        """
        corners = self.bounds_corners
        bounds = np.array([corners.min(axis=0),
                           corners.max(axis=0)])
        return bounds

    @caching.cache_decorator
    def extents(self):
        """
        Return the axis aligned box size of the current scene.

        Returns
        ----------
        extents: (3,) float, bounding box sides length
        """
        return np.diff(self.bounds, axis=0).reshape(-1)

    @caching.cache_decorator
    def scale(self):
        """
        The approximate scale of the mesh

        Returns
        -----------
        scale: float, the mean of the bounding box edge lengths
        """
        scale = (self.extents ** 2).sum() ** .5
        return scale

    @caching.cache_decorator
    def centroid(self):
        """
        Return the center of the bounding box for the scene.

        Returns
        --------
        centroid: (3) float point for center of bounding box
        """
        centroid = np.mean(self.bounds, axis=0)
        return centroid

    @caching.cache_decorator
    def triangles(self):
        """
        Return a correctly transformed polygon soup of the
        current scene.

        Returns
        ----------
        triangles: (n,3,3) float, triangles in space
        """
        triangles = collections.deque()
        triangles_node = collections.deque()

        for node_name in self.graph.nodes_geometry:
            # which geometry does this node refer to
            transform, geometry_name = self.graph[node_name]

            # get the actual potential mesh instance
            geometry = self.geometry[geometry_name]
            if not hasattr(geometry, 'triangles'):
                continue
            # append the (n, 3, 3) triangles to a sequence
            triangles.append(
                transformations.transform_points(
                    geometry.triangles.copy().reshape((-1, 3)),
                    matrix=transform))
            # save the node names for each triangle
            triangles_node.append(
                np.tile(node_name,
                        len(geometry.triangles)))
        # save the resulting nodes to the cache
        self._cache['triangles_node'] = np.hstack(triangles_node)
        triangles = np.vstack(triangles).reshape((-1, 3, 3))
        return triangles

    @caching.cache_decorator
    def triangles_node(self):
        """
        Which node of self.graph does each triangle come from.

        Returns
        ---------
        triangles_index : (len(self.triangles),)
            Node name for each triangle
        """
        populate = self.triangles
        return self._cache['triangles_node']

    @caching.cache_decorator
    def geometry_identifiers(self):
        """
        Look up geometries by identifier MD5

        Returns
        ---------
        identifiers: dict, identifier md5: key in self.geometry
        """
        identifiers = {mesh.identifier_md5: name
                       for name, mesh in self.geometry.items()}
        return identifiers

    @caching.cache_decorator
    def duplicate_nodes(self):
        """
        Return a sequence of node keys of identical meshes.

        Will combine meshes duplicated by copying in space with different keys in
        self.geometry, as well as meshes repeated by self.nodes.

        Returns
        -----------
        duplicates: (m) sequence of keys to self.nodes that represent
                     identical geometry
        """
        # if there is no geometry we can have no duplicate nodes
        if len(self.geometry) == 0:
            return []

        # geometry name : md5 of mesh
        mesh_hash = {k: int(m.identifier_md5, 16)
                     for k, m in self.geometry.items()}

        # the name of nodes in the scene graph with geometry
        node_names = np.array(self.graph.nodes_geometry)
        # the geometry names for each node in the same order
        node_geom = np.array([self.graph[i][1] for i in node_names])

        # the mesh md5 for each node in the same order
        node_hash = np.array([mesh_hash[v] for v in node_geom])

        # indexes of identical hashes
        node_groups = grouping.group(node_hash)

        # sequence of node names, where each sublist has identical geometry
        duplicates = [np.sort(node_names[g]).tolist() for g in node_groups]
        return duplicates

    def set_camera(self,
                   angles=None,
                   distance=None,
                   center=None):
        """
        Add a transform to self.graph for 'camera'

        If arguments are not passed sane defaults will be figured out.

        Parameters
        -----------
        angles : (3,) float
                     Initial euler angles in radians
        distance:  float
                     Distance from centroid
        center:    (3,) float
                     Point camera should be center on
        """
        if len(self.geometry) == 0:
            return

        if center is None:
            center = self.centroid

        if distance is None:
            # for a 60.0 degree horizontal FOV
            distance = ((self.extents.max() / 2) /
                        np.tan(np.radians(60.0) / 2.0))

        if angles is None:
            angles = np.zeros(3)

        translation = np.eye(4)
        translation[0:3, 3] = center
        # offset by a distance set by the model size
        # the FOV is set for the Y axis, we multiply by a lightly
        # padded aspect ratio to make sure the model is in view
        translation[2][3] += distance * 1.35

        transform = np.dot(transformations.rotation_matrix(
            angles[0],
            [1, 0, 0],
            point=center),
            transformations.rotation_matrix(
            angles[1],
            [0, 1, 0],
            point=center))
        transform = np.dot(transform, translation)

        self.graph.update(frame_from='camera',
                          frame_to=self.graph.base_frame,
                          matrix=transform)

    def rezero(self):
        """
        Move the current scene so that the AABB of the whole
        scene is centered at the origin.

        Does this by changing the base frame to a new, offset
        base frame.
        """
        if self.is_empty or np.allclose(self.centroid, 0.0):
            # early exit since what we want already exists
            return

        # the transformation to move the overall scene to AABB centroid
        matrix = np.eye(4)
        matrix[:3, 3] = -self.centroid

        # we are going to change the base frame
        new_base = str(self.graph.base_frame) + '_I'
        self.graph.update(frame_from=new_base,
                          frame_to=self.graph.base_frame,
                          matrix=matrix)
        self.graph.base_frame = new_base

    def dump(self):
        """
        Append all meshes in scene to a list of meshes.

        Returns
        ----------
        dumped: (n,) list, of Trimesh objects transformed to their
                           location the scene.graph
        """
        result = collections.deque()

        for node_name in self.graph.nodes_geometry:
            transform, geometry_name = self.graph[node_name]

            current = self.geometry[geometry_name].copy()
            current.apply_transform(transform)
            result.append(current)
        return np.array(result)

    @caching.cache_decorator
    def convex_hull(self):
        """
        The convex hull of the whole scene

        Returns
        ---------
        hull: Trimesh object, convex hull of all meshes in scene
        """
        points = util.vstack_empty([m.vertices for m in self.dump()])
        hull = convex.convex_hull(points)
        return hull

    @caching.cache_decorator
    def bounding_box(self):
        """
        An axis aligned bounding box for the current scene.

        Returns
        ----------
        aabb: trimesh.primitives.Box object with transform and extents defined
              to represent the axis aligned bounding box of the scene
        """
        from .. import primitives
        center = self.bounds.mean(axis=0)
        aabb = primitives.Box(
            transform=transformations.translation_matrix(center),
            extents=self.extents,
            mutable=False)
        return aabb

    @caching.cache_decorator
    def bounding_box_oriented(self):
        """
        An oriented bounding box for the current mesh.

        Returns
        ---------
        obb : trimesh.primitives.Box
            Box primitive with transform and extents defined
            to represent the minimum volume oriented bounding
            box of the mesh
        """
        from .. import primitives
        to_origin, extents = bounds_module.oriented_bounds(self)
        obb = primitives.Box(transform=np.linalg.inv(to_origin),
                             extents=extents,
                             mutable=False)
        return obb

    def export(self, file_type=None):
        """
        Export a snapshot of the current scene.

        Parameters
        ----------
        file_type: what encoding to use for meshes
                   ie: dict, dict64, stl

        Returns
        ----------
        export: dict with keys:
                meshes: list of meshes, encoded as per file_type
                transforms: edge list of transforms, eg:
                             ((u, v, {'matrix' : np.eye(4)}))
        """

        file_type = str(file_type).strip().lower()
        if file_type == 'gltf':
            return gltf.export_gltf(self)
        elif file_type == 'glb':
            return gltf.export_glb(self)

        export = {'graph': self.graph.to_edgelist(),
                  'geometry': {},
                  'scene_cache': {'bounds': self.bounds.tolist(),
                                  'extents': self.extents.tolist(),
                                  'centroid': self.centroid.tolist(),
                                  'scale': self.scale}}

        if file_type is None:
            file_type = {'Trimesh': 'ply',
                         'Path2D': 'dxf'}

        # if the mesh has an export method use it
        # otherwise put the mesh
        # itself into the export object
        for geometry_name, geometry in self.geometry.items():
            if hasattr(geometry, 'export'):
                if isinstance(file_type, dict):
                    # case where we have export types that are different
                    # for different classes of objects.
                    for query_class, query_format in file_type.items():
                        if util.is_instance_named(geometry, query_class):
                            export_type = query_format
                            break
                else:
                    # if file_type is not a dict, try to export everything in the
                    # scene as that value (probably a single string, like
                    # 'ply')
                    export_type = file_type
                exported = {'data': geometry.export(file_type=export_type),
                            'file_type': export_type}
                export['geometry'][geometry_name] = exported
            else:
                # case where mesh object doesn't have exporter
                # might be that someone replaced the mesh with a URL
                export['geometry'][geometry_name] = geometry
        return export

    def save_image(self, resolution=(1024, 768), **kwargs):
        """
        Get a PNG image of a scene.

        Parameters
        -----------
        resolution: (2,) int, resolution to render image
        **kwargs:  passed to SceneViewer constructor

        Returns
        -----------
        png: bytes, render of scene in PNG form
        """
        from .viewer import render_scene
        png = render_scene(scene=self,
                           resolution=resolution,
                           **kwargs)
        return png

    def convert_units(self, desired, guess=False):
        """
        If geometry has units defined, convert them to new units.

        Returns a new scene with geometries and transforms scaled.

        Parameters
        ----------
        units: str, target unit system. EG 'inches', 'mm', etc
        """
        # if there is no geometry do nothing
        if len(self.geometry) == 0:
            return self

        existing = [i.units for i in self.geometry.values()]
        if any(existing[0] != e for e in existing):
            # if all of our geometry doesn't have the same units already
            # this function will only do some hot nonsense
            raise ValueError('Models in scene have inconsistent units!')

        current = existing[0]
        if current is None:
            # will raise ValueError if not in metadata
            # and not allowed to guess
            current = units.units_from_metadata(self, guess=guess)

        # find the float conversion
        scale = units.unit_conversion(current=current,
                                      desired=desired)

        # exit early if our current units are the same as desired units
        if np.isclose(scale, 1.0):
            result = self.copy()
        else:
            result = self.scaled(scale=scale)

        for geometry in result.geometry.values():
            geometry.units = desired

        return result

    def explode(self, vector=None, origin=None):
        """
        Explode a scene around a point and vector.

        Parameters
        -----------
        vector: (3,) float, or float, explode in a direction or spherically
        origin: (3,) float, point to explode around
        """
        if origin is None:
            origin = self.centroid
        if vector is None:
            vector = self.scale / 25.0

        vector = np.asanyarray(vector, dtype=np.float64)
        origin = np.asanyarray(origin, dtype=np.float64)

        for node_name in self.graph.nodes_geometry:
            transform, geometry_name = self.graph[node_name]

            centroid = self.geometry[geometry_name].centroid
            # transform centroid into nodes location
            centroid = np.dot(transform,
                              np.append(centroid, 1))[:3]

            if vector.shape == ():
                # case where our vector is a single number
                offset = (centroid - origin) * vector
            elif np.shape(vector) == (3,):
                projected = np.dot(vector, (centroid - origin))
                offset = vector * projected
            else:
                raise ValueError('explode vector wrong shape!')

            transform[0:3, 3] += offset
            self.graph[node_name] = transform

    def scaled(self, scale):
        """
        Return a copy of the current scene, with meshes and scene graph
        transforms scaled to the requested factor.

        Parameters
        -----------
        scale: float, factor to scale meshes and transforms by
        """
        scale = float(scale)
        scale_matrix = np.eye(4) * scale

        transforms = np.array([self.graph[i][0]
                               for i in self.graph.nodes_geometry])
        geometries = np.array([self.graph[i][1]
                               for i in self.graph.nodes_geometry])

        result = self.copy()
        result.graph.clear()

        for group in grouping.group(geometries):
            geometry = geometries[group[0]]
            original = transforms[group[0]]
            new_geom = np.dot(scale_matrix, original)

            if result.geometry[geometry].vertices.shape[1] == 2:
                # if our scene is 2D only scale in 2D
                result.geometry[geometry].apply_transform(np.eye(3) * scale)
            else:
                # otherwise apply the full transform
                result.geometry[geometry].apply_transform(new_geom)

            for node, t in zip(self.graph.nodes_geometry[group],
                               transforms[group]):
                transform = util.multi_dot([scale_matrix,
                                            t,
                                            np.linalg.inv(new_geom)])
                transform[:3, 3] *= scale
                result.graph.update(frame_to=node,
                                    matrix=transform,
                                    geometry=geometry)
        return result

    def copy(self):
        """
        Return a deep copy of the current scene

        Returns
        ----------
        copied: trimesh.Scene, copy of the current scene
        """
        # use the geometries copy method to
        # allow them to handle references to unpickle-able objects
        geometry = {n: g.copy() for n, g in self.geometry.items()}
        # create a new scene with copied geometry and graph
        copied = Scene(geometry=geometry,
                       graph=self.graph.copy())
        return copied

    def show(self, viewer=None, **kwargs):
        """
        Display the current scene.

        Parameters
        -----------
        viewer: str 'gl':       open a pyglet window
                str,'notebook': return ipython.display.HTML
                None: automatically pick based on whether or not
                          we are in an ipython notebook
        smooth: bool, turn on or off automatic smooth shading
        """

        if viewer is None:
            # check to see if we are in a notebook or not
            from .viewerJS import in_notebook
            viewer = ['gl', 'notebook'][int(in_notebook())]

        if viewer == 'gl':
            # this imports pyglet, and will raise an ImportError
            # if pyglet is not available
            from .viewer import SceneViewer
            return SceneViewer(self, **kwargs)
        elif viewer == 'notebook':
            from .viewerJS import scene_to_notebook
            return scene_to_notebook(self, **kwargs)
        else:
            raise ValueError('viewer must be "gl", "notebook", or None')

    def __add__(self, other):
        """
        Concatenate the current scene with another scene or mesh.

        Parameters
        ------------
        other : trimesh.Scene, trimesh.Trimesh, trimesh.Path
           Other object to append into the result scene

        Returns
        ------------
        appended : trimesh.Scene
           Scene with geometry from both scenes
        """
        result = append_scenes([self, other],
                               common=[self.graph.base_frame])
        return result


def split_scene(geometry):
    """
    Given a geometry, list of geometries, or a Scene
    return them as a single Scene object.

    Parameters
    ----------
    geometry: splittable

    Returns
    ---------
    scene: trimesh.Scene
    """
    # already a scene, so return it
    if util.is_instance_named(geometry, 'Scene'):
        return geometry

    # a list of things
    if util.is_sequence(geometry):
        metadata = {}
        for g in geometry:
            try:
                metadata.update(g.metadata)
            except BaseException:
                continue
        return Scene(geometry,
                     metadata=metadata)

    # a single geometry so we are going to split
    split = collections.deque()
    metadata = {}
    for g in util.make_sequence(geometry):
        split.extend(g.split())
        metadata.update(g.metadata)
    scene = Scene(split, metadata=metadata)
    return scene


def append_scenes(iterable, common=['world']):
    """
    Concatenate multiple scene objects into one scene.

    Parameters
    -------------
    iterable : (n,) Trimesh or Scene
       Geometries that should be appended
    common : (n,) str
       Nodes that shouldn't be remapped

    Returns
    ------------
    result : trimesh.Scene
       Scene containing all geometry
    """
    if isinstance(iterable, Scene):
        return iterable

    # save geometry in dict
    geometry = {}
    # save transforms as edge tuples
    edges = []

    # nodes which shouldn't be remapped
    common = set(common)
    # nodes which are consumed and need to be remapped
    consumed = set()

    def node_remap(node):
        """
        Remap node to new name if necessary

        Parameters
        -------------
        node : hashable
           Node name in original scene

        Returns
        -------------
        name : hashable
           Node name in concatenated scene
        """

        # if we've already remapped a node use it
        if node in map_node:
            return map_node[node]

        # if a node is consumed and isn't one of the nodes
        # we're going to hold common between scenes remap it
        if node not in common and node in consumed:
            name = str(node) + '-' + util.unique_id().upper()
            map_node[node] = name
            node = name

        # keep track of which nodes have been used
        # in the current scene
        current.add(node)
        return node

    # loop through every geometry
    for s in iterable:
        # allow Trimesh/Path2D geometry to be passed
        if hasattr(s, 'scene'):
            s = s.scene()
        # if we don't have a scene raise an exception
        if not isinstance(s, Scene):
            raise ValueError('{} is not a scene!'.format(
                type(s).__name__))

        # remap geometries if they have been consumed
        map_geom = {}
        for k, v in s.geometry.items():
            # if a geometry already exists add a UUID to the name
            if k in geometry:
                name = str(k) + '-' + util.unique_id().upper()
            else:
                name = k
            # store name mapping
            map_geom[k] = name
            # store geometry with new name
            geometry[name] = v

        # remap nodes and edges so duplicates won't
        # stomp all over each other
        map_node = {}
        # the nodes used in this scene
        current = set()
        for a, b, attr in s.graph.to_edgelist():
            # remap node names from local names
            a, b = node_remap(a), node_remap(b)
            # remap geometry keys
            # if key is not in map_geom it means one of the scenes
            # referred to geometry that doesn't exist
            # rather than crash here we ignore it as the user
            # possibly intended to add in geometries back later
            if 'geometry' in attr and attr['geometry'] in map_geom:
                attr['geometry'] = map_geom[attr['geometry']]
            # save the new edge
            edges.append((a, b, attr))
        # mark nodes from current scene as consumed
        consumed.update(current)

    # add all data to a new scene
    result = Scene()
    result.graph.from_edgelist(edges)
    result.geometry.update(geometry)

    return result
