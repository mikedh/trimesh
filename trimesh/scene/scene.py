import numpy as np
import collections
import copy

from .. import util
from .. import grouping

from ..bounds import corners as bounds_corners
from ..transformations import rotation_matrix, transform_points
from .transforms import TransformForest


class Scene:
    '''
    A simple scene graph which can be rendered directly via pyglet/openGL,
    or through other endpoints such as a raytracer.

    Meshes and lights are added by name, which can then be moved by updating
    transform in the transform tree.
    '''

    def __init__(self,
                 geometry=None,
                 base_frame='world',
                 metadata={}):

        # mesh name : Trimesh object
        self.geometry = collections.OrderedDict()

        # graph structure of instances
        self.graph = TransformForest(base_frame=base_frame)

        self._cache = util.Cache(id_function=self.md5)

        if geometry is not None:
            self.add_geometry(geometry)
            self.set_camera()

        self.metadata = {}
        self.metadata.update(metadata)

    def add_geometry(self,
                     geometry,
                     node_name=None):
        '''
        Add a geometry to the scene.

        If the mesh has multiple transforms defined in its metadata, they will
        all be copied into the TransformForest of the current scene automatically.

        Parameters
        ----------
        geometry: Trimesh, Path3D, or list of same
        node_name: 
        Returns
        ----------
        node_name: str, name of node in self.graph
        '''

        # if passed a sequence call add_geometry on all elements
        if util.is_sequence(geometry):
            return [self.add_geometry(i) for i in geometry]

        # default values for transforms and name
        transforms = np.eye(4).reshape((-1, 4, 4))
        geometry_name = 'geometry_' + str(len(self.geometry))

        # if object has metadata indicating different transforms or name
        # use those values
        if hasattr(geometry, 'metadata'):
            if 'name' in geometry.metadata:
                geometry_name = geometry.metadata['name']

            if 'transforms' in geometry.metadata:
                transforms = np.asanyarray(geometry.metadata['transforms'])
                transforms = transforms.reshape((-1, 4, 4))

        # save the geometry reference
        self.geometry[geometry_name] = geometry

        for i, transform in enumerate(transforms):

            # if we haven't been passed a name to set in the graph
            # use the geometry name plus an index
            if node_name is None:
                node_name = geometry_name + '_' + str(i)

            self.graph.update(frame_to=node_name,
                              matrix=transform,
                              geometry=geometry_name,
                              geometry_flags={'visible': True})

    def md5(self):
        '''
        MD5 of scene, which will change when meshes or transforms are changed

        Returns
        --------
        hashed: str, MD5 hash of scene
        '''

        # get the MD5 of geometry and graph
        data = [i.md5() for i in self.geometry.values()]
        hashed = util.md5_object(np.append(data, self.graph.md5()))

        return hashed

    @util.cache_decorator
    def bounds(self):
        '''
        Return the overall bounding box of the scene.

        Returns
        --------
        bounds: (2,3) float points for min, max corner
        '''
        corners = collections.deque()

        for node_name in self.graph.nodes_geometry:
            # access the transform and geometry name for every node
            transform, geometry_name = self.graph[node_name]

            # not all nodes have associated geometry
            if geometry_name is None:
                continue

            # geometry objects have bounds properties, which are (2,3) or (2,2)
            current_bounds = self.geometry[geometry_name].bounds.copy()
            # find the 8 corner vertices of the axis aligned bounding box
            current_corners = bounds_corners(current_bounds)
            # transform those corners into where the geometry is located
            corners.extend(transform_points(current_corners,
                                            transform))
        corners = np.array(corners)
        bounds = np.array([corners.min(axis=0),
                           corners.max(axis=0)])
        return bounds

    @util.cache_decorator
    def extents(self):
        '''
        Return the axis aligned box size of the current scene.

        Returns
        ----------
        extents: (3,) float, bounding box sides length
        '''
        return np.diff(self.bounds, axis=0).reshape(-1)

    @util.cache_decorator
    def scale(self):
        '''
        The approximate scale of the mesh

        Returns
        -----------
        scale: float, the mean of the bounding box edge lengths
        '''
        return self.extents.mean()

    @util.cache_decorator
    def centroid(self):
        '''
        Return the center of the bounding box for the scene.

        Returns
        --------
        centroid: (3) float point for center of bounding box
        '''
        centroid = np.mean(self.bounds, axis=0)
        return centroid

    @util.cache_decorator
    def triangles(self):
        '''
        Return a correctly transformed polygon soup of the current scene.

        Returns
        ----------
        triangles: (n,3,3) float, triangles in space
        '''
        triangles = collections.deque()
        triangles_node = collections.deque()

        for node_name in self.graph.nodes_geometry:

            transform, geometry_name = self.graph[node_name]

            geometry = self.geometry[geometry_name]
            if not hasattr(geometry, 'triangles'):
                continue

            triangles.append(transform_points(geometry.triangles.copy().reshape((-1, 3)),
                                              transform))
            triangles_node.append(np.tile(node_name, len(geometry.triangles)))

        self._cache['triangles_node'] = np.hstack(triangles_node)
        triangles = np.vstack(triangles).reshape((-1, 3, 3))
        return triangles

    @util.cache_decorator
    def triangles_node(self):
        '''
        Which node of self.graph does each triangle come from.

        Returns
        ---------
        triangles_index: (len(self.triangles),) node name for each triangle
        '''
        populate = self.triangles
        return self._cache['triangles_node']

    @util.cache_decorator
    def duplicate_nodes(self):
        '''
        Return a sequence of node keys of identical meshes.

        Will combine meshes duplicated by copying in space with different keys in
        self.geometry, as well as meshes repeated by self.nodes.

        Returns
        -----------
        duplicates: (m) sequence of keys to self.nodes that represent identical geometry
        '''

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

    def set_camera(self, angles=None, distance=None, center=None):
        '''
        Add a transform to self.graph for 'camera'

        If arguments are not passed sane defaults will be figured out.

        Parameters
        -----------
        angles:    (3,) float, initial euler angles in radians
        distance:  float, distance away camera should be
        center:    (3,) float, point camera should center on

        '''
        if center is None:
            center = self.centroid
        if distance is None:
            distance = self.extents.max()
        if angles is None:
            angles = np.zeros(3)

        translation = np.eye(4)
        translation[0:3, 3] = center
        translation[2][3] += distance * 1.5

        transform = np.dot(rotation_matrix(angles[0], [1, 0, 0], point=center),
                           rotation_matrix(angles[1], [0, 1, 0], point=center))
        transform = np.dot(transform, translation)

        self.graph.update(frame_from='camera',
                          frame_to=self.graph.base_frame,
                          matrix=transform)

    def dump(self):
        '''
        Append all meshes in scene to a list of meshes.
        '''
        result = collections.deque()

        for node_name in self.graph.nodes_geometry:
            transform, geometry_name = self.graph[node_name]

            current = self.geometry[geometry_name].copy()
            current.apply_transform(transform)
            result.append(current)
        return np.array(result)

    def export(self, file_type=None):
        '''
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
        '''
        export = {'graph': self.graph.to_edgelist(),
                  'geometry': {},
                  'scene_cache': {'bounds': self.bounds.tolist(),
                                  'extents': self.extents.tolist(),
                                  'centroid': self.centroid.tolist(),
                                  'scale': self.scale}}

        if file_type is None:
            file_type = {'Trimesh': 'ply',
                         'Path2D': 'dxf'}

        # if the mesh has an export method use it, otherwise put the mesh
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

    def save_image(self, file_obj, resolution=(1024, 768), **kwargs):
        from .viewer import SceneViewer
        SceneViewer(self,
                    save_image=file_obj,
                    resolution=resolution,
                    **kwargs)

    def explode(self, vector=None, origin=None):
        '''
        Explode a scene around a point and vector.
        '''
        if origin is None:
            origin = self.centroid
        if vector is None:
            vector = self.scale / 25.0

        centroids = collections.deque()
        for node_name in self.graph.nodes_geometry:
            transform, geometry_name = self.graph[node_name]

            centroid = self.geometry[geometry_name].centroid
            # transform centroid into nodes location
            centroid = np.dot(transform,
                              np.append(centroid, 1))[:3]

            if (isinstance(vector, float) or
                    isinstance(vector, int)):
                offset = (centroid - origin) * vector
            elif np.shape(vector) == (3,):
                projected = np.dot(vector, (centroid - origin))
                offset = vector * projected
            else:
                raise ValueError('vector wrong shape')

            transform[0:3, 3] += offset
            self.graph[node_name] = transform

    def show(self, block=True, **kwargs):
        # this imports pyglet, and will raise an ImportError
        # if pyglet is not available
        from .viewer import SceneViewer

        def viewer():
            SceneViewer(self, **kwargs)

        if block:
            viewer()
        else:
            from threading import Thread
            Thread(target=viewer, kwargs=kwargs).start()


def split_scene(geometry):
    '''
    Given a possible sequence of geometries, decompose them into parts.

    Parameters
    ----------
    geometry: splittable

    Returns
    ---------
    scene: trimesh.Scene
    '''
    split = collections.deque()
    for g in util.make_sequence(geometry):
        split.extend(g.split())
    scene = Scene(split)
    return scene
