import numpy as np
import collections

from .. import util
from .. import grouping
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
                 node=None,
                 base_frame='world'):

        # instance name : mesh name
        self.nodes = {}
        self._cache = util.Cache(id_function=self.md5)

        # mesh name : Trimesh object
        self.geometry = collections.OrderedDict()
        self.flags = {}
        self.transforms = TransformForest(base_frame=base_frame)

        if node is not None:
            self.add_geometry(node)
            self.set_camera()
        self._cache.id_set()

    def add_geometry(self, geometry):
        '''
        Add a geometry to the scene.

        If the mesh has multiple transforms defined in its metadata, they will
        all be copied into the TransformForest of the current scene automatically.

        Arguments
        ----------
        geometry: Trimesh, Path3D, or list of same
        '''
        if util.is_sequence(geometry):
            return [self.add_geometry(i) for i in geometry]

        # default values for transforms and name
        transforms = np.eye(4).reshape((-1, 4, 4))
        name = 'geometry_' + str(len(self.geometry))

        if hasattr(geometry, 'metadata'):
            if 'name' in geometry.metadata:
                name = geometry.metadata['name']
            if 'transforms' in geometry.metadata:
                transforms = np.asanyarray(geometry.metadata['transforms'])
                transforms = transforms.reshape((-1, 4, 4))

        self.geometry[name] = geometry
        for i, transform in enumerate(transforms):
            name_node = name + '_' + str(i)
            self.nodes[name_node] = name
            self.flags[name_node] = {'visible': True}
            self.transforms.update(frame_to=name_node,
                                   matrix=transform)

    def md5(self):
        '''
        MD5 of scene, which will change when meshes or transforms are changed
        '''

        data = [hash(i) for i in self.geometry.values()]
        data.append(self.transforms.md5())
        hashed = util.md5_object(np.sort(data))

        return hashed

    @util.cache_decorator
    def bounds(self):
        '''
        Return the overall bounding box of the scene.

        Returns
        --------
        bounds: (2,3) float points for min, max corner
        '''
        # store the indexes for all 8 corners of a cube,
        # given an input of flattened min/max bounds
        minx, miny, minz, maxx, maxy, maxz = np.arange(6)
        corner_index = [minx, miny, minz,
                        maxx, miny, minz,
                        maxx, maxy, minz,
                        minx, maxy, minz,
                        minx, miny, maxz,
                        maxx, miny, maxz,
                        maxx, maxy, maxz,
                        minx, maxy, maxz]

        corners = collections.deque()
        for instance, mesh_name in self.nodes.items():
            transform = self.transforms.get(instance)
            current_bounds = self.geometry[mesh_name].bounds
            # handle 2D bounds
            if current_bounds.shape == (2, 2):
                current_bounds = np.column_stack((current_bounds, [0, 0]))
            current_corners = current_bounds.reshape(
                -1)[corner_index].reshape((-1, 3))
            corners.extend(transform_points(current_corners,
                                            transform))
        corners = np.array(corners)
        bounds = np.array([corners.min(axis=0),
                           corners.max(axis=0)])
        return bounds

    @util.cache_decorator
    def extents(self):
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
        triangles_index = collections.deque()
        for index, node_info in zip(range(len(self.nodes)),
                                    self.nodes.items()):
            node, geometry_name = node_info
            geometry = self.geometry[geometry_name]
            if not hasattr(geometry, 'triangles'):
                continue
            transform = self.transforms.get(node)
            triangles.append(transform_points(geometry.triangles.reshape((-1, 3)),
                                              transform))
            triangles_index.extend(np.ones(len(geometry.triangles),
                                           dtype=np.int64) * index)
        self._cache['triangles_index'] = np.array(
            triangles_index, dtype=np.int64)
        triangles = np.vstack(triangles).reshape((-1, 3, 3))
        return triangles

    @util.cache_decorator
    def triangles_index(self):
        '''
        Which index of self.nodes.values() does each triangle come from.

        Returns
        ---------
        triangles_index: (len(self.triangles),) int, index of self.nodes.values()
        '''
        populate = self.triangles
        return self._cache['triangles_index']

    def duplicate_nodes(self):
        '''
        Return a sequence of node keys of identical meshes.

        Will combine meshes duplicated by copying in space with different keys in
        self.geometry, as well as meshes repeated by self.nodes.

        Returns
        -----------
        duplicates: (m) sequence of keys to self.nodes that represent identical geometry
        '''
        mesh_ids = {k: int(m.identifier_md5, 16)
                    for k, m in self.geometry.items()}
        node_ids = np.array([mesh_ids[v] for v in self.nodes.values()])

        node_groups = grouping.group(node_ids)

        node_keys = np.array(list(self.nodes.keys()))
        duplicates = [np.sort(node_keys[g]).tolist() for g in node_groups]
        return duplicates

    def set_camera(self, angles=None, distance=None, center=None):
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

        self.transforms.update(frame_from='camera',
                               frame_to=self.transforms.base_frame,
                               matrix=transform)

    def dump(self):
        '''
        Append all meshes in scene to a list of meshes.
        '''
        result = collections.deque()
        for node_id, mesh_id in self.nodes.items():
            transform = self.transforms.get(node_id)
            current = self.geometry[mesh_id].copy()
            current.apply_transform(transform)
            result.append(current)
        return np.array(result)

    def export(self, file_type=None):
        '''
        Export a snapshot of the current scene.

        Arguments
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
        export = {'transforms': self.transforms.to_flattened(),
                  'nodes': self.nodes,
                  'geometry': {},
                  'scene_cache': self._cache.cache}

        if file_type is None:
            file_type = {'Trimesh': 'ply',
                         'Path2D': 'dxf'}

        # if the mesh has an export method use it, otherwise put the mesh
        # itself into the export object
        for node, geometry in self.geometry.items():
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

                export['geometry'][node] = {'bytes': geometry.export(file_type=export_type),
                                            'file_type': export_type}
            else:
                # case where mesh object doesn't have exporter
                # might be that someone replaced the mesh with a URL
                export['geometry'][node] = geometry
        return export

    def save_image(self, file_obj, resolution=(1024, 768), **kwargs):
        from .viewer import SceneViewer
        SceneViewer(self,
                    save_image=file_obj,
                    resolution=resolution,
                    **kwargs)

    def explode(self, vector=[0.0, 0.0, 1.0], origin=None):
        '''
        Explode a scene around a point and vector.
        '''
        if origin is None:
            origin = self.centroid
        centroids = np.array(
            [self.geometry[i].centroid for i in self.nodes.values()])

        if np.shape(vector) == (3,):
            vectors = np.tile(vector, (len(centroids), 1))
            projected = np.dot(vector, (centroids - origin).T)
            offsets = vectors * projected.reshape((-1, 1))
        elif isinstance(vector, float):
            offsets = (centroids - origin) * vector
        else:
            raise ValueError('Explode vector must by (3,) or float')

        for offset, node_key in zip(offsets, self.nodes.keys()):
            current = self.transforms[node_key]
            current[0:3, 3] += offset
            self.transforms[node_key] = current

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

    Arguments
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
