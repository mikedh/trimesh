import numpy as np
import collections

from .. import util
from .. import units
from .. import convex
from .. import caching
from .. import grouping
from .. import transformations

from .. import bounds as bounds_module

from ..exchange import export
from ..parent import Geometry3D

from . import cameras
from . import lighting

from .transforms import TransformForest


class Scene(Geometry3D):
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
                 graph=None,
                 camera=None,
                 lights=None,
                 camera_transform=None):
        """
        Create a new Scene object.

        Parameters
        -------------
        geometry : Trimesh, Path2D, Path3D PointCloud or list
          Geometry to initially add to the scene
        base_frame : str or hashable
          Name of base frame
        metadata : dict
          Any metadata about the scene
        graph : TransformForest or None
          A passed transform graph to use
        camera : Camera or None
          A passed camera to use
        lights : [trimesh.scene.lighting.Light] or None
          A passed lights to use
        camera_transform : (4, 4) float or None
          Camera transform in the base frame
        """
        # mesh name : Trimesh object
        self.geometry = collections.OrderedDict()

        # create a new graph
        self.graph = TransformForest(base_frame=base_frame)

        # create our cache
        self._cache = caching.Cache(id_function=self.md5)

        # add passed geometry to scene
        self.add_geometry(geometry)

        # hold metadata about the scene
        self.metadata = {}
        self.metadata.update(metadata)

        if graph is not None:
            # if we've been passed a graph override the default
            self.graph = graph

        self.camera = camera
        self.lights = lights
        self.camera_transform = camera_transform

    def apply_transform(self, transform):
        """
        Apply a transform to every geometry in the scene.

        Parameters
        --------------
        transform : (4, 4)
          Homogeneous transformation matrix
        """
        for geometry in self.geometry.values():
            geometry.apply_transform(transform)

    def add_geometry(self,
                     geometry,
                     node_name=None,
                     geom_name=None,
                     parent_node_name=None,
                     transform=None):
        """
        Add a geometry to the scene.

        If the mesh has multiple transforms defined in its
        metadata, they will all be copied into the
        TransformForest of the current scene automatically.

        Parameters
        ----------
        geometry : Trimesh, Path2D, Path3D PointCloud or list
          Geometry to initially add to the scene
        base_frame : str or hashable
          Name of base frame
        metadata : dict
          Any metadata about the scene
        graph : TransformForest or None
          A passed transform graph to use

        Returns
        ----------
        node_name : str
          Name of node in self.graph
        """

        if geometry is None:
            return
        # PointCloud objects will look like a sequence
        elif util.is_sequence(geometry):
            # if passed a sequence add all elements
            return [self.add_geometry(
                geometry=value,
                node_name=node_name,
                geom_name=geom_name,
                parent_node_name=parent_node_name,
                transform=transform) for value in geometry]
        elif isinstance(geometry, dict):
            # if someone passed us a dict of geometry
            for key, value in geometry.items():
                self.add_geometry(value, geom_name=key)
            return
        elif isinstance(geometry, Scene):
            # concatenate current scene with passed scene
            concat = self + geometry
            # replace geometry in-place
            self.geometry.clear()
            self.geometry.update(concat.geometry)
            # replace graph data with concatenated graph
            self.graph.transforms = concat.graph.transforms
            return
        elif not hasattr(geometry, 'vertices'):
            util.log.warning('unknown type ({}) added to scene!'.format(
                type(geometry).__name__))

        # get or create a name to reference the geometry by
        if geom_name is not None:
            # if name is passed use it
            name = geom_name
        elif 'name' in geometry.metadata:
            # if name is in metadata use it
            name = geometry.metadata['name']
        elif 'file_name' in geometry.metadata:
            name = geometry.metadata['file_name']
        else:
            # try to create a simple name
            name = 'geometry_' + str(len(self.geometry))

        # if its already taken add a unique random string to it
        if name in self.geometry:
            name += ':' + util.unique_id().upper()

        # save the geometry reference
        self.geometry[name] = geometry

        # create a unique node name if not passed
        if node_name is None:
            # if the name of the geometry is also a transform node
            if name in self.graph.nodes:
                # a random unique identifier
                unique = util.unique_id(increment=len(self.geometry))
                # geometry name + UUID
                node_name = name + '_' + unique.upper()
            else:
                # otherwise make the transform node name the same as the geom
                node_name = name

        if transform is None:
            # create an identity transform from parent_node
            transform = np.eye(4)

        self.graph.update(frame_to=node_name,
                          frame_from=parent_node_name,
                          matrix=transform,
                          geometry=name,
                          geometry_flags={'visible': True})
        return node_name

    def delete_geometry(self, names):
        """
        Delete one more multiple geometries from the scene and also
        remove any node in the transform graph which references it.

        Parameters
        --------------
        name : hashable
          Name that references self.geometry
        """
        # make sure we have a set we can check
        if util.is_string(names):
            names = [names]
        names = set(names)

        # remove the geometry reference from relevant nodes
        self.graph.remove_geometries(names)
        # remove the geometries from our geometry store
        [self.geometry.pop(name, None) for name in names]

    def md5(self):
        """
        MD5 of scene which will change when meshes or
        transforms are changed

        Returns
        --------
        hashed : str
          MD5 hash of scene
        """
        # start with transforms hash
        return util.md5_object(self._hashable())

    def crc(self):
        return caching.crc32(self._hashable())

    def _hashable(self):
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
        hashable = ''.join(sorted(hashes)).encode('utf-8')
        return hashable

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

    @property
    def is_valid(self):
        """
        Is every geometry connected to the root node.

        Returns
        -----------
        is_valid : bool
          Does every geometry have a transform
        """
        if len(self.geometry) == 0:
            return True

        try:
            referenced = {self.graph[i][1]
                          for i in self.graph.nodes_geometry}
        except BaseException:
            # if connectivity to world frame is broken return false
            return False

        # every geometry is referenced
        ok = referenced == set(self.geometry.keys())

        return ok

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
                        for k, v in self.geometry.items()
                        if v.bounds is not None}
        if len(corners_geom) == 0:
            return np.array([])

        for node_name in self.graph.nodes_geometry:
            # access the transform and geometry name from node
            transform, geometry_name = self.graph[node_name]
            # not all nodes have associated geometry
            if geometry_name not in corners_geom:
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
        bounds : (2, 3) float or None
          Position of [min, max] bounding box
          Returns None if no valid bounds exist
        """
        corners = self.bounds_corners
        if len(corners) == 0:
            return None
        bounds = np.array([corners.min(axis=0),
                           corners.max(axis=0)])
        return bounds

    @caching.cache_decorator
    def extents(self):
        """
        Return the axis aligned box size of the current scene.

        Returns
        ----------
        extents : (3,) float
          Bounding box sides length
        """
        return np.diff(self.bounds, axis=0).reshape(-1)

    @caching.cache_decorator
    def scale(self):
        """
        The approximate scale of the mesh

        Returns
        -----------
        scale : float
          The mean of the bounding box edge lengths
        """
        scale = (self.extents ** 2).sum() ** .5
        return scale

    @caching.cache_decorator
    def centroid(self):
        """
        Return the center of the bounding box for the scene.

        Returns
        --------
        centroid : (3) float
          Point for center of bounding box
        """
        centroid = np.mean(self.bounds, axis=0)
        return centroid

    @caching.cache_decorator
    def area(self):
        """
        What is the summed area of every geometry which
        has area.

        Returns
        ------------
        area : float
          Summed area of every instanced geometry
        """
        # get the area of every geometry that has an area property
        areas = {n: g.area for n, g in self.geometry.items()
                 if hasattr(g, 'area')}
        # get the name of every geometry instance in the scene
        geoms = [self.graph[n][1] for n in
                 self.graph.nodes_geometry]
        # sum the area for every instanced geometry
        area = sum(areas[n] for n in geoms if n in geoms)
        return area

    @caching.cache_decorator
    def triangles(self):
        """
        Return a correctly transformed polygon soup of the
        current scene.

        Returns
        ----------
        triangles : (n, 3, 3) float
          Triangles in space
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
        populate = self.triangles  # NOQA
        return self._cache['triangles_node']

    @caching.cache_decorator
    def geometry_identifiers(self):
        """
        Look up geometries by identifier MD5

        Returns
        ---------
        identifiers : dict
          {Identifier MD5: key in self.geometry}
        """
        identifiers = {mesh.identifier_md5: name
                       for name, mesh in self.geometry.items()}
        return identifiers

    @caching.cache_decorator
    def duplicate_nodes(self):
        """
        Return a sequence of node keys of identical meshes.

        Will include meshes with different geometry but identical
        spatial hashes as well as meshes repeated by self.nodes.

        Returns
        -----------
        duplicates : (m) sequenc
          Keys of self.nodes that represent identical geometry
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
        # sequence of node names where each
        # sublist has identical geometry
        duplicates = [np.sort(node_names[g]).tolist()
                      for g in node_groups]
        return duplicates

    def deduplicated(self):
        """
        Return a new scene where each unique geometry is only
        included once and transforms are discarded.

        Returns
        -------------
        dedupe : Scene
          One copy of each unique geometry from scene
        """
        # collect geometry
        geometry = {}
        # loop through groups of identical nodes
        for group in self.duplicate_nodes:
            # get the name of the geometry
            name = self.graph[group[0]][1]
            # collect our unique collection of geometry
            geometry[name] = self.geometry[name]

        return Scene(geometry)

    def set_camera(self,
                   angles=None,
                   distance=None,
                   center=None,
                   resolution=None,
                   fov=None):
        """
        Create a camera object for self.camera, and add
        a transform to self.graph for it.

        If arguments are not passed sane defaults will be figured
        out which show the mesh roughly centered.

        Parameters
        -----------
        angles : (3,) float
          Initial euler angles in radians
        distance : float
          Distance from centroid
        center : (3,) float
          Point camera should be center on
        camera : Camera object
          Object that stores camera parameters
        """

        if fov is None:
            fov = np.array([60, 45])

        # if no geometry nothing to set camera to
        if len(self.geometry) == 0:
            self._camera = cameras.Camera(fov=fov)
            self.graph[self._camera.name] = np.eye(4)
            return self._camera
        # set with no rotation by default
        if angles is None:
            angles = np.zeros(3)

        rotation = transformations.euler_matrix(*angles)
        transform = cameras.look_at(
            self.bounds_corners,
            fov=fov,
            rotation=rotation,
            distance=distance,
            center=center)

        if hasattr(self, '_camera') and self._camera is not None:
            self._camera.fov = fov
            if resolution is not None:
                self._camera.resolution = resolution
        else:
            # create a new camera object
            self._camera = cameras.Camera(fov=fov, resolution=resolution)

        self.graph[self._camera.name] = transform

        return self._camera

    @property
    def camera_transform(self):
        """
        Get camera transform in the base frame

        Returns
        -------
        camera_transform : (4, 4) float
          Camera transform in the base frame
        """
        return self.graph[self.camera.name][0]

    def camera_rays(self):
        """
        Calculate the trimesh.scene.Camera origin and ray
        direction vectors. Returns one ray per pixel as set
        in camera.resolution

        Returns
        --------------
        origin: (n, 3) float
          Ray origins in space
        vectors: (n, 3) float
          Ray direction unit vectors in world coordinates
        pixels : (n, 2) int
          Which pixel does each ray correspond to in an image
        """
        # get the unit vectors of the camera
        vectors, pixels = self.camera.to_rays()
        # find our scene's transform for the camera
        transform = self.camera_transform
        # apply the rotation to the unit ray direction vectors
        vectors = transformations.transform_points(
            vectors,
            transform,
            translate=False)
        # camera origin is single point so extract from
        origins = (np.ones_like(vectors) *
                   transformations.translation_from_matrix(transform))
        return origins, vectors, pixels

    @camera_transform.setter
    def camera_transform(self, camera_transform):
        """
        Set the camera transform in the base frame

        Parameters
        ----------
        camera_transform : (4, 4) float
          Camera transform in the base frame
        """
        if camera_transform is None:
            return
        self.graph[self.camera.name] = camera_transform

    @property
    def camera(self):
        """
        Get the single camera for the scene. If not manually
        set one will abe automatically generated.

        Returns
        ----------
        camera : trimesh.scene.Camera
          Camera object defined for the scene
        """
        # no camera set for the scene yet
        if not self.has_camera:
            # will create a camera with everything in view
            return self.set_camera()
        assert self._camera is not None

        return self._camera

    @camera.setter
    def camera(self, camera):
        """
        Set a camera object for the Scene.

        Parameters
        -----------
        camera : trimesh.scene.Camera
          Camera object for the scene
        """
        if camera is None:
            return
        self._camera = camera

    @property
    def has_camera(self):
        return hasattr(self, '_camera') and self._camera is not None

    @property
    def lights(self):
        """
        Get a list of the lights in the scene. If nothing is
        set it will generate some automatically.

        Returns
        -------------
        lights : [trimesh.scene.lighting.Light]
          Lights in the scene.
        """
        if not hasattr(self, '_lights') or self._lights is None:
            # do some automatic lighting
            lights, transforms = lighting.autolight(self)
            # assign the transforms to the scene graph
            for L, T in zip(lights, transforms):
                self.graph[L.name] = T
            # set the lights
            self._lights = lights
        return self._lights

    @lights.setter
    def lights(self, lights):
        """
        Assign a list of light objects to the scene

        Parameters
        --------------
        lights : [trimesh.scene.lighting.Light]
          Lights in the scene.
        """
        self._lights = lights

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

    def dump(self, concatenate=False):
        """
        Append all meshes in scene freezing transforms.

        Parameters
        ------------
        concatenate : bool
          If True, concatenate results into single mesh

        Returns
        ----------
        dumped : (n,) Trimesh or Trimesh
          Trimesh objects transformed to their
          location the scene.graph
        """
        result = []
        for node_name in self.graph.nodes_geometry:
            transform, geometry_name = self.graph[node_name]
            # get a copy of the geometry
            current = self.geometry[geometry_name].copy()
            # move the geometry vertices into the requested frame
            current.apply_transform(transform)
            current.metadata['name'] = node_name
            # save to our list of meshes
            result.append(current)

        if concatenate:
            return util.concatenate(result)

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

    def export(self,
               file_obj=None,
               file_type=None,
               **kwargs):
        """
        Export a snapshot of the current scene.

        Parameters
        ----------
        file_obj : str, file-like, or None
          File object to export to
        file_type : str or None
          What encoding to use for meshes
          IE: dict, dict64, stl

        Returns
        ----------
        export : bytes
          Only returned if file_obj is None
        """
        return export.export_scene(
            scene=self,
            file_obj=file_obj,
            file_type=file_type,
            **kwargs)

    def save_image(self, resolution=None, **kwargs):
        """
        Get a PNG image of a scene.

        Parameters
        -----------
        resolution : (2,) int
          Resolution to render image
        **kwargs
          Passed to SceneViewer constructor

        Returns
        -----------
        png : bytes
          Render of scene as a PNG
        """
        from ..viewer import render_scene
        png = render_scene(scene=self,
                           resolution=resolution,
                           **kwargs)
        return png

    @property
    def units(self):
        """
        Get the units for every model in the scene, and
        raise a ValueError if there are mixed units.

        Returns
        -----------
        units : str
          Units for every model in the scene
        """
        existing = [i.units for i in self.geometry.values()]

        if any(existing[0] != e for e in existing):
            # if all of our geometry doesn't have the same units already
            # this function will only do some hot nonsense
            raise ValueError('models in scene have inconsistent units!')

        return existing[0]

    @units.setter
    def units(self, value):
        """
        Set the units for every model in the scene without
        converting any units just setting the tag.

        Parameters
        ------------
        value : str
          Value to set every geometry unit value to
        """
        for m in self.geometry.values():
            m.units = value

    def convert_units(self, desired, guess=False):
        """
        If geometry has units defined convert them to new units.

        Returns a new scene with geometries and transforms scaled.

        Parameters
        ----------
        desired : str
          Desired final unit system: 'inches', 'mm', etc.
        guess : bool
          Is the converter allowed to guess scale when models
          don't have it specified in their metadata.

        Returns
        ----------
        scaled : trimesh.Scene
          Copy of scene with scaling applied and units set
          for every model
        """
        # if there is no geometry do nothing
        if len(self.geometry) == 0:
            return self.copy()

        current = self.units
        if current is None:
            # will raise ValueError if not in metadata
            # and not allowed to guess
            current = units.units_from_metadata(self, guess=guess)

        # find the float conversion
        scale = units.unit_conversion(
            current=current,
            desired=desired)

        # exit early if our current units are the same as desired units
        if np.isclose(scale, 1.0):
            result = self.copy()
        else:
            result = self.scaled(scale=scale)

        # apply the units to every geometry of the scaled result
        result.units = desired

        return result

    def explode(self, vector=None, origin=None):
        """
        Explode a scene around a point and vector.

        Parameters
        -----------
        vector : (3,) float or float
           Explode radially around a direction vector or spherically
        origin : (3,) float
          Point to explode around
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

            transform[:3, 3] += offset
            self.graph[node_name] = transform

    def scaled(self, scale):
        """
        Return a copy of the current scene, with meshes and scene
        transforms scaled to the requested factor.

        Parameters
        -----------
        scale : float
          Factor to scale meshes and transforms

        Returns
        -----------
        scaled : trimesh.Scene
          A copy of the current scene but scaled
        """
        scale = float(scale)
        # matrix for 2D scaling
        scale_2D = np.eye(3) * scale
        # matrix for 3D scaling
        scale_3D = np.eye(4) * scale

        # preallocate transforms and geometries
        nodes = np.array(self.graph.nodes_geometry)
        transforms = np.zeros((len(nodes), 4, 4))
        geometries = [None] * len(nodes)

        # collect list of transforms
        for i, node in enumerate(nodes):
            transforms[i], geometries[i] = self.graph[node]

        # result is a copy
        result = self.copy()
        # remove all existing transforms
        result.graph.clear()

        for group in grouping.group(geometries):
            # hashable reference to self.geometry
            geometry = geometries[group[0]]
            # original transform from world to geometry
            original = transforms[group[0]]
            # transform for geometry
            new_geom = np.dot(scale_3D, original)

            if result.geometry[geometry].vertices.shape[1] == 2:
                # if our scene is 2D only scale in 2D
                result.geometry[geometry].apply_transform(scale_2D)
            else:
                # otherwise apply the full transform
                result.geometry[geometry].apply_transform(new_geom)

            for node, T in zip(nodes[group],
                               transforms[group]):
                # generate the new transforms
                transform = util.multi_dot(
                    [scale_3D, T, np.linalg.inv(new_geom)])
                # apply scale to translation
                transform[:3, 3] *= scale
                # update scene with new transforms
                result.graph.update(frame_to=node,
                                    matrix=transform,
                                    geometry=geometry)
        return result

    def copy(self):
        """
        Return a deep copy of the current scene

        Returns
        ----------
        copied : trimesh.Scene
          Copy of the current scene
        """
        # use the geometries copy method to
        # allow them to handle references to unpickle-able objects
        geometry = {n: g.copy() for n, g in self.geometry.items()}

        if not hasattr(self, '_camera') or self._camera is None:
            # if no camera set don't include it
            camera = None
        else:
            # otherwise get a copy of the camera
            camera = self.camera.copy()
        # create a new scene with copied geometry and graph
        copied = Scene(geometry=geometry,
                       graph=self.graph.copy(),
                       metadata=self.metadata.copy(),
                       camera=camera)
        return copied

    def show(self, viewer=None, **kwargs):
        """
        Display the current scene.

        Parameters
        -----------
        viewer: str
          What kind of viewer to open, including
          'gl' to open a pyglet window, 'notebook'
          for a jupyter notebook or None
        kwargs : dict
          Includes `smooth`, which will turn
          on or off automatic smooth shading
        """

        if viewer is None:
            # check to see if we are in a notebook or not
            from ..viewer import in_notebook
            viewer = 'gl'
            if in_notebook():
                viewer = 'notebook'

        if viewer == 'gl':
            # this imports pyglet, and will raise an ImportError
            # if pyglet is not available
            from ..viewer import SceneViewer
            return SceneViewer(self, **kwargs)
        elif viewer == 'notebook':
            from ..viewer import scene_to_notebook
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


def split_scene(geometry, **kwargs):
    """
    Given a geometry, list of geometries, or a Scene
    return them as a single Scene object.

    Parameters
    ----------
    geometry : splittable

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
    split = []
    metadata = {}
    for g in util.make_sequence(geometry):
        split.extend(g.split(**kwargs))
        metadata.update(g.metadata)

    # if there is only one geometry in the mesh
    # name it from the file name
    if len(split) == 1 and 'file_name' in metadata:
        split = {metadata['file_name']: split[0]}

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
