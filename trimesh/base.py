"""
github.com/mikedh/trimesh
----------------------------

Library for importing, exporting and doing simple operations on triangular meshes.
"""

import numpy as np

import copy

from . import util
from . import units
from . import poses
from . import graph
from . import voxel
from . import visual
from . import sample
from . import repair
from . import convex
from . import remesh
from . import bounds
from . import inertia
from . import nsphere
from . import boolean
from . import grouping
from . import geometry
from . import permutate
from . import proximity
from . import triangles
from . import collision
from . import comparison
from . import decomposition
from . import intersections
from . import transformations


from .io.export import export_mesh
from .ray import ray_triangle
from .constants import log, _log_time, tol
from .scene import Scene

try:
    # optionally load an interface to the embree raytracer
    from .ray import ray_pyembree
    _has_embree = True
except ImportError:
    _has_embree = False


class Trimesh(object):

    def __init__(self,
                 vertices=None,
                 faces=None,
                 face_normals=None,
                 vertex_normals=None,
                 face_colors=None,
                 vertex_colors=None,
                 metadata=None,
                 process=True,
                 validate=False,
                 use_embree=True,
                 initial_cache={},
                 **kwargs):
        """
        A Trimesh object contains a triangular 3D mesh.

        Parameters
        ----------
        vertices:       (n,3) float set of vertex locations

        faces:          (m,3) int set of triangular faces
                              (quad faces will be triangulated)

        face_normals:   (m,3) float set of normal vectors for faces.

        vertex_normals: (n,3) float set of normal vectors for vertices

        metadata:       dict, any metadata about the mesh

        process:        bool, if True, Nan and Inf values will be removed
                        immediatly and vertices will be merged

        validate:       bool, if True, degenerate and duplicate faces will be
                        removed immediatly, and some functions will alter
                        the mesh to ensure consistent results.

        use_embree:     bool, if True try to use pyembree raytracer.
                        If pyembree is not available it will automatically fall
                        back to a much slower rtree/numpy implementation

        initial_cache:  dict, a way to pass things to the cache in case expensive
                        things were calculated before creating the mesh object.

        **kwargs:       stored in self._kwargs if needed later
        """

        # self._data stores information about the mesh which
        # CANNOT be regenerated.
        # in the base class all that is stored here is vertex and
        # face information
        # any data put into the store is converted to a TrackedArray
        # which is a subclass of np.ndarray that provides md5 and crc
        # methods which can be used to detect changes in the array.
        self._data = util.DataStore()

        # self._cache stores information about the mesh which CAN be
        # regenerated from self._data, but may be slow to calculate.
        # In order to maintain consistency
        # the cache is cleared when self._data.crc() changes
        self._cache = util.Cache(id_function=self._data.crc)
        self._cache.update(initial_cache)

        # if validate we are allowed to alter the mesh silently
        # to ensure valid results
        self._validate = bool(validate)

        # check for None only to avoid warning messages in subclasses
        if vertices is not None:
            # (n, 3) float, set of vertices
            self.vertices = vertices
        if faces is not None:
            # (m, 3) int of triangle faces, references self.vertices
            self.faces = faces

        # hold visual information about the mesh (vertex and face colors)
        if 'visual' in kwargs:
            self.visual = kwargs['visual']
        else:
            self.visual = visual.create_visual(face_colors=face_colors,
                                               vertex_colors=vertex_colors,
                                               mesh=self,
                                               **kwargs)
        # add a back reference to this mesh for its visual property object
        self.visual.mesh = self

        # normals are accessed through setters/properties and are regenerated
        # if dimensions are inconsistant, but can be set by the constructor
        # to avoid a substantial number of cross products
        if face_normals is not None:
            self.face_normals = face_normals

        # (n, 3) float of vertex normals, can be created from face normals
        if vertex_normals is not None:
            self.vertex_normals = vertex_normals

        # embree is a much, much faster raytracer written by Intel
        # if you have pyembree installed you should use it
        # although both raytracers were designed to have a common API
        if _has_embree and use_embree:
            self.ray = ray_pyembree.RayMeshIntersector(self)
        else:
            # create a ray-mesh query object for the current mesh
            # initializing is very inexpensive and object is convenient to have.
            # On first query expensive bookkeeping is done (creation of r-tree),
            # and is cached for subsequent queries
            self.ray = ray_triangle.RayMeshIntersector(self)

        # a quick way to get permuted versions of the current mesh
        self.permutate = permutate.Permutator(self)

        # convience class for nearest point queries
        self.nearest = proximity.ProximityQuery(self)

        # store metadata about the mesh in a dictionary
        self.metadata = dict()
        # update the mesh metadata with passed metadata
        if isinstance(metadata, dict):
            self.metadata.update(metadata)

        # Set the default center of mass and density
        self._density = 1.0
        self._center_mass = None

        # process will remove NaN and Inf values and merge vertices
        # if validate, will remove degenerate and duplicate faces
        if process or validate:
            self.process()

        # store all passed kwargs for debugging purposes
        self._kwargs = kwargs

    def process(self):
        """
        Do the bare minimum processing to make a mesh useful.

        Does this by:
            1) removing NaN and Inf values

            2) merging duplicate vertices

        If self._validate:
            3) Remove triangles which have one edge of their rectangular 2D
               oriented bounding box shorter than tol.merge

            4) remove duplicated triangles

        Returns
        ------------
        self: Trimesh object
        """
        # if there are no vertices or faces exit early
        if self.is_empty:
            return self

        # avoid clearing the cache during operations
        with self._cache:
            self.remove_infinite_values()
            self.merge_vertices()
            # if we're cleaning remove duplicate
            # and degenerate faces
            if self._validate:
                self.remove_duplicate_faces()
                self.remove_degenerate_faces()
        # since none of our process operations moved vertices or faces,
        # we can keep face and vertex normals in the cache without recomputing
        # if faces or vertices have been removed, normals are validated before
        # being returned so there is no danger of inconsistent dimensions
        self._cache.clear(exclude=['face_normals',
                                   'vertex_normals'])
        self.metadata['processed'] = True
        return self

    def md5(self):
        """
        An MD5 of the core geometry information for the mesh,
        faces and vertices.

        Generated from TrackedArray, which subclasses np.ndarray to monitor for
        changes and returns a correct, but lazily evaluated md5 so it only has to
        recalculate the hash occasionally, rather than on every call.

        Returns
        ----------
        md5: string, md5 of md5 hashes of everything in the DataStore
        """
        md5 = self._data.md5()
        return md5

    def crc(self):
        """
        A zlib.adler32 checksum for the current mesh data.

        This is about 5x faster than an MD5, and the checksum is
        checked every time something is requested from the cache so
        it gets called a lot.

        Returns
        ----------
        crc: int, checksum of current mesh data
        """
        return self._data.crc()

    @property
    def faces(self):
        """
        The faces of the mesh.

        This is regarded as core information which cannot be regenerated from
        cache, and as such is stored in self._data which tracks the array for
        changes and clears cached values of the mesh if this is altered.

        Returns
        ----------
        faces: (n,3) int, representing triangles which reference self.vertices
        """
        return self._data['faces']

    @faces.setter
    def faces(self, values):
        """
        Set the vertex indexes that make up triangular faces.

        Parameters
        --------------
        values: (n, 3) int, indexes of self.vertices
        """
        if values is None:
            values = []
        values = np.asanyarray(values, dtype=np.int64)
        # automatically triangulate quad faces
        if util.is_shape(values, (-1, 4)):
            log.info('Triangulating quad faces')
            values = geometry.triangulate_quads(values)
        self._data['faces'] = values

    @util.cache_decorator
    def faces_sparse(self):
        """
        A sparse matrix representation of the faces.

        Returns
        ----------
        sparse: scipy.sparse.coo_matrix with:
                dtype: bool
                shape: (len(self.vertices), len(self.faces))
        """
        sparse = geometry.index_sparse(column_count=len(self.vertices),
                                       indices=self.faces)
        return sparse

    @util.cache_decorator
    def face_normals(self):
        """
        Return the unit normal vector for each face.

        If a face is degenerate and a normal can't be generated
        a zero magnitude unit vector will be returned for that face.

        Returns
        -----------
        normals: (len(self.faces), 3) float64, normal vectors
        """
        # if the shape of the cached normals is incorrect, generate normals
        if (np.shape(self._cache['face_normals']) !=
                np.shape(self._data['faces'])):
            log.debug('generating face normals as shape was incorrect')
            # use cached triangle cross products to generate normals
            # this will always return the correct shape but some values
            # will be zero or an arbitrary vector if the inputs had a cross
            # produce below machine epsilon
            normals, valid = triangles.normals(triangles=self.triangles,
                                               crosses=self.triangles_cross)
            if valid.all():
                return normals
            # make a padded list of normals to make sure shape is correct
            padded = np.zeros((len(self.triangles), 3), dtype=np.float64)
            padded[valid] = normals
            return padded

    @face_normals.setter
    def face_normals(self, values):
        """
        Assign values to face normals

        Parameters
        -------------
        values: (len(self.faces), 3) float, unit face normals
        """
        if values is not None:
            # make sure face normals are C- contiguous float
            self._cache['face_normals'] = np.asanyarray(values,
                                                        order='C',
                                                        dtype=np.float64)

    @property
    def vertices(self):
        """
        The vertices of the mesh.

        This is regarded as core information which cannot be regenerated
        from cache and as such is stored in self._data which tracks the array
        for changes and clears cached values of the mesh if this is altered.

        Returns
        ----------
        vertices: (n, 3) float representing points in cartesian space
        """
        return self._data['vertices']

    @vertices.setter
    def vertices(self, values):
        """
        Assign vertex values to the mesh.

        Parameters
        --------------
        values: (n, 3) float, points in space
        """
        self._data['vertices'] = np.asanyarray(values,
                                               order='C',
                                               dtype=np.float64)

    @util.cache_decorator
    def vertex_normals(self):
        """
        The vertex normals of the mesh. If the normals were loaded, we check to
        make sure we have the same number of vertex normals and vertices before
        returning them. If there are no vertex normals defined, or a shape mismatch
        we calculate the vertex normals from the mean normals of the faces the
        vertex is used in.

        Returns
        ----------
        vertex_normals: (n,3) float, where n == len(self.vertices)
                         Represents the surface normal at each vertex.
        """
        # make sure we have faces_sparse
        assert hasattr(self.faces_sparse, 'dot')
        vertex_normals = geometry.mean_vertex_normals(len(self.vertices),
                                                      self.faces,
                                                      self.face_normals,
                                                      sparse=self.faces_sparse)
        return vertex_normals

    @vertex_normals.setter
    def vertex_normals(self, values):
        """
        Assign values to vertex normals

        Parameters
        -------------
        values: (len(self.vertices), 3) float, unit normal vectors
        """
        if values is not None:
            values = np.asanyarray(values,
                                   order='C',
                                   dtype=np.float64)
            if values.shape == self.vertices.shape:
                self._cache['vertex_normals'] = values

    @util.cache_decorator
    def bounding_box(self):
        """
        An axis aligned bounding box for the current mesh.

        Returns
        ----------
        aabb: trimesh.primitives.Box object with transform and extents defined
              to represent the axis aligned bounding box of the mesh
        """
        from . import primitives
        center = self.bounds.mean(axis=0)
        transform = transformations.translation_matrix(center)
        aabb = primitives.Box(transform=transform,
                              extents=self.extents,
                              mutable=False)
        return aabb

    @util.cache_decorator
    def bounding_box_oriented(self):
        """
        An oriented bounding box for the current mesh.

        Returns
        ---------
        obb: trimesh.primitives.Box object with transform and extents defined
             to represent the minimum volume oriented bounding box of the mesh
        """
        from . import primitives
        to_origin, extents = bounds.oriented_bounds(self)
        obb = primitives.Box(transform=np.linalg.inv(to_origin),
                             extents=extents,
                             mutable=False)
        return obb

    @util.cache_decorator
    def bounding_sphere(self):
        """
        A minimum volume bounding sphere for the current mesh.

        Note that the Sphere primitive returned has an unpadded, exact
        sphere_radius so while the distance of every vertex of the current
        mesh from sphere_center will be less than sphere_radius, the faceted
        sphere primitive may not contain every vertex

        Returns
        --------
        minball: trimesh.primitives.Sphere object
        """
        from . import primitives
        center, radius = nsphere.minimum_nsphere(self)
        minball = primitives.Sphere(center=center,
                                    radius=radius,
                                    mutable=False)
        return minball

    @util.cache_decorator
    def bounding_cylinder(self):
        """
        A minimum volume bounding cylinder for the current mesh.

        Returns
        --------
        mincyl: trimesh.primitives.Cylinder object
        """
        from . import primitives
        kwargs = bounds.minimum_cylinder(self)
        mincyl = primitives.Cylinder(mutable=False, **kwargs)
        return mincyl

    @util.cache_decorator
    def bounding_primitive(self):
        """
        The minimum volume primitive (box, sphere, or cylinder) that
        bounds the mesh.

        Returns
        ---------
        bounding_primitive: trimesh.primitives.Sphere
                            trimesh.primitives.Box
                            trimesh.primitives.Cylinder
        """
        options = [self.bounding_box_oriented,
                   self.bounding_sphere,
                   self.bounding_cylinder]
        volume_min = np.argmin([i.volume for i in options])
        bounding_primitive = options[volume_min]
        return bounding_primitive

    @util.cache_decorator
    def bounds(self):
        """
        The axis aligned bounds of the mesh.

        Returns
        -----------
        bounds: (2,3) float, bounding box with [min, max] coordinates
        """
        # we use triangles instead of faces because
        # if there is an unused vertex it will screw up the bounds
        in_mesh = self.triangles.reshape((-1, 3))
        bounds = np.vstack((in_mesh.min(axis=0),
                            in_mesh.max(axis=0)))
        return bounds

    @util.cache_decorator
    def extents(self):
        """
        The length, width, and height of the bounding box of the mesh.

        Returns
        -----------
        extents: (3,) float array containing axis aligned [l,w,h]
        """
        extents = self.bounds.ptp(axis=0)
        return extents

    @util.cache_decorator
    def scale(self):
        """
        A metric for the overall scale of the mesh, the length of the
        diagonal of the axis aligned bounding box of the mesh.

        Returns
        ----------
        scale: float, the diagonal of the meshes AABB
        """
        scale = (self.extents ** 2).sum() ** .5
        return scale

    @util.cache_decorator
    def centroid(self):
        """
        The point in space which is the average of the triangle centroids
        weighted by the area of each triangle.

        This will be valid even for non- watertight meshes,
        unlike self.center_mass

        Returns
        ----------
        centroid: (3,) float, the average vertex
        """

        # use the centroid of each triangle weighted by
        # the area of the triangle to find the overall centroid
        centroid = np.average(self.triangles_center,
                              axis=0,
                              weights=self.area_faces)
        return centroid

    @property
    def center_mass(self):
        """
        The point in space which is the center of mass/volume.

        If the current mesh is not watertight, this is meaningless garbage
        unless it was explicitly set.

        Returns
        -----------
        center_mass: (3,) float array, volumetric center of mass of the mesh
        """
        center_mass = self.mass_properties['center_mass']
        return center_mass

    @center_mass.setter
    def center_mass(self, cm):
        self._center_mass = cm
        self._cache.delete('mass_properties')

    @property
    def density(self):
        """
        The density of the mesh.

        Returns
        -----------
        density: float, the density of the mesh.
        """
        density = self.mass_properties['density']
        return density

    @density.setter
    def density(self, value):
        self._density = float(value)
        self._cache.delete('mass_properties')

    @property
    def volume(self):
        """
        Volume of the current mesh.
        If the current mesh isn't watertight this is garbage.

        Returns
        ---------
        volume: float, volume of the current mesh
        """
        volume = self.mass_properties['volume']
        return volume

    @property
    def mass(self):
        """
        Mass of the current mesh.
        If the current mesh isn't watertight this is garbage.

        Returns
        ---------
        mass: float, mass of the current mesh
        """
        mass = self.mass_properties['mass']
        return mass

    @property
    def moment_inertia(self):
        """
        Return the moment of inertia matrix of the current mesh.
        If mesh isn't watertight this is garbage.

        Returns
        ---------
        inertia: (3,3) float, moment of inertia of the current mesh.
        """
        inertia = self.mass_properties['inertia']
        return inertia

    @util.cache_decorator
    def principal_inertia_components(self):
        """
        Return the principal components of inertia

        Ordering corresponds to mesh.principal_inertia_vectors

        Returns
        ----------
        components: (3,) float, principal components of inertia
        """
        components, vectors = inertia.principal_axis(self.moment_inertia)
        self._cache['principal_inertia_vectors'] = vectors

        return components

    @property
    def principal_inertia_vectors(self):
        """
        Return the principal axis of inertia.

        Ordering corresponds to mesh.principal_inertia_components

        Returns
        ----------
        vectors:    (3,3) float, 3 vectors pointing along the
                                 principal axis of inertia
        """
        populate = self.principal_inertia_components
        return self._cache['principal_inertia_vectors']

    @util.cache_decorator
    def principal_inertia_transform(self):
        """
        A transform which moves the current mesh so the principal
        inertia vectors are on the X,Y, and Z axis, and the centroid is
        at the origin.

        Returns
        ----------
        tranform: (4,4) float, homogenous transformation matrix
        """
        order = np.argsort(self.principal_inertia_components)[1:][::-1]
        vectors = self.principal_inertia_vectors[order]
        vectors = np.vstack((vectors, np.cross(*vectors)))

        transform = np.eye(4)
        transform[:3, :3] = vectors
        transform = geometry.transform_around(transform, self.centroid)
        transform[:3, 3] -= self.centroid

        return transform

    @util.cache_decorator
    def symmetry(self):
        """
        Check whether a mesh has rotational symmetry.

        Returns
        -----------
        symmetry: None         No rotational symmetry
                  'radial'     Symmetric around an axis
                  'spherical'  Symmetric around a point
        """
        symmetry, axis, section = inertia.radial_symmetry(self)
        self._cache['symmetry_axis'] = axis
        self._cache['symmetry_section'] = section
        return symmetry

    @property
    def symmetry_axis(self):
        """
        If a mesh has rotational symmetry, return the axis.

        Returns
        ------------
        axis: (3,) float, axis around which a 2D profile
                          was revolved to generate this mesh
        """
        if self.symmetry is not None:
            return self._cache['symmetry_axis']

    @property
    def symmetry_section(self):
        """
        If a mesh has rotational symmetry, return the two
        vectors which make up a section coordinate frame.

        Returns
        ----------
        section: (2, 3) float, vectors to take a section along
        """
        if self.symmetry is not None:
            return self._cache['symmetry_section']

    @util.cache_decorator
    def triangles(self):
        """
        Actual triangles of the mesh (points, not indexes)

        Returns
        ---------
        triangles: (n,3,3) float points of vertices grouped into triangles
        """
        # use of advanced indexing on our tracked arrays will
        # trigger a change flag which means the MD5 will have to be
        # recomputed. We can escape this check by viewing the array.
        triangles = self.vertices.view(np.ndarray)[self.faces]
        # make triangles (which are derived from faces/vertices) not writeable
        triangles.flags.writeable = False
        return triangles

    @util.cache_decorator
    def triangles_tree(self):
        """
        An R-tree containing each face of the mesh.

        Returns
        ----------
        tree: rtree.index where each triangle in self.faces has a rectangular cell
        """
        tree = triangles.bounds_tree(self.triangles)
        return tree

    @util.cache_decorator
    def triangles_center(self):
        """
        The center of each triangle (barycentric [1/3, 1/3, 1/3])

        Returns
        ---------
        triangles_center: (len(self.faces), 3) float, center of each triangular face
        """
        triangles_center = self.triangles.mean(axis=1)
        return triangles_center

    @util.cache_decorator
    def triangles_cross(self):
        """
        The cross product of two edges of each triangle.

        Returns
        ---------
        crosses: (n,3) float, cross product of each triangle
        """
        crosses = triangles.cross(self.triangles)
        return crosses

    @util.cache_decorator
    def edges(self):
        """
        Edges of the mesh (derived from faces).

        Returns
        ---------
        edges: (n,2) int, set of vertex indices
        """
        edges, index = geometry.faces_to_edges(self.faces.view(np.ndarray),
                                               return_index=True)
        self._cache['edges_face'] = index
        return edges

    @util.cache_decorator
    def edges_face(self):
        """
        Which face does each edge belong to.

        Returns
        ---------
        edges_face: (n,) int, index of self.faces
        """
        populate = self.edges
        return self._cache['edges_face']

    @util.cache_decorator
    def edges_unique(self):
        """
        The unique edges of the mesh.

        Returns
        ----------
        edges_unique: (n,2) int, set of vertex indices for unique edges
        """
        unique, inverse = grouping.unique_rows(self.edges_sorted)
        edges_unique = self.edges_sorted[unique]
        # edges_unique will be added automatically by the decorator
        # additional terms generated need to be added to the cache manually
        self._cache['edges_unique_idx'] = unique
        self._cache['edges_unique_inv'] = inverse
        return edges_unique

    @util.cache_decorator
    def edges_sorted(self):
        """
        Returns
        ----------
        self.edges, but sorted along axis 1
        """
        edges_sorted = np.sort(self.edges, axis=1)
        return edges_sorted

    @util.cache_decorator
    def faces_unique_edges(self):
        """
        For each face return which indexes in mesh.unique_edges constructs that face.

        Returns
        ---------
        faces_unique_edges: self.faces.shape int, which indexes of self.edges_unique
                            construct self.faces

        Examples
        ---------
        In [0]: mesh.faces[0:2]
        Out[0]:
        TrackedArray([[    1,  6946, 24224],
                      [ 6946,  1727, 24225]])

        In [1]: mesh.edges_unique[mesh.faces_unique_edges[0:2]]
        Out[1]:
        array([[[    1,  6946],
                [ 6946, 24224],
                [    1, 24224]],
               [[ 1727,  6946],
                [ 1727, 24225],
                [ 6946, 24225]]])
        """
        # make sure we have populated unique edges
        populate = self.edges_unique
        # we are relying on the fact that edges are stacked in triplets
        result = self._cache['edges_unique_inv'].reshape((-1, 3))
        return result

    @util.cache_decorator
    def euler_number(self):
        """
        Return the Euler characteristic (a topological invariant) for the mesh
        In order to guarantee correctness, this should be called after
        remove_unreferenced_vertices

        Returns
        ----------
        euler_number: int, topological invarient
        """
        euler = len(self.vertices) - len(self.edges_unique) + len(self.faces)
        return euler

    @property
    def units(self):
        """
        Definition of units for the mesh.

        Returns
        ----------
        units: str, unit system mesh is in, or None if not defined
        """
        if 'units' in self.metadata:
            return self.metadata['units']
        else:
            return None

    @units.setter
    def units(self, value):
        value = str(value).lower()
        self.metadata['units'] = value

    def convert_units(self, desired, guess=False):
        """
        Convert the units of the mesh into a specified unit.

        Parameters
        ----------
        desired: string, units to convert to (eg 'inches')
        guess:   boolean, if self.units are not defined should we
                 guess the current units of the document and then convert?
        """
        units._set_units(self, desired, guess)

    def merge_vertices(self):
        """
        If a mesh has vertices that are closer than trimesh.constants.tol.merge
        redefine them to be the same vertex and replace face references
        """
        grouping.merge_vertices_hash(self)

    def update_vertices(self, mask, inverse=None):
        """
        Update vertices with a mask.

        Parameters
        ----------
        vertex_mask: (len(self.vertices)) boolean array of which
                     vertices to keep
        inverse:     (len(self.vertices)) int array to reconstruct
                     vertex references (such as output by np.unique)
        """
        # if the mesh is already empty we can't remove anything
        if self.is_empty:
            return

        # make sure mask is a numpy array
        mask = np.asanyarray(mask)

        if ((mask.dtype.name == 'bool' and mask.all()) or
                len(mask) == 0 or self.is_empty):
            # mask doesn't remove any vertices so exit early
            return

        # re- index faces from inverse
        if inverse is not None and util.is_shape(self.faces, (-1, 3)):
            self.faces = inverse[self.faces.reshape(-1)].reshape((-1, 3))

        # update the visual object with our mask
        self.visual.update_vertices(mask)
        # get the normals from cache before dumping
        cached_normals = self._cache.get('vertex_normals')

        # actually apply the mask
        self.vertices = self.vertices[mask]

        # if we had passed vertex normals try to save them
        if util.is_shape(cached_normals, (-1, 3)):
            try:
                self.vertex_normals = cached_normals[mask]
            except BaseException:
                pass

    def update_faces(self, mask):
        """
        In many cases, we will want to remove specific faces.
        However, there is additional bookkeeping to do this cleanly.
        This function updates the set of faces with a validity mask,
        as well as keeping track of normals and colors.

        Parameters
        ---------
        valid: either (m) int, or (len(self.faces)) bool.
        """
        # if the mesh is already empty we can't remove anything
        if self.is_empty:
            return

        mask = np.asanyarray(mask)
        if mask.dtype.name == 'bool' and mask.all():
            # mask removes no faces so exit early
            return

        # try to save face normals before dumping cache
        cached_normals = self._cache.get('face_normals')

        faces = self._data['faces']
        # if Trimesh has been subclassed and faces have been moved from data
        # to cache, get faces from cache.
        if not util.is_shape(faces, (-1, 3)):
            faces = self._cache['faces']

        # actually apply the mask
        self.faces = faces[mask]
        # apply the mask to the visual object
        self.visual.update_faces(mask)

        # if our normals were the correct shape apply them
        if util.is_shape(cached_normals, (-1, 3)):
            self.face_normals = cached_normals[mask]

    def remove_infinite_values(self):
        """
        Ensure that every vertex and face consists of finite numbers.

        This will remove vertices or faces containing np.nan and np.inf

        Alters
        ----------
        self.faces:    masked to remove np.inf/np.nan
        self.vertices: masked to remove np.inf/np.nan
        """
        if util.is_shape(self.faces, (-1, 3)):
            # (len(self.faces),) bool, mask for faces
            face_mask = np.isfinite(self.faces).all(axis=1)
            self.update_faces(face_mask)

        if util.is_shape(self.vertices, (-1, 3)):
            # (len(self.vertices),) bool, mask for vertices
            vertex_mask = np.isfinite(self.vertices).all(axis=1)
            self.update_vertices(vertex_mask)

    def remove_duplicate_faces(self):
        """
        On the current mesh remove any faces which are duplicates.

        Alters
        ----------
        self.faces: removes duplicates
        """
        unique, inverse = grouping.unique_rows(np.sort(self.faces, axis=1))
        self.update_faces(unique)

    def rezero(self):
        """
        Translate the mesh so that all vertex vertices are positive.

        Alters
        ----------
        self.vertices: Translated to first octant (all values > 0)
        """
        self.apply_translation(self.bounds[0] * -1.0)

    @_log_time
    def split(self, only_watertight=True, adjacency=None, **kwargs):
        """
        Returns a list of Trimesh objects, based on face connectivity.
        Splits into individual components, sometimes referred to as 'bodies'

        Parameters
        ---------
        only_watertight: only meshes which are watertight are returned
        adjacency: if not None, override face adjacency with custom values (n,2)

        Returns
        ---------
        meshes: (n) list of Trimesh objects
        """
        meshes = graph.split(self,
                             only_watertight=only_watertight,
                             adjacency=adjacency,
                             **kwargs)
        return meshes

    @util.cache_decorator
    def face_adjacency(self):
        """
        Find faces that share an edge, which we call here 'adjacent'.

        Returns
        ----------
        adjacency: (n,2) int, pairs of faces which share an edge

        Examples
        ---------

        In [1]: mesh = trimesh.load('models/featuretype.STL')

        In [2]: mesh.face_adjacency
        Out[2]:
        array([[   0,    1],
               [   2,    3],
               [   0,    3],
               ...,
               [1112,  949],
               [3467, 3475],
               [1113, 3475]])

        In [3]: mesh.faces[mesh.face_adjacency[0]]
        Out[3]:
        TrackedArray([[   1,    0,  408],
                      [1239,    0,    1]], dtype=int64)

        In [4]: import networkx as nx

        In [5]: graph = nx.from_edgelist(mesh.face_adjacency)

        In [6]: groups = nx.connected_components(graph)
        """
        adjacency, edges = graph.face_adjacency(mesh=self,
                                                return_edges=True)
        self._cache['face_adjacency_edges'] = edges
        return adjacency

    @util.cache_decorator
    def face_adjacency_edges(self):
        """
        Returns the edges that are shared by the adjacent faces.

        Returns
        --------
        edges: (n, 2) list of vertex indices which correspond to face_adjacency
        """
        # this value is calculated as a byproduct of the face adjacency
        populate = self.face_adjacency
        return self._cache['face_adjacency_edges']

    @util.cache_decorator
    def face_adjacency_angles(self):
        """
        Return the angle between adjacent faces

        Returns
        --------
        adjacency_angle: (n,) float angle between adjacent faces.
                         Each value corresponds with self.face_adjacency
        """
        pairs = self.face_normals[self.face_adjacency]
        angles = geometry.vector_angle(pairs)
        return angles

    @util.cache_decorator
    def face_adjacency_projections(self):
        """
        The projection of the non- shared vertex of a triangle onto
        its adjacent face

        Returns
        ----------
        projections: (len(self.face_adjacency),) float, dot product of vertex
                     onto plane of adjacent triangle.
        """
        projections = convex.adjacency_projections(self)
        return projections

    @util.cache_decorator
    def face_adjacency_convex(self):
        """
        Return faces which are adjacent and locally convex.

        What this means is that given faces A and B, the one vertex
        in B that is not shared with A, projected onto the plane of A
        has a projection that is zero or negative.

        Returns
        ----------
        are_convex: (len(self.face_adjacency),) bool, face pairs that are
                    locally convex.
        """
        are_convex = self.face_adjacency_projections < tol.merge
        return are_convex

    @util.cache_decorator
    def face_adjacency_unshared(self):
        """
        Return the vertex index of the two vertices not in the shared
        edge between two adjacent faces

        Parameters
        ----------
        mesh: Trimesh object

        Returns
        -----------
        vid_unshared: (len(mesh.face_adjacency), 2) int, indexes of mesh.vertices
        """
        vid_unshared = graph.face_adjacency_unshared(self)
        return vid_unshared

    @util.cache_decorator
    def face_adjacency_radius(self):
        """
        The approximate radius of a cylinder that fits inside adjacent faces.

        Returns
        ------------
        radii: (len(self.face_adjacency),) float, approximate radius formed
        """
        radii, span = graph.face_adjacency_radius(mesh=self)
        self._cache['face_adjacency_span'] = span
        return radii

    @util.cache_decorator
    def face_adjacency_span(self):
        """
        The approximate perpendicular projection of the non- shared
        vertices in a pair of adjacent faces onto the shared edge of
        the two faces.

        Returns
        ------------
        radii: (len(self.face_adjacency),) float, approximate radius formed
        """
        populate = self.face_adjacency_radius
        return self._cache['face_adjacency_span']

    @util.cache_decorator
    def vertex_adjacency_graph(self):
        """
        Returns a networkx graph representing the vertices and their connections
        in the mesh.

        Parameters
        ----------
        mesh:         Trimesh object

        Returns
        ---------
        graph: networkx.Graph(), graph representing vertices and edges between
                                 them,where vertices are networkx Nodes and edges
                                 are Edges.

        Examples
        ----------
        This is useful for getting nearby vertices for a given vertex,
        potentially for some simple smoothing techniques.

        mesh = trimesh.primitives.Box()
        graph = mesh.vertex_adjacency_graph
        graph.neighbors(0)
        > [1,2,3,4]
        """

        adjacency_g = graph.vertex_adjacency_graph(mesh=self)
        return adjacency_g

    @util.cache_decorator
    def vertex_neighbors(self):
        """
        The vertex neighbors of each vertex of the mesh, determined from
        the cached vertex_adjacency_graph, if already existant.

        Returns
        ----------
        vertex_neighbors: (n,) int, where n == len(self.vertices)
                         Represents each vertex's immediate neighbors along
                         the edge of a triangle.

        Examples
        ----------
        This is useful for getting nearby vertices for a given vertex,
        potentially for some simple smoothing techniques.

        >>> mesh = trimesh.primitives.Box()
        >>> mesh.vertex_neighbors[0]
        [1,2,3,4]
        """
        g = self.vertex_adjacency_graph
        l = [g.neighbors(v_i) for v_i, _ in enumerate(self.vertices)]
        return np.array(l)

    @util.cache_decorator
    def is_winding_consistent(self):
        """
        Does the mesh have consistent winding or not.
        A mesh with consistent winding has each shared edge
        going in an opposite direction from the other in the pair.

        Returns
        --------
        consistent: bool, if winding is consistent or not
        """
        if self.is_empty:
            return False
        # consistent winding check is populated into the cache by is_watertight
        populate = self.is_watertight
        return bool(self._cache['is_winding_consistent'])

    @util.cache_decorator
    def is_watertight(self):
        """
        Check if a mesh is watertight by making sure every edge is included in
        two faces.

        Returns
        ----------
        is_watertight: bool, is mesh watertight or not
        """
        if self.is_empty:
            return False
        watertight, is_reversed = graph.is_watertight(
            edges=self.edges, edges_sorted=self.edges_sorted)
        self._cache['is_winding_consistent'] = is_reversed
        return bool(watertight)

    @util.cache_decorator
    def is_volume(self):
        """
        Check if a mesh has all the properties required to represent
        a valid volume, rather than just a surface.

        These properties include being watertight, having consistent winding,
        and outward facing normals.

        Returns
        ---------
        valid: bool, does the mesh represent a volume
        """
        valid = bool(self.is_watertight and
                     self.is_winding_consistent and
                     self.volume > 0)
        return valid

    @util.cache_decorator
    def is_empty(self):
        """
        Does the current mesh have data defined.

        Returns
        --------
        empty: if True, no data exists in the mesh.
        """
        return bool(self._data.is_empty())

    @util.cache_decorator
    def is_convex(self):
        """
        Check if a mesh is convex or not.

        Returns
        ----------
        is_convex: bool, is mesh convex or not
        """
        if self.is_empty:
            return False

        is_convex = bool(convex.is_convex(self))
        return is_convex

    @util.cache_decorator
    def kdtree(self):
        """
        Return a scipy.spatial.cKDTree of the vertices of the mesh.
        Not cached as this lead to observed memory issues and segfaults.

        Returns
        ---------
        tree: scipy.spatial.cKDTree containing mesh vertices
        """

        from scipy.spatial import cKDTree as KDTree
        tree = KDTree(self.vertices.view(np.ndarray))
        return tree

    def remove_degenerate_faces(self, height=tol.merge):
        """
        Remove degenerate faces (faces without 3 unique vertex indices)
        from the current mesh.

        If a height is specified, it will remove any face with a 2D oriented
        bounding box with one edge shorter than that height.

        If not specified, it will remove any face with a zero normal.

        Parameters
        ------------
        height: float, if specified removes faces with an oriented bounding
                box shorter than this on one side.

        Returns
        -------------
        nondegenerate: (len(self.faces),) bool, mask used to remove faces
        """
        nondegenerate = triangles.nondegenerate(self.triangles,
                                                areas=self.area_faces,
                                                height=height)
        self.update_faces(nondegenerate)

        return nondegenerate

    @util.cache_decorator
    def facets(self):
        """
        Return a list of face indices for coplanar adjacent faces.

        Returns
        ---------
        facets: (n) sequence int, groups of indexes for self.faces
        """
        facets = graph.facets(self)
        return facets

    @util.cache_decorator
    def facets_area(self):
        """
        Return an array containing the area of each facet.

        Returns
        ---------
        area:   (len(self.facets),) float, list of face group area
        """
        # avoid thrashing the cache inside a loop
        area_faces = self.area_faces
        # sum the area of each group of faces represented by facets
        areas = np.array([area_faces[i].sum() for i in self.facets])
        return areas

    @util.cache_decorator
    def facets_normal(self):
        """
        Return the normal of each facet

        Returns
        ---------
        normals: (n,3) float, normal vector of each facet
        """
        if len(self.facets) == 0:
            return np.array([])
        index_first = np.array([i[0] for i in self.facets])
        normals = self.face_normals[index_first]
        return normals

    @util.cache_decorator
    def facets_boundary(self):
        """
        Return the edges which represent the boundary of each facet

        Returns
        ---------
        edges_boundary: sequence of (n,2) int, indices of self.vertices
        """
        # make each row correspond to a single face
        edges = self.edges_sorted.reshape((-1, 6))
        # get the edges for each facet
        edges_facet = [edges[i].reshape((-1, 2)) for i in self.facets]
        edges_boundary = np.array([i[grouping.group_rows(i, require_count=1)]
                                   for i in edges_facet])
        return edges_boundary

    @util.cache_decorator
    def facets_on_hull(self):
        """
        Find which facets of the mesh are on the convex hull.

        Returns
        ---------
        on_hull: (len(mesh.facets),) bool, is facet on convex hull
        """
        # the index of the largest face in each facet to test
        face_id = [f[self.area_faces[f].argmax()] for f in self.facets]

        # test the triangle center and 3 vertices
        # if all 4 coplanar points are on the convex hull
        # it is a very strong indication that the whole planar
        # facet region is on the convex hull
        test = np.zeros((len(face_id), 4, 3), dtype=np.float64)
        test[:, :3, :] = self.triangles[face_id]
        test[:, 3, :] = self.triangles_center[face_id]

        # distance between the hull surface and our four test points
        distance = self.convex_hull.nearest.on_surface(
            test.reshape((-1, 3)))[1]

        # threshold the distance and check all points
        ok = (distance < tol.merge).reshape((-1, 4)).all(axis=1)

        return ok

    @_log_time
    def fix_normals(self):
        """
        Find and fix problems with self.face_normals and self.faces winding direction.

        For face normals ensure that vectors are consistently pointed outwards,
        and that self.faces is wound in the correct direction for all
        connected components.
        """
        repair.fix_normals(self)

    def fill_holes(self):
        """
        Fill single triangle and single quad holes in the current mesh.

        Returns
        ----------
        watertight: bool, is the mesh watertight after the function completes
        """
        return repair.fill_holes(self)

    def compute_stable_poses(self,
                             center_mass=None,
                             sigma=0.0,
                             n_samples=1,
                             threshold=0.0):
        """
        Computes stable orientations of a mesh and their quasi-static probabilites.

        This method samples the location of the center of mass from a multivariate
        gaussian (mean at com, cov equal to identity times sigma) over n_samples.
        For each sample, it computes the stable resting poses of the mesh on a
        a planar workspace and evaulates the probabilities of landing in
        each pose if the object is dropped onto the table randomly.

        This method returns the 4x4 homogenous transform matrices that place
        the shape against the planar surface with the z-axis pointing upwards
        and a list of the probabilities for each pose.
        The transforms and probabilties that are returned are sorted, with the
        most probable pose first.

        Parameters
        ----------
        mesh:        Trimesh object, the target mesh
        center_mass: (3,) float,     the object center of mass (if None, this method
                                     assumes uniform density and watertightness and
                                     computes a center of mass explicitly)
        sigma:     float,            the covariance for the multivariate gaussian used
                                     to sample center of mass locations
        n_samples: int,             the number of samples of the center of mass loc
        threshold: float,           the probability value at which to threshold
                                      returned stable poses

        Returns
        -------
        transforms: list of (4,4) floats, the homogenous matrices that transform the
                                        object to rest in a stable pose, with the
                                        new z-axis pointing upwards from the table
                                        and the object just touching the table.

        probs:      list of floats,       a probability in (0, 1) for each pose
        """
        return poses.compute_stable_poses(mesh=self,
                                          center_mass=center_mass,
                                          sigma=sigma,
                                          n_samples=n_samples,
                                          threshold=threshold)

    def subdivide(self, face_index=None):
        """
        Subdivide a mesh, with each subdivided face replaced with four
        smaller faces.

        Parameters
        ----------
        mesh: Trimesh object
        face_index: faces to subdivide.
                    if None: all faces of mesh will be subdivided
                    if (n,) int array of indices: only specified faces will be
                       subdivided. Note that in this case the mesh will generally
                       no longer be manifold, as the additional vertex on the midpoint
                       will not be used by the adjacent faces to the faces specified,
                       and an additional postprocessing step will be required to
                       make resulting mesh watertight
        """
        vertices, faces = remesh.subdivide(vertices=self.vertices,
                                           faces=self.faces,
                                           face_index=face_index)
        return Trimesh(vertices=vertices, faces=faces)

    @_log_time
    def smoothed(self, angle=.4):
        """
        Return a version of the current mesh which will render nicely.
        Does not change current mesh in any way.

        Returns
        ---------
        smoothed: Trimesh object, non watertight version of current mesh
                  which will render nicely with smooth shading.
        """

        # smooth should be recomputed if visuals change, so we
        # store it in the visuals cache rather than the main mesh cache
        cached = self.visual._cache.get('smoothed')
        if cached is not None:
            return cached
        return self.visual._cache.set(key='smoothed',
                                      value=graph.smoothed(self, angle))

    def section(self,
                plane_normal,
                plane_origin):
        """
        Returns a cross section of the current mesh and plane defined by
        origin and normal.

        Parameters
        ---------
        plane_normal: (3) vector for plane normal
        plane_origin: (3) vector for plane origin

        Returns
        ---------
        intersections: Path3D of intersections
        """

        from .io.load import load_path
        lines, face_index = intersections.mesh_plane(mesh=self,
                                                     plane_normal=plane_normal,
                                                     plane_origin=plane_origin,
                                                     return_faces=True)
        if len(lines) == 0:
            return None
        path = load_path(lines)
        path.metadata['face_index'] = face_index

        return path

    @util.cache_decorator
    def convex_hull(self):
        """
        Get a new Trimesh object representing the convex hull of the
        current mesh.

        Returns
        --------
        convex: Trimesh object of convex hull of current mesh
        """
        hull = convex.convex_hull(self)
        return hull

    def sample(self, count, return_index=False):
        """
        Return random samples distributed normally across the
        surface of the mesh

        Parameters
        ---------
        count: int, number of points to sample
        return_index: bool, if True will also return face index

        Returns
        ---------
        samples:    (count, 3) float, points on surface of mesh
        face_index: (count,) int, index of self.faces
        """
        samples, index = sample.sample_surface(self, count)
        if return_index:
            return samples, index
        return samples

    def remove_unreferenced_vertices(self):
        """
        Remove all vertices in the current mesh which are not referenced
        by a face.
        """
        unique, inverse = np.unique(self.faces.reshape(-1),
                                    return_inverse=True)
        self.faces = inverse.reshape((-1, 3))
        self.vertices = self.vertices[unique]

    def unmerge_vertices(self):
        """
        Removes all face references so that every face contains three
        unique vertex indices and no faces are adjacent.
        """
        vertices = self.vertices[self.faces].reshape((-1, 3))
        faces = np.arange(len(vertices),
                          dtype=np.int64).reshape((-1, 3))

        self.faces = faces
        self.vertices = vertices
        self._cache.clear(exclude=['face_normals'])

    def apply_translation(self, translation):
        """
        Translate the current mesh.

        Parameters
        ----------
        translation: (3,) float, translation in XYZ
        """
        translation = np.asanyarray(translation, dtype=np.float64)
        if translation.shape != (3,):
            raise ValueError('Translation must be (3,)!')

        matrix = np.eye(4)
        matrix[:3, 3] = translation
        self.apply_transform(matrix)

    def apply_scale(self, scaling):
        """
        Scale the mesh equally on all axis.

        Parameters
        ----------
        scaling: float, scale factor
        """
        scaling = float(scaling)
        if not np.isfinite(scaling):
            raise ValueError('Scaling factor must be finite number!')

        matrix = np.eye(4)
        matrix[:3, :3] *= scaling
        # apply_transform will work nicely even on negative scales
        self.apply_transform(matrix)

    def apply_obb(self):
        """
        Apply the oriented bounding box transform to the current mesh.

        This will result in a mesh with an AABB centered at the origin and the
        same dimensions as the OBB.

        Returns
        ----------
        matrix: (4,4) float, transformation matrix that was applied
                             to mesh to move it into OBB frame.
        """
        matrix = self.bounding_box_oriented.primitive.transform
        matrix = np.linalg.inv(matrix)
        self.apply_transform(matrix)
        return matrix

    def apply_transform(self, matrix):
        """
        Transform mesh by a homogenous transformation matrix.

        Also does bookkeeping to avoid recomputing things, this function
        should be used rather than directly modifying self.vertices
        if possible

        Parameters
        ----------
        matrix: (4,4) float, homogenous transformation matrix
        """

        matrix = np.asanyarray(matrix, order='C', dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError('Transformation matrix must be (4,4)!')

        if np.allclose(matrix, np.eye(4)):
            log.debug('apply_tranform recieved identity matrix')
            return

        new_vertices = transformations.transform_points(self.vertices,
                                                        matrix)

        if self._center_mass is not None:
            self._center_mass = transformations.transform_points(
                np.array([self._center_mass, ]), matrix)[0]

        # force generation of face normals so we can check against them
        new_normals = np.dot(matrix[0:3, 0:3], self.face_normals.T).T
        # easier than figuring out what the scale factor of the matrix is
        unitized, valid = util.unitize(new_normals, check_valid=True)
        # invalid normals are zero anyway
        new_normals[valid] = unitized

        # check the first face against the first normal to check winding
        aligned_pre = triangles.windings_aligned(self.vertices[self.faces[:1]],
                                                 self.face_normals[:1])[0]
        aligned_post = triangles.windings_aligned(new_vertices[self.faces[:1]],
                                                  new_normals[:1])[0]
        if aligned_pre != aligned_post:
            log.debug('Triangle normals not aligned after transform; flipping')
            self.faces = np.fliplr(self.faces)
        with self._cache:
            self.vertices = new_vertices
            self.face_normals = new_normals
        self._cache.clear(exclude=['face_normals'])

        log.debug('Mesh transformed by matrix, normals restored to cache')
        return self

    def voxelized(self, pitch, max_iter=10):
        """
        Return a Voxel object representing the current mesh
        discretized into voxels at the specified pitch

        Parameters
        ----------
        pitch: float, the edge length of a single voxel

        Returns
        ----------
        voxelized: Voxel object representing the current mesh
        """
        voxelized = voxel.VoxelMesh(self,
                                    pitch=pitch,
                                    max_iter=max_iter)
        return voxelized

    def outline(self, face_ids=None):
        """
        Given a set of face ids, find the outline of the faces,
        and return it as a Path3D.

        The outline is defined here as every edge which is only
        included by a single triangle.

        Note that this implies a non-watertight section,
        and the 'outline' of a watertight mesh is an empty path.

        Parameters
        ----------
        face_ids: (n) int, list of indices for self.faces to
                  compute the outline of.
                  If None, outline of full mesh will be computed.
        Returns
        ----------
        path:     Path3D object of the outline
        """
        from .path.io.misc import faces_to_path
        from .path.io.load import _create_path

        path = _create_path(**faces_to_path(self, face_ids))
        return path

    @util.cache_decorator
    def area(self):
        """
        Summed area of all triangles in the current mesh.

        Returns
        ---------
        area: float, surface area of mesh
        """
        area = self.area_faces.sum()
        return area

    @util.cache_decorator
    def area_faces(self):
        """
        The area of each face in the mesh.

        Returns
        ---------
        area_faces: (n,) float, area of each face.
        """
        area_faces = triangles.area(crosses=self.triangles_cross, sum=False)
        return area_faces

    @util.cache_decorator
    def mass_properties(self):
        """
        Returns the mass properties of the current mesh.

        Assumes uniform density, and result is probably garbage if mesh
        isn't watertight.

        Returns
        ----------
        properties: dict, with keys:
          'volume'      : in global units^3
          'mass'        : From specified density
          'density'     : Included again for convenience (same as kwarg density)
          'inertia'     : Taken at the center of mass and aligned with global
                         coordinate system
          'center_mass' : Center of mass location, in global coordinate system
        """
        mass = triangles.mass_properties(triangles=self.triangles,
                                         crosses=self.triangles_cross,
                                         density=self._density,
                                         center_mass=self._center_mass,
                                         skip_inertia=False)

        # if magical clean- up mode is enabled
        # and mesh is watertight/wound correctly but with negative
        # volume it means that every triangle is probably facing
        # inwards, so we invert it in- place without dumping cache
        if (self._validate and
            self.is_watertight and
            self.is_winding_consistent and
            np.linalg.det(mass['inertia']) < 0.0 and
            mass['mass'] < 0.0 and
                mass['volume'] < 0.0):

            # negate mass properties so we don't need to recalculate
            mass['inertia'] = -mass['inertia']
            mass['mass'] = -mass['mass']
            mass['volume'] = -mass['volume']
            # invert the faces and normals of the mesh
            self.invert()
        return mass

    def invert(self):
        """
        Invert the mesh in- place by reversing the winding of every face
        and negating normals without dumping the cache.

        Alters
        ---------
        self.faces:          columns reversed
        self.face_normals:   negated if defined
        self.vertex_normals: negated if defined
        """
        with self._cache:
            if 'face_normals' in self._cache:
                self.face_normals *= -1.0
            if 'vertex_normals' in self._cache:
                self.vertex_normals *= -1.0
            self.faces = np.fliplr(self.faces)

    def scene(self, **kwargs):
        """
        Get a Scene object containing the current mesh.

        Returns
        ---------
        trimesh.scene.scene.Scene object, containing the current mesh
        """
        return Scene(self, **kwargs)

    def show(self, **kwargs):
        """
        Render the mesh in an opengl window. Requires pyglet.

        Parameters
        -----------
        smooth: bool, run smooth shading on mesh or not.
                      Large meshes will be slow

        Returns
        -----------
        scene: trimesh.scene.Scene object, of scene with current mesh in it
        """
        scene = self.scene()
        scene.show(**kwargs)
        return scene

    def submesh(self, faces_sequence, **kwargs):
        """
        Return a subset of the mesh.

        Parameters
        ----------
        faces_sequence: sequence of face indices from mesh
        only_watertight: only return submeshes which are watertight.
        append: return a single mesh which has the faces appended.
                 if this flag is set, only_watertight is ignored

        Returns
        ---------
        if append: Trimesh object
        else:      list of Trimesh objects
        """
        return util.submesh(mesh=self,
                            faces_sequence=faces_sequence,
                            **kwargs)

    @util.cache_decorator
    def identifier(self):
        """
        Return a float vector which is unique to the mesh
        and is robust to rotation and translation.

        Returns
        -----------
        identifier: (6,) float
        """
        identifier = comparison.identifier_simple(self)
        return identifier

    @util.cache_decorator
    def identifier_md5(self):
        """
        An MD5 of the rotation invarient identifier vector

        Returns
        ---------
        hashed: str, MD5 hash of the identifier vector
        """
        hashed = comparison.identifier_hash(self.identifier)
        return hashed

    def export(self, file_obj=None, file_type=None):
        """
        Export the current mesh to a file object.
        If file_obj is a filename, file will be written there.

        Supported formats are stl, off, ply, collada, json, dict, glb,
        dict64, msgpack.

        Parameters
        ---------
        file_obj: open writeable file object
                  str, file name where to save the mesh
                  None, if you would like this function to return the export blob
        file_type: str, which file type to export as.
                   If file name is passed this is not required
        """
        return export_mesh(mesh=self,
                           file_obj=file_obj,
                           file_type=file_type)

    def to_dict(self):
        """
        Return a dictionary representation of the current mesh, with keys
        that can be used as the kwargs for the Trimesh constructor, eg:

        a = Trimesh(**other_mesh.to_dict())

        Returns
        ----------
        result: dict, with keys that match trimesh constructor
        """
        result = self.export(file_type='dict')
        return result

    def convex_decomposition(self, engine=None, maxhulls=20, **kwargs):
        """
        Compute an approximate convex decomposition of a mesh.

        testVHACD Parameters which can be passed as kwargs:

        Name                                        Default
        -----------------------------------------------------
        resolution                                  100000
        max. concavity                              0.001
        plane down-sampling                         4
        convex-hull down-sampling                   4
        alpha                                       0.05
        beta                                        0.05
        maxhulls                                    10
        pca                                         0
        mode                                        0
        max. vertices per convex-hull               64
        min. volume to add vertices to convex-hulls 0.0001
        convex-hull approximation                   1
        OpenCL acceleration                         1
        OpenCL platform ID                          0
        OpenCL device ID                            0
        output                                      output.wrl
        log                                         log.txt


        Parameters
        ----------
        mesh:      Trimesh object
        maxhulls:  int, maximum number of convex hulls to return
        engine:    string, which backend to use. Valid choice is 'vhacd'.
        **kwargs:  testVHACD keyword arguments

        Returns
        -------
        meshes: list of Trimesh objects, a set of nearly convex meshes
                                         that approximate the original
        """
        result = decomposition.convex_decomposition(self,
                                                    engine=engine,
                                                    maxhulls=maxhulls,
                                                    **kwargs)
        return result

    def union(self, other, engine=None):
        """
        Boolean union between this mesh and n other meshes

        Parameters
        ---------
        other: Trimesh, or list of Trimesh objects

        Returns
        ---------
        union: Trimesh, union of self and other Trimesh objects
        """
        result = boolean.union(meshes=np.append(self, other),
                               engine=engine)
        return result

    def difference(self, other, engine=None):
        """
        Boolean difference between this mesh and n other meshes

        Parameters
        ---------
        other: Trimesh, or list of Trimesh objects

        Returns
        ---------
        difference: Trimesh, difference between self and other Trimesh objects
        """
        result = boolean.difference(meshes=np.append(self, other),
                                    engine=engine)
        return result

    def intersection(self, other, engine=None):
        """
        Boolean intersection between this mesh and n other meshes

        Parameters
        ---------
        other: Trimesh, or list of Trimesh objects

        Returns
        ---------
        intersection: Trimesh of the volume contained by all passed meshes
        """
        result = boolean.intersection(meshes=np.append(self, other),
                                      engine=engine)
        return result

    def contains(self, points):
        """
        Given a set of points, determine whether or not they are inside the mesh.
        This raises an error if called on a non- watertight mesh.

        Parameters
        ---------
        points: (n,3) set of points in space

        Returns
        ---------
        contains: (n) boolean array, whether or not a point is inside the mesh
        """
        if not self.is_watertight:
            log.warning('Mesh is non- watertight for contained point query!')
        contains = self.ray.contains_points(points)
        return contains

    def copy(self):
        """
        Safely get a copy of the current mesh.

        Copied objects will have emptied caches to avoid memory issues and
        so may be slow on initial operations until caches are regenerated.

        Current object will *not* have its cache cleared.

        Returns
        ---------
        copied: copy of current mesh
        """
        copied = Trimesh()

        # copy vertex and face data
        copied._data.data = copy.deepcopy(self._data.data)
        # copy visual information
        copied.visual._data.data = copy.deepcopy(self.visual._data.data)
        # get metadata
        copied.metadata = copy.deepcopy(self.metadata)
        # get center_mass and density
        if self._center_mass is not None:
            copied.center_mass = self.center_mass
        copied._density = self._density

        # make sure cache is set from here
        copied._cache.id_set()

        return copied

    def eval_cached(self, statement, *args):
        """
        Evaluate a statement and cache the result before returning.

        Statements are evaluated inside the Trimesh object, and

        Parameters
        -----------
        statement: str, statement of valid python code
        *args:     available inside statement as args[0], etc

        Returns
        -----------
        result: result of running eval on statement with args

        Examples
        -----------
        r = mesh.eval_cached('np.dot(self.vertices, args[0])', [0,0,1])
        """

        statement = str(statement)
        key = 'eval_cached_' + statement
        key += '_'.join(str(i) for i in args)

        if key in self._cache:
            return self._cache[key]

        result = eval(statement)
        self._cache[key] = result
        return result

    def __hash__(self):
        """
        Return the MD5 hash of the mesh as an integer.

        Returns
        ----------
        hashed: int, MD5 of mesh data
        """
        hashed = int(self.md5(), 16)
        return hashed

    def __add__(self, other):
        """
        Concatenate the mesh with another mesh.

        Parameters
        ------------
        other: Trimesh object, to combine with self

        Returns
        ----------
        concat: Trimesh object of combined result
        """
        concat = util.concatenate(self, other)
        return concat
