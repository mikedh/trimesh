'''
github.com/mikedh/trimesh

Library for importing, exporting and doing simple operations on triangular meshes.
'''

import numpy as np

from copy import deepcopy

from . import triangles
from . import grouping
from . import geometry
from . import graph
from . import visual
from . import sample
from . import repair
from . import comparison
from . import boolean
from . import intersections
from . import transformations
from . import util
from . import convex
from . import remesh
from . import bounds
from . import units
from . import permutate
from . import nsphere

from .io.export    import export_mesh
from .ray.ray_mesh import RayMeshIntersector, contains_points
from .voxel        import Voxel
from .constants    import log, _log_time, tol
from .scene        import Scene

class Trimesh(object):
    def __init__(self,
                 vertices       = None, 
                 faces          = None, 
                 face_normals   = None, 
                 vertex_normals = None,
                 metadata       = None,
                 process        = True,
                 **kwargs):
        '''
        A Trimesh object contains a triangular 3D mesh. 

        Arguments
        ----------
        vertices:       (n,3) float set of vertex locations
        faces:          (m,3) int set of triangular faces (quad faces will be triangulated)
        face_normals:   (m,3) float set of normal vectors for faces. Passing these only
                        serves as a speedup as otherwise they will be computed with 
                        crossproducts
        vertex_normals: (n,3) float set of normal vectors for vertices
        metadata:       dict, any metadata about the mesh
        process:        bool, if True basic mesh cleanup will be done on instantiation
        '''
        # self._data stores information about the mesh which CANNOT be regenerated.
        # in the base class all that is stored here is vertex and face information
        # any data put into the store is converted to a TrackedArray (np.ndarray subclass)
        # which provides an md5() method which can be used to detect changes in the array.
        self._data = util.DataStore()

        # self._cache stores information about the mesh which CAN be regenerated from
        # self._data, but may be slow to calculate. In order to maintain consistency
        # the cache is cleared when self._data.md5() changes
        self._cache = util.Cache(id_function = self._data.md5)

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
            self.visual = visual.VisualAttributes(**kwargs)
        self.visual.mesh = self
        
        # normals are accessed through setters/properties and are regenerated if the
        # dimensions are inconsistant, but can be set by the constructor to save
        # the substantial number of cross products required to generate them
        if face_normals is not None:
            self.face_normals = face_normals

        # (n, 3) float of vertex normals, can be created from face normals
        if vertex_normals is not None:
            self.vertex_normals = vertex_normals

        # create a ray-mesh query object for the current mesh
        # initializing is very inexpensive and object is convenient to have.
        # On first query expensive bookkeeping is done (creation of r-tree),
        # and is cached for subsequent queries
        self.ray = RayMeshIntersector(self)

        # a quick way to get permuated versions of the current mesh
        self.permutate = permutate.Permutator(self)
        
        # store metadata about the mesh in a dictionary
        self.metadata = dict()
        # update the mesh metadata with passed metadata
        if isinstance(metadata, dict):
            self.metadata.update(metadata)

        # on returning faces or face normals, validate faces to ensure nonzero 
        # normals and matching shape. Not validating can mean that you get different
        # number of values depending on the order which you look at faces and face normals,
        # but for some operations validation may want to be turned off during the operation
        # then reinitialized for the end of the operation. 
        self._validate = True

        # process is a cleanup function which brings the mesh to a consistant state
        # by merging vertices and removing zero- area and duplicate faces
        if (process and
            vertices is not None and
            faces    is not None):
            self.process()

        # store all passed kwargs for debugging purposes
        self._kwargs = kwargs

    def process(self):
        '''
        Convenience function to remove garbage and make mesh sane.
        
        Does this by:
            1) merging duplicate vertices
            2) removing duplicate faces
            3) removing zero- area faces
            
        On an 234213 face mesh this function executes in .0005s 
        with a late 2014 i7.
        '''
        # avoid clearing the cache during operations
        with self._cache:
            self.merge_vertices()
            self.remove_duplicate_faces()
            self.remove_degenerate_faces()
        # since none of our process operations moved vertices or faces,
        # we can keep face and vertex normals in the cache without recomputing
        # if faces or vertices have been removed, normals are validated before
        # being returned so there is no danger of inconsistent dimensions
        self._cache.clear(exclude = ['face_normals',
                                     'vertex_normals'])
        return self

    def md5(self):
        '''
        An MD5 of the core geometry information for the mesh (faces and vertices).
        Generated from TrackedArray, which subclasses np.ndarray to monitor for 
        changes and returns a correct, but lazily evaluated md5 (so it only has to
        recalculate the hash occasionally, rather than on every call)

        Returns
        ----------
        md5: string, appended md5 hashes of numpy arrays for faces and vertices
        '''
        md5 = self._data.md5()
        return md5

    @property
    def faces(self):
        '''
        The faces of the mesh.
        
        This is regarded as core information which cannot be regenerated from cache,
        and as such is stored in self._data, which tracks the array for changes
        and clears cached values of the mesh if this is altered. 
        
        Returns
        ----------
        faces: (n,3) int, representing triangles which reference self.vertices
        '''
        # we validate face normals as the validation process may remove
        # zero- area faces. If we didn't do this check here, the shape of 
        # self.faces and self.face_normals could differ depending on the 
        # order they were queried in
        self._validate_face_normals()
        return self._data['faces']
        
    @faces.setter
    def faces(self, values):
        if values is None: values = []
        values = np.asanyarray(values, dtype=np.int64)
        if util.is_shape(values, (-1,4)):
            log.info('Triangulating quad faces')
            values = geometry.triangulate_quads(values)
        self._data['faces'] = values
        
    @util.cache_decorator
    def faces_sparse(self):
        '''
        A sparse matrix representation of the faces.
        
        Returns
        ----------
        sparse: scipy.sparse.coo_matrix with dtype=bool, shape=(len(self.vertices), len(self.faces)) 
        '''
        sparse = geometry.index_sparse(column_count = len(self.vertices),
                                       indices      = self.faces)
        return sparse

    @property
    def face_normals(self):
        self._validate_face_normals()
        cached = self._cache['face_normals']
        return cached

    @face_normals.setter
    def face_normals(self, values):
        if values is None: return
        self._cache['face_normals'] = np.asanyarray(values, dtype=np.float64)
        
    @property
    def vertices(self):
        '''
        The vertices of the mesh.
        
        This is regarded as core information which cannot be regenerated from cache,
        and as such is stored in self._data, which tracks the array for changes
        and clears cached values of the mesh if this is altered. 
        
        Returns
        ----------
        vertices: (n,3) float representing points in cartesian space
        '''
        return self._data['vertices']
        
    @vertices.setter
    def vertices(self, values):
        # make sure vertices are stored as a float64 for consistency
        self._data['vertices'] = np.asanyarray(values, dtype=np.float64)

    def _validate_face_normals(self, faces=None):
        '''
        Make sure face normals are of correct shape. 

        This function also removes faces which are zero area, and so must be 
        called before returning faces or triangles to avoid inconsistant results
        depending on which order functions are called.
        
        Arguments
        ----------
        faces: (n,3) int, if None uses self.faces
               Available as an argument to avoid a circular reference in some functions
        '''
        if not self._validate: return 
        # pull faces directly from DataStore to avoid infinite recursion
        if faces is None:
            faces = self._data['faces']
        if np.shape(self._cache.get('face_normals')) != np.shape(faces):
            log.debug('Generating face normals as shape was incorrect')
            tri_cached = self.vertices.view(np.ndarray)[faces]
            face_normals, valid = triangles.normals(tri_cached)
            self.update_faces(valid)
            self._cache['face_normals'] = face_normals

    @util.cache_decorator
    def vertex_normals(self):
        '''
        The vertex normals of the mesh. If the normals were loaded, we check to 
        make sure we have the same number of vertex normals and vertices before
        returning them. If there are no vertex normals defined, or a shape mismatch
        we calculate the vertex normals from the mean normals of the faces the vertex
        is used in.
        
        Returns
        ----------
        vertex_normals: (n,3) float, where n == len(self.vertices)
                         Represents the surface normal at each vertex. 
        '''
        assert hasattr(self.faces_sparse, 'dot')
        vertex_normals = geometry.mean_vertex_normals(len(self.vertices),
                                                      self.faces,
                                                      self.face_normals,
                                                      sparse = self.faces_sparse)
        return vertex_normals

    @vertex_normals.setter
    def vertex_normals(self, values):
        if values is None: return
        values = np.asanyarray(values, dtype=np.float64)
        if values.shape == self.vertices.shape:
            self._cache['vertex_normals'] = values

    @util.cache_decorator
    def bounding_box(self):
        '''
        An axis aligned bounding box for the current mesh.
        
        Returns
        ----------
        aabb: trimesh.primitives.Box object with transform and extents defined
              to represent the axis aligned bounding box of the mesh
        '''
        from . import primitives
        aabb = primitives.Box(transform = transformations.translation_matrix(self.bounds.mean(axis=0)),
                              extents   = self.extents,
                              mutable   = False)
        return aabb

    @util.cache_decorator
    def bounding_box_oriented(self):
        '''
        An oriented bounding box for the current mesh.
        
        Returns
        ---------
        obb: trimesh.primitives.Box object with transform and extents defined
             to represent the minimum volume oriented bounding box of the mesh
        '''
        from . import primitives
        to_origin, extents = bounds.oriented_bounds(self)
        obb = primitives.Box(transform = np.linalg.inv(to_origin),
                             extents   = extents,
                             mutable   = False)
        return obb

    @util.cache_decorator
    def bounding_sphere(self):
        '''
        A minimum volume bounding sphere for the current mesh.

        Note that the Sphere primitive returned has an UNPADDED, EXACT
        sphere_radius so while the distance of every vertex of the current 
        mesh from sphere_center will be less than sphere_radius, the faceted 
        sphere primitive may NOT CONTAIN EVER VERTEX

        Returns
        --------
        minball: trimesh.primitives.Sphere object 
        '''
        from . import primitives
        center, radius = nsphere.minimum_nsphere(self)
        minball = primitives.Sphere(center = center,
                                    radius = radius,
                                    mutable   = False)
        return minball

    @util.cache_decorator
    def bounding_cylinder(self):
        '''
        A minimum volume bounding cylinder for the current mesh.

        Returns
        --------
        mincyl: trimesh.primitives.Cylinder object 
        '''
        from . import primitives
        kwargs = bounds.minimum_cylinder(self)
        mincyl = primitives.Cylinder(mutable=False, **kwargs)
        return mincyl
        
    @util.cache_decorator
    def bounding_primitive(self):
        '''
        The minimum volume primitive (box, sphere, or cylinder) that bounds the mesh.
        
        Returns
        ---------
        bounding_primitive: trimesh.primitives.Sphere or 
                            trimesh.primitives.Box
                            trimesh.primitives.Cylinder
        '''
        options = [self.bounding_box_oriented, 
                   self.bounding_sphere,
                   self.bounding_cylinder]
        volume_min = np.argmin([i.volume for i in options])
        bounding_primitive = options[volume_min]
        return bounding_primitive
        
    @util.cache_decorator
    def bounds(self):
        '''
        The axis aligned bounds of the mesh. 

        Returns
        -----------
        bounds: (2,3) float, bounding box with [min, max] coordinates
        '''
        # we use triangles instead of faces because
        # if there is an unused vertex it will screw up the bounds
        in_mesh = self.triangles.reshape((-1,3))
        bounds = np.vstack((in_mesh.min(axis=0),
                            in_mesh.max(axis=0)))
        return bounds

    @util.cache_decorator
    def extents(self):
        '''
        The length, width, and height of the bounding box of the mesh.

        Returns
        -----------
        extents: (3,) float array containing axis aligned [l,w,h]
        '''
        extents = np.diff(self.bounds, axis=0)[0]
        return extents

    @util.cache_decorator
    def scale(self):
        '''
        A metric for the overall scale of the mesh.

        Returns
        ----------
        scale: float, the mean of the bounding box edges rounded to one signifigant figure
        '''
        scale = self.extents.max()
        return scale

    @util.cache_decorator
    def centroid(self):
        '''
        The point in space which is the average of the triangle centroids
        weighted by the area of each triangle.

        This will be valid even for non- watertight meshes, unlike self.center_mass
        
        Returns
        ----------
        centroid: (3,) float, the average vertex
        '''

        # use the centroid of each triangle weighted by
        # the area of the triangle to find the overall centroid
        centroid = np.average(self.triangles_center,
                              axis    = 0,
                              weights = self.area_faces)
        return centroid

    @util.cache_decorator
    def center_mass(self):
        '''
        The point in space which is the center of mass/volume.

        If the current mesh is not watertight, this is meaningless garbage.

        Returns
        -----------
        center_mass: (3,) float array, volumetric center of mass of the mesh
        '''
        if not self.is_watertight:
            log.warning('Center of mass requested for non- watertight mesh! Expect garbage!')
        center_mass = np.array(self.mass_properties(skip_inertia=True)['center_mass'])
        return center_mass

    @util.cache_decorator
    def volume(self):
        '''
        Volume of the current mesh. 
        If the current mesh isn't watertight this is garbage.

        Returns
        ---------
        volume: float, volume of the current mesh
        '''
        volume = self.mass_properties(skip_inertia=True)['volume']
        return volume

    @util.cache_decorator
    def moment_inertia(self):
        '''
        Return the moment of inertia matrix of the current mesh.
        If mesh isn't watertight this is garbage.
 
        Returns
        ---------
        inertia: (3,3) float, moment of inertia of the current mesh.
        '''
        inertia = np.array(self.mass_properties(skip_inertia=False)['inertia'])
        return inertia

    @util.cache_decorator
    def triangles(self):
        '''
        Actual triangles of the mesh (points, not indexes)

        Returns
        ---------
        triangles: (n,3,3) float points of vertices grouped into triangles
        '''
        # use of advanced indexing on our tracked arrays will 
        # trigger a change flag which means the MD5 will have to be
        # recomputed. We can escape this check by viewing the array.
        triangles = self.vertices.view(np.ndarray)[self.faces]
        # make triangles (which are derived from faces/vertices) not writeable
        triangles.flags.writeable = False
        return triangles

    def triangles_tree(self):
        '''
        An R-tree containing each face of the mesh.
        
        Returns
        ----------
        tree: rtree.index where each triangle in self.faces has a rectangular cell
        '''
        tree = triangles.bounds_tree(self.triangles)
        return tree

    @util.cache_decorator
    def triangles_center(self):
        '''
        The center of each triangle (barycentric [1/3, 1/3, 1/3])

        Returns
        ---------
        triangles_center: (len(self.faces), 3) float, center of each triangular face
        '''
        triangles_center = self.triangles.mean(axis=1)
        return triangles_center
        
    @util.cache_decorator
    def triangles_cross(self):
        '''
        The cross product of two edges of each triangle. 

        Returns
        ---------
        crosses: (n,3) float, cross product of each triangle
        '''
        crosses = triangles.cross(self.triangles)
        return crosses

    @util.cache_decorator
    def edges(self):
        '''
        Edges of the mesh (derived from faces).

        Returns
        ---------
        edges: (n,2) int, set of vertex indices
        '''
        edges, index = geometry.faces_to_edges(self.faces.view(np.ndarray), 
                                               return_index=True)
        self._cache['edges_face'] = index
        return edges
        
    @util.cache_decorator
    def edges_face(self):
        '''
        Which face does each edge belong to.

        Returns
        ---------
        edges_face: (n,) int, index of self.faces
        '''
        populate = self.edges
        return self._cache['edges_face']
        
    @util.cache_decorator
    def edges_unique(self):
        '''
        The unique edges of the mesh.

        Returns
        ----------
        edges_unique: (n,2) int, set of vertex indices for unique edges
        '''
        unique, inverse = grouping.unique_rows(self.edges_sorted)
        edges_unique    = self.edges_sorted[unique]
        # edges_unique will be added automatically by the decorator
        # additional terms generated need to be added to the cache manually
        self._cache['edges_unique_idx'] = unique
        self._cache['edges_unique_inv'] = inverse
        return edges_unique
        
    @util.cache_decorator
    def edges_sorted(self):
        '''
        Returns
        ----------
        self.edges, but sorted along axis 1
        '''
        edges_sorted = np.sort(self.edges, axis=1)
        return edges_sorted

    @util.cache_decorator
    def faces_unique_edges(self):
        '''
        For each face return which indexes in mesh.unique_edges constructs that face.

        Returns
        ---------
        faces_unique_edges: self.faces.shape int, which indexes of self.edges_unique
                            construct self.faces

        Example
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
        '''
        # make sure we have populated unique edges
        populate = self.edges_unique
        # we are relying on the fact that edges are stacked in triplets
        result = self._cache['edges_unique_inv'].reshape((-1,3))
        return result

    @util.cache_decorator
    def euler_number(self):
        '''
        Return the Euler characteristic (a topological invariant) for the mesh
        In order to guarantee correctness, this should be called after 
        remove_unreferenced_vertices

        Returns
        ----------
        euler_number: int, topological invarient
        '''
        euler = len(self.vertices) - len(self.edges_unique) + len(self.faces)
        return euler

    @property
    def units(self):
        '''
        Definition of units for the mesh.
        
        Returns
        ----------
        units: str, unit system mesh is in, or None if not defined
        '''
        if 'units' in self.metadata: 
            return self.metadata['units']
        else:
            return None

    @units.setter
    def units(self, value):
        value = str(value).lower()
        if not units.validate(value):
            raise ValueError(value + ' are not a valid unit!')
        self.metadata['units'] = value
        
    def convert_units(self, desired, guess=False):
        '''
        Convert the units of the mesh into a specified unit.
   
        Arguments
        ----------
        desired: string, units to convert to (eg 'inches')
        guess:   boolean, if self.units are not defined should we 
                 guess the current units of the document and then convert?
        '''
        units._set_units(self, desired, guess)

    def merge_vertices(self):
        '''
        If a mesh has vertices that are closer than trimesh.constants.tol.merge
        redefine them to be the same vertex and replace face references
        '''        
        grouping.merge_vertices_hash(self)
        
    def update_vertices(self, mask, inverse=None):
        '''
        Update vertices with a mask.
        
        Arguments
        ----------
        vertex_mask: (len(self.vertices)) boolean array of which
                     vertices to keep
        inverse:     (len(self.vertices)) int array to reconstruct
                     vertex references (such as output by np.unique)
        '''
        mask = np.asanyarray(mask)
        if mask.dtype.name == 'bool' and mask.all(): 
            return
        if len(mask) == 0 or self.is_empty: 
            return
        if inverse is not None:
            self.faces = inverse[[self.faces.reshape(-1)]].reshape((-1,3))
        self.visual.update_vertices(mask)
        cached_normals = self._cache.get('vertex_normals')
        if util.is_shape(cached_normals, (-1,3)):
            try: self.vertex_normals = cached_normals[mask]
            except: pass
        self.vertices = self.vertices[mask]

    def update_faces(self, mask):
        '''
        In many cases, we will want to remove specific faces. 
        However, there is additional bookkeeping to do this cleanly. 
        This function updates the set of faces with a validity mask,
        as well as keeping track of normals and colors.

        Arguments
        ---------
        valid: either (m) int, or (len(self.faces)) bool. 
        '''
        if self.is_empty: return
        mask = np.asanyarray(mask)
        if mask.dtype.name == 'bool':
            if mask.all(): return
        elif mask.dtype.name != 'int':
            mask = mask.astype(np.int)
        cached_normals = self._cache.get('face_normals')
        if util.is_shape(cached_normals, (-1,3)):
            self.face_normals = cached_normals[mask]
            #except: pass
        faces = self._data['faces']
        # if Trimesh has been subclassed and faces have been moved from data 
        # to cache, get faces from cache. 
        if not util.is_shape(faces, (-1,3)):
            faces = self._cache['faces']
        self.faces = faces[mask]        
        self.visual.update_faces(mask)
        
    def remove_duplicate_faces(self):
        '''
        On the current mesh remove any faces which are duplicates. 
        '''
        unique, inverse = grouping.unique_rows(np.sort(self.faces, axis=1))
        self.update_faces(unique)
        
    def rezero(self):
        '''
        Translate the mesh so that all vertex vertices are positive.
        '''
        self.apply_translation(self.bounds[0] * -1.0)

    @_log_time
    def split(self, only_watertight=True, adjacency=None):
        '''
        Returns a list of Trimesh objects, based on face connectivity.
        Splits into individual components, sometimes referred to as 'bodies'

        Arguments
        ---------
        only_watertight: only meshes which are watertight are returned
        adjacency: if not None, override face adjacency with custom values (n,2)

        Returns
        ---------
        meshes: (n) list of Trimesh objects
        '''
        meshes = graph.split(self, 
                             only_watertight = only_watertight,
                             adjacency       = adjacency)
        log.info('split found %i components', len(meshes))
        return meshes
        
    @util.cache_decorator
    def face_adjacency(self):
        '''
        Find faces that share an edge, which we call here 'adjacent'.
        
        Returns
        ----------
        adjacency: (n,2) int, pairs of faces which share an edge
        
        Example
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
        '''
        adjacency, edges = graph.face_adjacency(mesh = self,
                                                return_edges = True)
        self._cache['face_adjacency_edges'] = edges
        return adjacency
 
    @util.cache_decorator
    def face_adjacency_edges(self):
        '''
        Returns the edges that are shared by the adjacent faces.

        Returns
        --------
        edges: (n, 2) list of vertex indices which correspond to face_adjacency
        '''
        # this value is calculated as a byproduct of the face adjacency
        populate = self.face_adjacency
        return self._cache['face_adjacency_edges']
        
    @util.cache_decorator    
    def face_adjacency_angles(self):
        '''
        Return the angle between adjacent faces
        
        Returns
        --------
        adjacency_angle: (n,) float angle between adjacent faces.
                         Each value corresponds with self.face_adjacency
        '''
        pairs = self.face_normals[self.face_adjacency]
        angles = geometry.vector_angle(pairs)
        return angles
        
    @util.cache_decorator
    def is_winding_consistent(self):
        '''
        Does the mesh have consistent winding or not. 
        A mesh with consistent winding has each shared edge 
        going in an opposite direction from the other in the pair.
        
        Returns
        --------
        consistent: bool, if winding is consistent or not
        '''
        # consistent winding check is populated into the cache by is_watertight query
        populate = self.is_watertight
        return self._cache['is_winding_consistent']

    @util.cache_decorator
    def is_watertight(self):
        '''
        Check if a mesh is watertight by making sure every edge is used by two faces.

        Returns
        ----------
        is_watertight: bool, is mesh watertight or not
        '''

        watertight, reversed = graph.is_watertight(self.edges, 
                                                   return_winding=True)
        self._cache['is_watertight'] = watertight
        self._cache['is_winding_consistent'] = reversed
        return watertight

    @util.cache_decorator
    def is_empty(self):
        '''
        Does the current mesh have data defined.

        Returns
        --------
        empty: if True, no data exists in the mesh.
        '''
        return self._data.is_empty()

    @util.cache_decorator
    def is_convex(self):
        '''
        Check if a mesh is convex or not.

        Returns
        ----------
        is_convex: bool, is mesh convex or not
        '''
        is_convex = convex.is_convex(self)
        return is_convex

    def kdtree(self):
        '''
        Return a scipy.spatial.cKDTree of the vertices of the mesh.
        Not cached as this lead to observed memory issues and segfaults.

        Returns
        ---------
        tree: scipy.spatial.cKDTree containing mesh vertices 
        '''

        from scipy.spatial import cKDTree as KDTree
        tree = KDTree(self.vertices.view(np.ndarray))
        return tree

    def remove_degenerate_faces(self):
        '''
        Remove degenerate faces (faces without 3 unique vertex indices) 
        from the current mesh.
        '''
        nondegenerate = triangles.nondegenerate(self.triangles)
        self.update_faces(nondegenerate)

    def facets(self, return_area=False):
        '''
        Return a list of face indices for coplanar adjacent faces.

        Arguments
        ---------
        return_area: boolean, if True return area of each group of faces

        Returns
        ---------
        facets: (n) sequence of face indices
        area:   (n) float list of face group area (if return_area)
        '''
        key = 'facets_' + str(int(return_area))
        cached = self._cache[key]
        if cached is not None:
            return cached

        facets = graph.facets(self)
        if return_area:
            area = np.array([self.area_faces[i].sum() for i in facets])
            result = (facets, area)
        else: 
            result = facets
        self._cache[key] = result
        return result
            
    @_log_time    
    def fix_normals(self):
        '''
        Find and fix problems with self.face_normals and self.faces winding direction.
        
        For face normals ensure that vectors are consistently pointed outwards,
        and that self.faces is wound in the correct direction for all connected components.
        '''
        repair.fix_normals(self)

    def fill_holes(self):
        '''
        Fill single triangle and single quad holes in the current mesh.
        
        Returns
        ----------
        watertight: bool, is the mesh watertight after the function completes
        '''
        return repair.fill_holes(self)

    def subdivide(self, face_index=None):
        '''
        Subdivide a mesh, with each subdivided face replaced with four 
        smaller faces.

        Arguments
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
        '''
        remesh.subdivide(self, face_index=face_index)

    @_log_time
    def smoothed(self, angle=.4):
        '''
        Return a version of the current mesh which will render nicely.
        Does not change current mesh in any way.

        Returns
        ---------
        smoothed: Trimesh object, non watertight version of current mesh
                  which will render nicely with smooth shading. 
        '''
        
        # smooth should be recomputed if visuals change, so we 
        # store it in the visuals cache rather than the main mesh cache
        cached = self.visual._cache.get('smoothed')
        if cached is not None:
            return cached
        return self.visual._cache.set(key   = 'smoothed',
                                      value = graph.smoothed(self, angle))

    def section(self,
                plane_normal,
                plane_origin):
        '''
        Returns a cross section of the current mesh and plane defined by
        origin and normal.
        
        Arguments
        ---------
        plane_normal: (3) vector for plane normal
        plane_origin: (3) vector for plane origin
                            
        Returns
        ---------
        intersections: Path3D of intersections
        '''
        
        from .io.load      import load_path
        lines = intersections.mesh_plane(mesh = self, 
                                         plane_normal = plane_normal, 
                                         plane_origin = plane_origin)
        if len(lines) == 0:
            raise ValueError('Specified plane doesn\'t intersect mesh!')
        path = load_path(lines)
        return path

    @util.cache_decorator
    def convex_hull_raw(self):
        '''
        The raw convex hull from qhull which has inconsistent face winding 
        and face normals. This is exposed as fixing winding (as in mesh.convex_hull)
        is quite expensive, and if normals or winding aren't required to be consistent
        using this attribute will result in a signifigant speedup.
        
        Returns
        ---------
        hull: Trimesh object, raw from qhull with backwards faces
        '''
        hull = convex.convex_hull(self, clean=False)
        return hull

    @util.cache_decorator
    def convex_hull(self):
        '''
        Get a new Trimesh object representing the convex hull of the 
        current mesh. Requires scipy >.12.

        Returns
        --------
        convex: Trimesh object of convex hull of current mesh
        '''
        hull = self.convex_hull_raw.copy()
        hull.fix_normals()
        return hull

    def sample(self, count):
        '''
        Return random samples distributed normally across the 
        surface of the mesh

        Arguments
        ---------
        count: int, number of points to sample

        Returns
        ---------
        samples: (count, 3) float, points on surface of mesh
        '''
        return sample.sample_surface(self, count)
            
    def remove_unreferenced_vertices(self):
        '''
        Remove all vertices in the current mesh which are not referenced by a face. 
        '''
        unique, inverse = np.unique(self.faces.reshape(-1), return_inverse=True)
        self.faces      = inverse.reshape((-1,3))          
        self.vertices   = self.vertices[unique]

    def unmerge_vertices(self):
        '''
        Removes all face references so that every face contains three unique 
        vertex indices and no faces are adjacent.
        '''
        with self._cache:
            self.update_vertices(mask = self.faces.reshape(-1))
            self.faces = np.arange(len(self.vertices)).reshape((-1,3))
        self._cache.clear(exclude='face_normals')

    def apply_translation(self, translation):
        '''
        Translate the current mesh.
        
        Arguments
        ----------
        translation: (3,) float, translation in XYZ
        '''
        translation = np.asanyarray(translation, dtype=np.float64)
        if translation.shape != (3,):
            raise ValueError('Translation must be (3,)!')
            
        with self._cache:
            self.vertices += translation
        # we are doing a simple translation so normals are preserved
        self._cache.clear(exclude = ['face_normals',
                                     'vertex_normals'])
        
    def apply_scale(self, scaling):
        '''
        Scale the mesh equally on all axis.

        Arguments
        ----------
        scaling: float, scale factor
        '''
        scaling = float(scaling)
        if not np.isfinite(scaling):
            raise ValueError('Scaling factor must be finite number!')
            
        matrix = np.eye(4)
        matrix[:3,:3] *= scaling
        # apply_transform will work nicely even on negative scales
        self.apply_transform(matrix)

    def apply_transform(self, matrix):
        '''
        Transform mesh by a homogenous transformation matrix.
        Also transforms normals to avoid having to recompute them.

        Arguments
        ----------
        matrix: (4,4) float, homogenous transformation matrix
        '''
        
        matrix = np.asanyarray(matrix, dtype=np.float64)
        if matrix.shape != (4,4):
            raise ValueError('Transformation matrix must be (4,4)!')

        if np.allclose(matrix, np.eye(4)):
            log.debug('apply_tranform recieved identity matrix, returning without applying')
            return
            
        new_normals = np.dot(matrix[0:3, 0:3], self.face_normals.T).T
        # easier than figuring out what the scale factor of the matrix is
        new_normals = util.unitize(new_normals)
        new_vertices = transformations.transform_points(self.vertices, matrix)
        # check the first face against the first normal to see if winding is correct
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
        self._cache.clear(exclude = ['face_normals'])

        log.debug('Mesh transformed by matrix, normals restored to cache')
        return self

    def voxelized(self, pitch):
        '''
        Return a Voxel object representing the current mesh
        discretized into voxels at the specified pitch

        Arguments
        ----------
        pitch: float, the edge length of a single voxel

        Returns
        ----------
        voxelized: Voxel object representing the current mesh
        '''
        voxelized = Voxel(self, pitch)
        return voxelized

    def outline(self, face_ids=None):
        '''
        Given a set of face ids, find the outline of the faces,
        and return it as a Path3D. 

        The outline is defined here as every edge which is only 
        included by a single triangle.

        Note that this implies a non-watertight section, 
        and the 'outline' of a watertight mesh is an empty path. 

        Arguments
        ----------
        face_ids: (n) int, list of indices for self.faces to 
                  compute the outline of. 
                  If None, outline of full mesh will be computed.
        Returns
        ----------
        path:     Path3D object of the outline
        '''
        from .path.io.misc import faces_to_path
        from .path.io.load import _create_path

        path = _create_path(**faces_to_path(self, face_ids))
        return path
        
    @util.cache_decorator
    def area(self):
        '''
        Summed area of all triangles in the current mesh.
        
        Returns
        ---------
        area: float, surface area of mesh
        '''
        area = self.area_faces.sum()
        return area

    @util.cache_decorator    
    def area_faces(self):
        '''
        The area of each face in the mesh.
        
        Returns
        ---------
        area_faces: (n,) float, area of each face. 
        '''
        area_faces = triangles.area(crosses = self.triangles_cross, sum=False)
        return area_faces
                         
    def mass_properties(self, density=1.0, skip_inertia=False):
        '''
        Returns the mass properties of the current mesh.
        
        Assumes uniform density, and result is probably garbage if mesh
        isn't watertight. 
        
        Arguments
        ----------
        density:      float, density of the solid
        skip_inertia: bool, skip inertia calculation or not
        
        Returns
        ----------
        properties: dict, with keys: 
                    'volume'      : in global units^3
                    'mass'        : From specified density
                    'density'     : Included again for convenience (same as kwarg density)
                    'inertia'     : Taken at the center of mass and aligned with global 
                                    coordinate system
                    'center_mass' : Center of mass location, in global coordinate system
        '''
        key  = 'mass_properties_' 
        key += str(int(skip_inertia)) + '_'
        key += str(int(density * 1e5))
        cached = self._cache[key]
        if cached is not None: 
            return cached
        mass = triangles.mass_properties(triangles    = self.triangles,
                                         crosses      = self.triangles_cross,
                                         density      = density,
                                         skip_inertia = skip_inertia)
        self._cache[key] = mass
        return mass

    def scene(self):
        '''
        Get a Scene object containing the current mesh.
        
        Returns
        ---------
        trimesh.scene.scene.Scene object, containing the current mesh
        '''
        return Scene(self)

    def show(self, block=True, **kwargs):
        '''
        Render the mesh in an opengl window. Requires pyglet.
        
        Arguments
        -----------
        block:  bool, open window in new thread, or block until window is closed
        smooth: bool, run smooth shading on mesh or not. Large meshes will be slow

        Returns
        -----------
        scene: trimesh.scene.Scene object, of scene with current mesh in it
        '''
        scene = self.scene()
        scene.show(block=block, **kwargs)
        return scene

    def submesh(self, faces_sequence, **kwargs): 
        '''
        Return a subset of the mesh.
        
        Arguments
        ----------
        faces_sequence: sequence of face indices from mesh
        only_watertight: only return submeshes which are watertight. 
        append: return a single mesh which has the faces specified appended.
                 if this flag is set, only_watertight is ignored

        Returns
        ---------
        if append: Trimesh object
        else:      list of Trimesh objects
        '''
        return util.submesh(mesh = self,
                            faces_sequence=faces_sequence, 
                            **kwargs)

    @util.cache_decorator
    def identifier(self):
        '''
        Return a float vector which is unique to the mesh
        and is robust to rotation and translation.

        Returns
        -----------
        identifier: (tol.id_len,) float
        '''
        identifier = comparison.rotationally_invariant_identifier(self, tol.id_len)
        return identifier

    def export(self,file_obj=None, file_type='stl'):
        '''
        Export the current mesh to a file object. 
        If file_obj is a filename, file will be written there. 

        Supported formats are stl, off, ply, collada, json, dict, dict64, msgpack.

        Arguments
        ---------
        file_obj: open writeable file object 
                  str, file name where to save the mesh
                  None, if you would like this function to return the export blob
        file_type: str, which file type to export as. 
                   If file name is passed this is not required
        '''
        return export_mesh(mesh = self, 
                           file_obj  = file_obj, 
                           file_type = file_type)

    def to_dict(self):
        '''
        Return a dictionary representation of the current mesh, with keys 
        that can be used as the kwargs for the Trimesh constructor, eg:
        
        a = Trimesh(**other_mesh.to_dict())

        Returns
        ----------
        result: dict, with keys that match trimesh constructor
        '''
        result = self.export(file_type='dict')
        return result

    def union(self, other, engine=None):
        '''
        Boolean union between this mesh and n other meshes

        Arguments
        ---------
        other: Trimesh, or list of Trimesh objects

        Returns
        ---------
        union: Trimesh, union of self and other Trimesh objects
        '''
        return Trimesh(process=True, 
                       **boolean.union(meshes = np.append(self, other), 
                                       engine = engine))

    def difference(self, other, engine=None):
        '''
        Boolean difference between this mesh and n other meshes

        Arguments
        ---------
        other: Trimesh, or list of Trimesh objects

        Returns
        ---------
        difference: Trimesh, difference between self and other Trimesh objects
        '''
        return Trimesh(process=True, 
                       **boolean.difference(meshes = np.append(self, other), 
                                            engine = engine))
        
    def intersection(self, other, engine=None):
        '''
        Boolean intersection between this mesh and n other meshes

        Arguments
        ---------
        other: Trimesh, or list of Trimesh objects

        Returns
        ---------
        intersection: Trimesh of the volume contained by all passed meshes
        '''
        return Trimesh(process=True, 
                       **boolean.intersection(meshes = np.append(self, other), 
                                              engine = engine))
    def contains(self, points):
        '''
        Given a set of points, determine whether or not they are inside the mesh.
        This raises an error if called on a non- watertight mesh.

        Arguments
        ---------
        points: (n,3) set of points in space
        
        Returns
        ---------
        contains: (n) boolean array, whether or not a point is inside the mesh
        '''
        if not self.is_watertight: 
            log.warning('Mesh is non- watertight for contained point query!')
        contains = contains_points(self, points)
        return contains

    def copy(self):
        '''
        Get a copy of the current mesh.
        
        Returns
        ---------
        copied: current mesh deep copied.
        '''
        return deepcopy(self)

    def __hash__(self):
        # hash function requires an integer instead of a hex string
        hashed = int(self.md5(), 16)
        return hashed

    def __add__(self, other):
        '''
        Concatenate the mesh with another mesh
        '''
        result = util.concatenate(self, other)
        return result
