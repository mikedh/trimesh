'''
trimesh.py

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
from . import util
from . import convex

from .io.export    import export_mesh
from .ray.ray_mesh import RayMeshIntersector, contains_points
from .voxel        import Voxel
from .points       import unitize, transform_points
from .units        import _set_units
from .constants    import log, _log_time, tol

try: from .scene import Scene
except ImportError: log.warning('Mesh previewing unavailable!', exc_info=True)

try: 
    from .path.io.misc import faces_to_path
    from .path.io.load import _create_path, load_path
except ImportError:
    log.warning('trimesh.path unavailable!', exc_info=True)

class Trimesh(object):
    def __init__(self,
                 vertices       = None, 
                 faces          = None, 
                 face_normals   = None, 
                 vertex_normals = None,
                 metadata       = None,
                 process        = True,
                 **kwargs):
                 
        # cache computed values which are cleared when
        # self.md5() changes, forcing a recompute
        self._cache = util.Cache(id_function = self.md5)
        
        # (n, 3) float, set of vertices
        self.vertices = vertices
        # (m, 3) int of triangle faces, references self.vertices
        self.faces    = faces

        # hold vertex and face colors
        if 'visual' in kwargs:
            self.visual = kwargs['visual']
        else:
            self.visual = visual.VisualAttributes(**kwargs)
        self.visual.mesh = self
        
        # normals are accessed through setters/properties to 
        # ensure they are at least somewhat reasonable
        self.face_normals = face_normals

        # (n, 3) float of vertex normals.
        # can be created from face normals
        self.vertex_normals = vertex_normals

        # create a ray-mesh query object for the current mesh
        # initializing is very inexpensive and object is convenient to have.
        # On first query expensive bookkeeping is done (creation of r-tree),
        # and is cached for subsequent queries
        self.ray     = RayMeshIntersector(self)

        # any metadata that should be tracked per- mesh
        self.metadata = dict()
        # update the mesh metadata with passed metadata
        if isinstance(metadata, dict):
            self.metadata.update(metadata)

        if process:
            self.process()

    def process(self):
        '''
        Convenience function to remove garbage and make mesh sane.
        Does this by merging duplicate vertices, removing duplicate 
        and zero- area faces. 
        '''
        # avoid clearing the cache during operations
        with self._cache:
            self.merge_vertices()
            self.remove_duplicate_faces()
            self.remove_degenerate_faces()
        # since none of our process operations moved vertices or faces,
        # we can keep face and vertex normals in the cache without recomputing
        self._cache.clear(exclude = ['face_normals',
                                     'vertex_normals'])
        return self

    @property
    def faces(self):
        # we validate face normals as the validation process may remove
        # zero- area faces. If we didn't do this check here, the shape of 
        # self.faces and self.face_normals could differ depending on the order
        # they were queried in
        self._validate_face_normals()
        return self._faces
        
    @faces.setter
    def faces(self, values):
        if values is None: 
            values = []
        values = np.array(values, dtype=np.int64)
        if util.is_shape(values, (-1,4)):
            log.info('Triangulating quad faces')
            values = geometry.triangulate_quads(values)
        self._faces = util.tracked_array(values)

    @property
    def vertices(self):
        return self._vertices
        
    @vertices.setter
    def vertices(self, values):
        # make sure vertices are stored as a TrackedArray, which provides 
        # an md5() method which can be used to monitor the array for changes.
        # we also make sure vertices are stored as a float64 for consistency
        self._vertices = util.tracked_array(values, dtype=np.float64)

    def _validate_face_normals(self):
        cached = self._cache.get('face_normals')
        if np.shape(cached) != np.shape(self._faces):
            log.debug('Generating face normals as shape was incorrect')
            face_normals, valid = triangles.normals(self.triangles)
            self.update_faces(valid)
            self._cache.set(key = 'face_normals',
                            value = face_normals)

    @property
    def face_normals(self):
        self._validate_face_normals()
        cached = self._cache.get('face_normals')
        assert cached.shape == self._faces.shape
        return cached

    @face_normals.setter
    def face_normals(self, values):
        self._cache.set(key = 'face_normals',
                        value = np.asanyarray(values))

    @property
    def vertex_normals(self):
        cached = self._cache.get('vertex_normals')
        if np.shape(cached) == np.shape(self.vertices):
            return cached
        log.debug('Generating vertex normals')
        vertex_normals = geometry.mean_vertex_normals(len(self.vertices),
                                                      self.faces,
                                                      self.face_normals)
        return self._cache.set(key   = 'vertex_normals',
                               value =  vertex_normals)

    @vertex_normals.setter
    def vertex_normals(self, values):
        self._cache.set(key = 'vertex_normals',
                        value = np.asanyarray(values))

    def md5(self):
        '''
        Return an appended MD5 for the faces and vertices. 
        '''
        result  = self._faces.md5()
        result += self._vertices.md5()
        return result
        
    @property
    def bounds(self):
        '''
        (2,3) float, bounding box of the mesh of [min, max] coordinates
        '''
        bounds = np.vstack((np.min(self.vertices, axis=0),
                            np.max(self.vertices, axis=0)))
        return bounds
        
    @property
    def extents(self):
        return np.diff(self.bounds, axis=0)[0]
        
    @property
    def scale(self):
        return self.extents.max()

    @property                 
    def centroid(self):
        '''
        The (3) point in space which is the average vertex. 
        '''
        return np.mean(self.vertices, axis=0)
        
    @property
    def center_mass(self):
        '''
        The (3) point in space which is the center of mass/volume. 
        
        If the current mesh is not watertight, this is meaningless garbage. 
        '''
        center_mass = np.array(self.mass_properties(skip_inertia=True)['center_mass'])
        return center_mass
                               
    @property
    def volume(self):
        '''
        Volume of the current mesh.
        '''
        volume = self.mass_properties(skip_inertia=True)['volume']
        return volume

    @property
    def moment_inertia(self):
        '''
        Return the (3,3) moment of inertia of the current matrix.
        '''
        inertia = np.array(self.mass_properties(skip_inertia=False)['inertia'])
        return inertia

    @property
    def triangles(self):
        # use of advanced indexing on our tracked arrays will 
        # trigger a change (which nukes the cache)
        cached = self._cache.get('triangles')
        if cached is not None:
            return cached
        triangles = self.vertices.view(np.ndarray)[self._faces]
        return self._cache.set(key   = 'triangles',
                               value =  triangles)

    def triangles_tree(self):
        '''
        Return an R-tree containing each face of the mesh.
        '''
        tree = triangles.bounds_tree(self.triangles)
        return tree

    @property
    def edges(self):
        cached = self._cache.get('edges')
        if cached is not None:
            return cached
        edges = geometry.faces_to_edges(self.faces.view(np.ndarray))
        return self._cache.set(key   = 'edges',
                               value =  edges)

    @property
    def edges_unique(self):
        cached = self._cache.get('edges_unique')
        if cached is not None:
            return cached
        edges_sorted = np.sort(self.edges, axis=1)
        unique = grouping.unique_rows(edges_sorted)[0]
        edges_unique = edges_sorted[unique]
        return self._cache.set(key   = 'edges_unique',
                               value =  edges_unique)

    @property
    def euler_number(self):
        '''
        Return the Euler characteristic, a topological invariant, for the mesh
        '''
        euler = len(self.vertices) - len(self.edges_unique) + len(self.faces)
        return euler

    @property
    def units(self):
        if 'units' in self.metadata:
            return self.metadata['units']
        else:
            return None

    @units.setter
    def units(self, units):
        self.metadata['units'] = units
        
    def set_units(self, desired, guess=False):
        '''
        Convert the units of the mesh into a specified unit.
   
        Arguments
        ----------
        desired: string, units to convert to (eg 'inches')
        guess:   boolean, if self.units are not valid should we 
                 guess the current units of the document and then convert?
        '''
        _set_units(self, desired, guess)

    def merge_vertices(self, angle=None):
        '''
        If a mesh has vertices that are closer than TOL_MERGE, 
        redefine them to be the same vertex, and replace face references

        Arguments
        ---------
        angle_max: if defined, only vertices which are closer than TOL_MERGE
                   AND have vertex normals less than angle_max will be merged.
                   This is useful for smooth shading, but is much slower. 
        '''
        if angle is None:
            grouping.merge_vertices_hash(self)
        else:
            grouping.merge_vertices_kdtree(self, angle)

    def update_vertices(self, mask, inverse):
        '''
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
        self.faces = inverse[[self.faces.reshape(-1)]].reshape((-1,3))
        self.visual.update_vertices(mask)

        cached_normals = self._cache.get('vertex_normals')
        if cached_normals is not None:
            try: 
                self.vertex_normals = cached_normals[mask]
            except: 
                pass
        self._vertices = self.vertices[mask]

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
        mask = np.asanyarray(mask)

        if mask.dtype.name == 'bool':
            if mask.all(): 
                return
        elif mask.dtype.name != 'int':
            mask = mask.astype(np.int)

        cached_normals = self._cache.get('face_normals')
        if cached_normals is not None:
            self.face_normals = cached_normals[mask]
    
        self._faces = self._faces[mask]        
        self.visual.update_faces(mask)
        
    def remove_duplicate_faces(self):
        '''
        For the current mesh, remove any faces which are duplicated. 
        '''
        unique, inverse = grouping.unique_rows(np.sort(self.faces, axis=1))
        self.update_faces(unique)
        
    def rezero(self):
        '''
        Move the mesh so that all vertex vertices are positive.
        Does this by subtracting the min XYZ value from all vertices
        '''
        self.vertices -= self.vertices.min(axis=0)
        
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
        
    @property
    def face_adjacency(self):
        '''
        Returns an (n,2) list of face indices.
        Each pair of faces in the list shares an edge, making them adjacent.
        
        This is useful for lots of things, for example finding connected subgraphs:
        
        graph = nx.Graph()
        graph.add_edges_from(mesh.face_adjacency)
        groups = nx.connected_components(graph_connected.subgraph(interesting_faces))
        '''
        cached = self._cache.get('face_adjacency')
        if cached is None:
            adjacency, edges = graph.face_adjacency(self.faces.view(np.ndarray),
                                                    return_edges = True)
            self._cache.set(key   = 'face_adjacency_edges',
                            value = edges)
            self._cache.set(key   = 'face_adjacency', 
                            value = adjacency)
            return adjacency
        return cached

    @property
    def face_adjacency_edges(self):
        '''
        Returns the edges that are shared by the adjacent faces. 

        Returns
        --------
        edges: (n, 2) list of vertex indices which correspond to face_adjacency
        '''
        cached = self._cache.get('face_adjacency_edges')
        if cached is None:
            adjacency = self.face_adjacency
            return self._cache.get('face_adjacency_edges')
        return cached

    @property
    def is_watertight(self):
        '''
        Check if a mesh is watertight. 
        '''
        cached = self._cache.get('is_watertight')
        if cached is not None: return cached
        return self._cache.set(key   = 'is_watertight', 
                               value = graph.is_watertight(self.edges))
       
    @property
    def is_convex(self):
        '''
        Check if a mesh is convex. 
        '''
        cached = self._cache.get('is_convex')
        if cached is not None: return cached
        return self._cache.set(key   = 'is_convex', 
                               value = convex.is_convex(self))     
       
    def kdtree(self):
        '''
        Return a scipy.spatial.cKDTree of the vertices of the mesh.
        Not cached as this lead to memory issues and segfaults.
        '''
        from scipy.spatial import cKDTree as KDTree
        return KDTree(self.vertices.view(np.ndarray))

    def remove_degenerate_faces(self):
        '''
        Remove degenerate faces (faces with zero area) from the current mesh.
        '''
        nondegenerate = geometry.nondegenerate_faces(self.faces)
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
        if key in self._cache:
            return self._cache.get(key)

        facets = graph.facets(self)
        if return_area:
            area = np.array([self.area_faces[i].sum() for i in facets])
            result = (facets, area)
        else: 
            result = facets

        return self._cache.set(key   = key,
                               value = result)

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
        
        Returns:
        watertight: bool, is the mesh watertight after the function is done?
        '''
        return repair.fill_holes(self)

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
        # this should be recomputed if visuals change, so we 
        # store it in the visuals cache rather than the main mesh cache

        cached = self.visual._cache.get('smoothed')
        if cached is not None:
            return cached
        return self.visual._cache.set(key   = 'smoothed',
                                      value = graph.smoothed(self, angle))

    def section(self,
                plane_normal,
                plane_origin = None):
        '''
        Returns a cross section of the current mesh and plane defined by
        origin and normal.
        
        Arguments
        ---------
        plane_normal: (3) vector for plane normal
        plane_origin: (3) vector for plane origin. If None, will use [0,0,0]
                            
        Returns
        ---------
        intersections: Path3D of intersections             
        '''
        segments = intersections.mesh_plane_intersection(mesh         = self, 
                                                         plane_normal = plane_normal, 
                                                         plane_origin = plane_origin)
        path     = load_path(segments)
        return path

    @property
    def convex_hull(self):
        '''
        Get a new Trimesh object representing the convex hull of the 
        current mesh. Requires scipy >.12.

        Returns
        --------
        convex: Trimesh object of convex hull of current mesh
        '''

        cached = self._cache.get('convex_hull')
        if cached is not None: 
            return cached
        hull = convex.convex_hull(self)
        return self._cache.set(key   = 'convex_hull', 
                               value = hull)

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
        vertex indices and no faces are adjacent
        '''
        self.vertices = self.triangles.reshape((-1,3))
        self.faces = np.arange(len(self.vertices)).reshape((-1,3))
        
    def transform(self, matrix):
        '''
        Transform mesh vertices by matrix
        '''
        self.vertices = transform_points(self.vertices, matrix)

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
        path = _create_path(**faces_to_path(self, face_ids))
        return path
        
    @property
    def area(self):
        '''
        Summed area of all triangles in the current mesh.
        '''
        key    = 'area'
        cached = self._cache.get(key)
        if cached is not None: 
            return cached
        area = self.area_faces.sum()
        return self._cache.set(key   = key, 
                               value = area)

    @property                           
    def area_faces(self):
        '''
        The area of each face in the mesh
        '''
        key    = 'area_faces'
        cached = self._cache.get(key)
        if cached is not None: 
            return cached
        area_faces = triangles.area(self.triangles, sum = False)
        return self._cache.set(key   = key, 
                               value = area_faces) 
                         
    def mass_properties(self, density = 1.0, skip_inertia=False):
        '''
        Returns the mass properties of the current mesh.
        
        Assumes uniform density, and result is probably garbage if mesh
        isn't watertight. 

        Returns dictionary with keys: 
            'volume'      : in global units^3
            'mass'        : From specified density
            'density'     : Included again for convenience (same as kwarg density)
            'inertia'     : Taken at the center of mass and aligned with global 
                            coordinate system
            'center_mass' : Center of mass location, in global coordinate system
        '''
        key  = 'mass_properties_' 
        key += str(int(skip_inertia)) + '_' 
        key += str(int(density * 1e3))
        cached = self._cache.get(key)
        if cached is not None: 
            return cached
        mass = triangles.mass_properties(triangles    = self.triangles,
                                         density      = density,
                                         skip_inertia = skip_inertia)
        return self._cache.set(key   = key, 
                               value = mass)

    def scene(self):
        '''
        Return a Scene object containing the current mesh. 
        '''
        return Scene(self)

    def show(self, **kwargs):
        '''
        Render the mesh in an opengl window. Requires pyglet.
        Smooth will re-merge vertices to fix the shading, but can be slow
        on larger meshes. 
        '''
        scene = self.scene()
        scene.show(**kwargs)
        return scene

    def submesh(self, faces_sequence, **kwargs): 
        '''
        Return a subset of the mesh.
        
        Arguments
        ----------
        mesh: Trimesh object
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

    def identifier(self, length=6, as_json=False):
        '''
        Return a (length) float vector which is unique to the mesh,
        and is robust to rotation and translation.
        '''
        key = 'identifier' + str(length) + str(as_json)
        cached = self._cache.get(key)
        if cached is not None: return cached
        identifier = comparison.rotationally_invariant_identifier(self, 
                                                                  length, 
                                                                  as_json=as_json)
        return self._cache.set(key   = key,
                               value = identifier)

    def export(self, file_type='stl', file_obj=None):
        '''
        Export the current mesh to a file object. 
        If file_obj is a filename, file will be written there. 

        Supported formats are stl, off, and collada. 
        '''
        return export_mesh(mesh = self, 
                           file_obj  = file_obj, 
                           file_type = file_type)

    def to_dict(self):
        '''
        Return a dictionary representation of the current mesh, with keys 
        that can be used as the kwargs for the Trimesh constructor, eg:
        
        a = Trimesh(**other_mesh.to_dict())
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
        return deepcopy(self)

    def __eq__(self, other):
        equal = comparison.equal(self, other)
        return equal

    def __hash__(self):
        return self.identifier(as_json=True)

    def __add__(self, other):
        '''
        Meshes can be added to each other.
        Addition is defined as for the following meshes:
        a + b = c
        
        c is a mesh which has all the faces from a and b, and
        accompanying bookkeeping is done. 

        Defining this also allows groups of meshes to be summed easily, 
        for example like:

        a = np.sum(meshes).show()
        '''
        new_normals  = np.vstack((self.face_normals, other.face_normals))
        new_faces    = np.vstack((self.faces, (other.faces + len(self.vertices))))
        new_vertices = np.vstack((self.vertices, other.vertices))
        new_visual   = visual.visuals_union(self.visual, other.visual)
        result = Trimesh(vertices     = new_vertices, 
                         faces        = new_faces,
                         face_normals = new_normals,
                         visual       = new_visual)

        return result
