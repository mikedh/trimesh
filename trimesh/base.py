
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
from . import color
from . import sample
from . import repair
from . import comparison
from . import boolean
from . import intersections
from . import util

from .io.export    import export_mesh
from .ray.ray_mesh import RayMeshIntersector
from .voxel        import Voxel
from .points       import unitize, transform_points, contains_points
from .convex       import convex_hull
from .units        import _set_units
from .constants    import log, _log_time, tol

from scipy.spatial import cKDTree as KDTree

try: 
    from .path.io.misc import faces_to_path
    from .path.io.load import _create_path, load_path
except ImportError:
    log.warning('trimesh.path unavailable!', exc_info=True)

try:
    from .scene import Scene
except ImportError: 
    log.warning('Mesh previewing unavailable!', exc_info=True)

class Trimesh(object):
    def __init__(self, 
                 vertices        = None, 
                 faces           = None, 
                 face_normals    = None, 
                 vertex_normals  = None,
                 metadata        = None,
                 process         = False,
                 **kwargs):
                 
        # cache computed values which are cleared when
        # self.md5() changes, forcing a recompute
        self._cache = util.Cache(id_function = self.md5)
        
        # (n, 3) float, set of vertices
        self.vertices = vertices
        # (m, 3) int of triangle faces, references self.vertices
        self.faces    = faces

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

        # hold vertex and face colors
        self.visual = color.VisualAttributes(self)
        
        # any metadata that should be tracked per- mesh
        self.metadata = dict()
        # update the mesh metadata with passed metadata
        if isinstance(metadata, dict):
            self.metadata.update(metadata)

            
        self.merge_vertices()
            
        # if requested do basic mesh clean-up immediately
        if process:
            self.process()
            
    def process(self):
        '''
        Convenience function to do basic processing on a raw mesh
        '''
        self.remove_duplicate_faces()
        self.remove_degenerate_faces()
        return self
        
    @property
    def faces(self):
        return self._faces
        
    @faces.setter
    def faces(self, values):
        values = np.array(values)
        shape  = values.shape
        if len(shape) != 2 or not shape[1] in [3,4]:
            raise ValueError('Faces must be (n,3) or (n,4)!')
        elif shape[1] == 4:
            log.info('Triangulating quad faces')
            values = geometry.triangulate_quads(values)
        self._faces = util.tracked_array(values)

    @property
    def vertices(self):
        return self._vertices
        
    @vertices.setter
    def vertices(self, values):
        self._vertices = util.tracked_array(values)
        
    def md5(self):
        '''
        Return an appended MD5 for the faces and vertices. 
        '''
        result = self.faces.md5()
        result += self.vertices.md5()
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
        return self.mass_properties(skip_inertia=True)['center_mass']
        
    @property
    def box_size(self):
        return np.diff(self.bounds, axis=0)[0]
        
    @property
    def scale(self):
        return self.box_size.max()

    @property
    def body_count(self):
        '''
        Return the number of groups of connected faces.
        Bodies aren't necessarily watertight.
        '''
        return graph.split(self, only_count=True)

    @property
    def triangles(self):
        # use of advanced indexing on our tracked arrays will 
        # trigger a change (which nukes the cache)
        return self.vertices.view(np.ndarray)[self.faces]

    @property
    def edges(self):
        return geometry.faces_to_edges(self.faces.view(np.ndarray))

    @property
    def units(self):
        if 'units' in self.metadata:
            return self.metadata['units']
        else:
            return None

    @units.setter
    def units(self, units):
        self.metadata['units'] = units

    @property
    def face_normals(self):
        if np.shape(self._face_normals) != np.shape(self.faces):
            self._generate_face_normals()
            log.debug('Generating face normals')
        return self._face_normals

    @face_normals.setter
    def face_normals(self, values):
        self._face_normals = np.array(values)

    @property
    def vertex_normals(self):
        if np.shape(self._vertex_normals) != np.shape(self.vertices):
            self._generate_vertex_normals()
            log.debug('Generating vertex normals')
        return self._vertex_normals

    @vertex_normals.setter
    def vertex_normals(self, values):
        self._vertex_normals = np.array(values)

    def set_units(self, desired, guess=False):
        _set_units(self, desired, guess)

    def _generate_face_normals(self):
        face_normals, valid = triangles.normals(self.vertices[[self.faces]])
        self.update_faces(valid)
        self._face_normals = face_normals

    def _generate_vertex_normals(self):
        '''
        If face normals are defined, produce approximate vertex normals based on the
        average of the adjacent faces.
        
        If vertices are merged with no regard to normal angle, this is
        going to render with weird shading.
        '''
        self._vertex_normals = geometry.mean_vertex_normals(len(self.vertices),
                                                            self.faces,
                                                            self.face_normals)

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
        if mask.dtype.name == 'bool' and mask.all(): return
        self.faces = inverse[[self.faces.reshape(-1)]].reshape((-1,3))
        self.visual.update_vertices(mask)
        
        if np.shape(self._vertex_normals) == np.shape(self.vertices):
            self._vertex_normals = self._vertex_normals[mask]
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
        if mask.dtype.name == 'bool' and mask.all(): return
        if np.shape(self._face_normals) == np.shape(self.faces):
            self._face_normals = self._face_normals[mask]
        self.faces = self.faces[mask]        
        self.visual.update_faces(mask)
        
    def remove_duplicate_faces(self):
        '''
        For the current mesh, remove any faces which are duplicated. 
        This can occur if faces are below the 
        '''
        unique,inverse = grouping.unique_rows(np.sort(self.faces, axis=1))
        self.update_faces(unique)
        
    def rezero(self):
        '''
        Move the mesh so that all vertex vertices are positive.
        IE subtract the min vertex from all vertices, moving it to
        the first octant
        '''
        self.vertices -= self.vertices.min(axis=0)
        
    @_log_time
    def split(self, check_watertight=True):
        '''
        Returns a list of Trimesh objects, based on face connectivity.
        Splits into individual components, sometimes referred to as 'bodies'

        if check_watertight: only meshes which are watertight are returned
        '''
        meshes = graph.split(self, check_watertight=check_watertight)
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
            adjacency = graph.face_adjacency(self.faces.view(np.ndarray))
            return self._cache.set(key   = 'face_adjacency', 
                                   value = adjacency)
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
    def kdtree(self):
        '''
        Return a KDTree of the vertices of the mesh
        '''
        cached = self._cache.get('kdtree')
        if cached is not None: return cached
        kdtree = KDTree(self.vertices)
        return self._cache.set(key   = 'kdtree',
                               value = kdtree)
       
    def remove_degenerate_faces(self):
        '''
        Removes degenerate faces, or faces that have zero area.
        This function will only work if vertices are merged. 
        '''
        nondegenerate = geometry.nondegenerate_faces(self.faces)
        self.update_faces(nondegenerate)

    def facets(self, return_area=False, group_normals=False):
        '''
        Return a list of face indices for coplanar adjacent faces
        '''
        facets = [graph.facets, graph.facets_group][group_normals](self)
        if return_area:
            area = np.array([self.area_faces[i].sum() for i in facets])
            return facets, area
        return facets

    @_log_time    
    def fix_normals(self):
        '''
        Find and fix problems with self.face_normals and self.faces winding direction.
        
        For face normals ensure that vectors are consistently pointed outwards,
        and that self.faces is wound in the correct direction for all connected components.
        '''
        repair.fix_normals(self)

    def fill_holes(self, raise_watertight=True):
        '''
        Fill single triangle and single quad holes in the mesh.
        
        Arguments
        ---------
        raise_watertight: will raise an error if the current mesh cannot be
                          repaired to be watertight.
        '''
        repair.fill_holes(self, raise_watertight)

    def smooth(self, angle=.4):
        '''
        Process a mesh so that smooth shading renders nicely.
        This is done by merging vertices with a max angle critera. 
        '''
        self.unmerge_vertices()
        self.merge_vertices(angle)
    
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

    @_log_time   
    def convex_hull(self, clean=True):
        '''
        Get a new Trimesh object representing the convex hull of the 
        current mesh. Requires scipy >.12.

        Argments
        --------
        clean: boolean, if True will fix normals and winding
               to be coherent (as qhull/scipy outputs are not)

        Returns
        --------
        convex: Trimesh object of convex hull of current mesh
        '''
        result = convex_hull(self, clean)
        return result

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
        return sample.random_sample(self, count)
            
    def remove_unreferenced_vertices(self):
        '''
        For the current mesh, remove all vertices which are not referenced by
        a face. 
        '''
        unique, inverse = np.unique(self.faces.reshape(-1), return_inverse=True)
        self.faces      = inverse.reshape((-1,3))          
        self.vertices   = self.vertices[unique]

    def unmerge_vertices(self):
        '''
        Removes all face references, so that every face contains
        three unique vertex indices.
        '''
        self.vertices = self.vertices[[self.faces]].reshape((-1,3))
        self.faces    = np.arange(len(self.vertices)).reshape((-1,3))
        
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

    def export(self, file_obj=None, file_type='stl'):
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

    def union(self, other):
        '''
        Boolean union between this mesh and n other meshes

        Arguments
        ---------
        other: Trimesh, or list of Trimesh objects

        Returns
        ---------
        union: Trimesh, union of self and other Trimesh objects
        '''
        return Trimesh(process=True, **boolean.union(self, other))

    def difference(self, other):        
        '''
        Boolean difference between this mesh and n other meshes

        Arguments
        ---------
        other: Trimesh, or list of Trimesh objects

        Returns
        ---------
        difference: Trimesh, difference between self and other Trimesh objects
        '''
        return Trimesh(process=True, **boolean.difference(self, other))
        
    def intersection(self, other):
        '''
        Boolean intersection between this mesh and n other meshes

        Arguments
        ---------
        other: Trimesh, or list of Trimesh objects

        Returns
        ---------
        intersection: Trimesh of the volume contained by all passed meshes
        '''
        return Trimesh(process=True, **boolean.intersection(self, other))
    
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
            raise ValueError('Non-watertight meshes can\'t contain anything!')
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
        new_faces    = np.vstack((self.faces, (other.faces + len(self.vertices))))
        new_vertices = np.vstack((self.vertices, other.vertices))
        new_normals  = np.vstack((self.face_normals, other.face_normals))

        new_colors = np.vstack((self.visual.face_colors, 
                                other.visual.face_colors))

        result = Trimesh(vertices     = new_vertices, 
                         faces        = new_faces,
                         face_normals = new_normals)
        result.visual.face_colors = new_colors

        return result
