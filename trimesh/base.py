
'''
trimesh.py

Library for importing and doing simple operations on triangular meshes.
'''

import numpy as np

from . import triangles
from . import grouping
from . import geometry
from . import graph
from . import color
from . import sample
from . import repair
from . import comparison
from . import intersections
from . import units

from .io.export    import export_mesh
from .ray.ray_mesh import RayMeshIntersector
from .points       import unitize, transform_points
from .convex       import convex_hull
from .constants    import *

try: 
    from .path.io.misc import faces_to_path
    from .path.io.load import _create_path, load_path
except ImportError:
    log.warning('trimesh.path unavailable!', exc_info=True)

class Trimesh(object):
    def __init__(self, 
                 vertices        = None, 
                 faces           = None, 
                 face_normals    = None, 
                 vertex_normals  = None,
                 metadata        = None,
                 process         = False,
                 **kwargs):

        # (n, 3) float, set of vertices
        self.vertices        = np.array(vertices)
        # (m, 3) int of triangle faces, references self.vertices
        self.faces           = np.array(faces)

        # normals are accessed through setters/properties to 
        # ensure they are at least somewhat reasonable
        self._face_normals    = np.array(face_normals)
        # (n, 3) float of vertex normals.
        # can be created from face normals
        self._vertex_normals  = np.array(vertex_normals)

        # any metadata that should be tracked per- mesh
        self.metadata        = dict()

        # create a ray- mesh intersector for the current mesh
        # initializing is very inexpensive and object is convienent to have
        # on first query expensive bookkeeping is done (creation of r-tree)
        # and is cached for subsequent queries
        self.ray     = RayMeshIntersector(self)

        # hold vertex and face colors, as well as textures someday
        self.visual = color.VisualAttributes(self)

        # update the mesh metadata with passed metadata
        if isinstance(metadata, dict):
            self.metadata.update(metadata)
        # if requested, do basic mesh cleanup
        if process:
            self.process()
            
    def process(self):
        '''
        Convenience function to do basic processing on a raw mesh
        '''
        self.merge_vertices()
        self.remove_duplicate_faces()
        self.remove_degenerate_faces()
        return self

    @property
    def bounds(self):
        '''
        (2,3) float, bounding box of the mesh of [min, max] coordinates
        '''
        return np.vstack((np.min(self.vertices, axis=0),
                          np.max(self.vertices, axis=0)))

    @property                 
    def centroid(self):
        return np.mean(self.vertices, axis=0)
        
    @property
    def center_mass(self):
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
        return self.vertices[self.faces]

    @property
    def edges(self):
        return geometry.faces_to_edges(self.faces)

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
        if np.shape(values) != np.shape(self.faces):
            log.warning('Faces are %s, passed normals are %s', 
                      np.shape(self.faces),
                      np.shape(values))
        self._face_normals = np.array(values)

    @property
    def vertex_normals(self):
        if np.shape(self._vertex_normals) != np.shape(self.vertices):
            self._generate_vertex_normals()
            log.debug('Generating vertex normals')
        return self._vertex_normals

    @vertex_normals.setter
    def vertex_normals(self, values):
        if np.shape(values) != np.shape(self.vertices):
            log.warning('Vertex normals are incorrect shape!')
        self._vertex_normals = np.array(values)

    def set_units(self, desired, guess=False):
        if self.units is None:
            if guess:
                log.warn('Current document doesn\'t have units specified, guessing!')
                self.units = units.unit_guess(self.scale)
            else: 
                raise ValueError('No units specified, and not allowed to guess!')
        log.info('Converting units from %s to %s', self.units, desired)
        conversion = units.unit_conversion(self.units, desired)
        self.vertices         *= conversion
        self.metadata['units'] = desired

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
        vertex_normals = np.zeros((len(self.vertices), 3,3))
        vertex_normals[[self.faces[:,0],0]] = self.face_normals
        vertex_normals[[self.faces[:,1],1]] = self.face_normals
        vertex_normals[[self.faces[:,2],2]] = self.face_normals
        mean_normals        = vertex_normals.mean(axis=1)
        unit_normals, valid = geometry.unitize(mean_normals, check_valid=True)

        mean_normals[valid] = unit_normals
        # if the mean normal is zero, it generally means that you've encountered 
        # the edge case where
        # a) the vertex is only shared by 2 faces (mesh is not watertight)
        # b) the two faces that share the vertex have normals pointed exactly
        #    opposite each other. 
        # since this means the vertex normal isn't defined, just make it anything
        mean_normals[np.logical_not(valid)] = [1,0,0]
        self._vertex_normals = mean_normals
        
    def merge_vertices(self, angle_max=None):
        '''
        If a mesh has vertices that are closer than TOL_MERGE, 
        redefine them to be the same vertex, and replace face references

        Arguments
        ---------
        angle_max: if defined, only vertices which are closer than TOL_MERGE
                   AND have vertex normals less than angle_max will be merged.
                   This is useful for smooth shading, but is much slower. 
        '''
        if angle_max is None:
            grouping.merge_vertices_hash(self)
        else:
            grouping.merge_vertices_kdtree(self, angle_max)

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
        
        if self._vertex_normals.shape == self.vertices.shape:
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
        if self._face_normals.shape == self.faces.shape:
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
        
    @log_time
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
        return graph.face_adjacency(self.faces)

    @property
    def is_watertight(self):
        '''
        Check if a mesh is watertight. 
        This currently only checks to see if every face has three adjacent faces
        '''
        return graph.is_watertight(self)

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
            area = [triangles.area(self.vertices[self.faces[i]]) for i in facets]
            return facets, area
        return facets

    @log_time    
    def fix_normals(self):
        '''
        Find and fix problems with self.face_normals and self.faces winding direction.
        
        For face normals ensure that vectors are consistently pointed outwards,
        and that self.faces is wound in the correct direction for all connected components.
        '''
        graph.fix_normals(self)

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

    @log_time   
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

    def outline(self, face_ids=None):
        '''
        Given a set of face ids, find the outline of the faces,
        and return it as a Path3D. Note that this implies a non-
        watertight section, and the 'outline' of a watertight
        mesh or subset of a mesh is an empty path. 

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
        
    def area(self, sum=True):
        '''
        Summed area of all triangles in the current mesh.
        '''
        return triangles.area(self.vertices[self.faces], sum=sum)
        
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
        return triangles.mass_properties(triangles    = self.vertices[[self.faces]], 
                                         density      = density,
                                         skip_inertia = skip_inertia)

    def scene(self):
        '''
        Return a Scene object containing the current mesh. 
        '''
        from .scene import Scene
        return Scene(self)

    def show(self, block=True):
        '''
        Render the mesh in an opengl window. Requires pyglet.
        Smooth will re-merge vertices to fix the shading, but can be slow
        on larger meshes. 
        '''
        scene = self.scene()
        scene.show(block = block)
        return scene

    def identifier(self, length=6):
        '''
        Return a (length) float vector which is unique to the mesh,
        and is robust to rotation and translation.
        '''
        return comparison.rotationally_invariant_identifier(self, length)

    def export(self, file_obj=None, file_type='stl'):
        '''
        Export the current mesh to a file object. 
        If file_obj is a filename, file will be written there. 

        Supported formats are stl, off, and collada. 
        '''
        return export_mesh(self, file_obj, file_type)

    def __add__(self, other):
        '''
        Meshes can be added to each other.
        Addition is defined as for the following meshes:
        a + b = c
        
        c is a mesh which has all the faces from a and b, and
        acompanying bookkeeping is done. 

        Defining this also allows groups of meshes to be summed easily, 
        for example like:

        a = np.sum(meshes).show()
        '''
        new_faces    = np.vstack((self.faces, (other.faces + len(self.vertices))))
        new_vertices = np.vstack((self.vertices, other.vertices))
        new_normals  = np.vstack((self.face_normals, other.face_normals))

        new_colors   = np.vstack((self.visual.face_colors, 
                                  other.visual.face_colors))

        result =  Trimesh(vertices     = new_vertices, 
                          faces        = new_faces,
                          face_normals = new_normals)
        result.visual.face_colors = new_colors

        return result
