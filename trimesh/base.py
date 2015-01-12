'''
trimesh.py

Library for importing and doing simple operations on triangular meshes.
'''

import numpy as np

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

from . import triangles
from . import grouping
from . import geometry
from . import graph_ops
from . import color

from .constants import *
from .geometry import unitize, transform_points


def log_time(method):
    def timed(*args, **kwargs):
        tic    = time_function()
        result = method(*args, **kwargs)
        log.debug('%s executed in %.4f seconds.',
                  method.__name__,
                  time_function()-tic)
        return result
    return timed

class Trimesh():
    def __init__(self, 
                 vertices        = None, 
                 faces           = None, 
                 face_normals    = None, 
                 vertex_normals  = None,
                 face_colors     = None,
                 vertex_colors   = None,
                 metadata        = None,
                 process_on_load = True):

        
        self.vertices        = np.array(vertices)
        self.faces           = np.array(faces)
        self.face_normals    = np.array(face_normals)
        self.vertex_normals  = np.array(vertex_normals)
        self.face_colors     = np.array(face_colors)
        self.vertex_colors   = np.array(vertex_colors)
        self.metadata        = dict()
        
        if isinstance(metadata, dict): self.metadate.update(metadata)
        if process_on_load:            self.process()
            
    def process(self):
        '''
        Convenience function to do basic processing on a raw mesh
        '''
        self.merge_vertices()
        self.remove_duplicate_faces()
        self.remove_degenerate_faces()
        self.verify_face_normals()
        return self
        
    def rezero(self):
        self.vertices -= self.vertices.min(axis=0)
        
    @log_time
    def split(self, check_watertight=True):
        '''
        Returns a list of Trimesh objects, based on face connectivity.
        Splits into individual components, sometimes referred to as 'bodies'

        if check_watertight: only meshes which are watertight are returned
        '''
        meshes = graph_ops.split(self, check_watertight)
        log.info('split found %i components', len(meshes))
        return meshes

    @property
    def body_count(self):
        return graph_ops.split(self, only_count=True)

    def face_adjacency(self):
        '''
        Returns an (n,2) list of face indices.
        Each pair of faces in the list shares an edge, making them adjacent.
        
        This is useful for lots of things, for example finding connected subgraphs:
        
        graph = nx.Graph()
        graph.add_edges_from(mesh.face_adjacency())
        groups = nx.connected_components(graph_connected.subgraph(interesting_faces))
        '''
        
        # first generate the list of edges for the current faces
        edges = geometry.faces_to_edges(self.faces, sort=True)
        # this will return the indices for duplicate edges
        # every edge appears twice in a well constructed mesh
        # so for every row in edge_idx, edges[edge_idx[*][0]] == edges[edge_idx[*][1]]
        # in this call to group rows, we discard edges which don't occur twice
        edge_idx = grouping.group_rows(edges, require_count=2)

        if len(edge_idx) == 0:
            log.error('No adjacent faces detected! Did you merge vertices?')
        # returns the pairs of all adjacent faces
        # so for every row in face_idx, self.faces[face_idx[*][0]] and self.faces[face_idx[*][1]]
        # will share an edge
        face_idx = np.tile(np.arange(len(self.faces)), (3,1)).T.reshape(-1)[[edge_idx]]
        return face_idx

    def is_watertight(self):
        '''
        Check if a mesh is watertight. 
        This currently only checks to see if every face has three adjacent faces
        '''
        return graph_ops.is_watertight(self)

    def remove_degenerate_faces(self):
        '''
        Removes degenerate faces, or faces that have zero area.
        This function will only work if vertices are merged. 
        '''
        nondegenerate = geometry.nondegenerate_faces(self.faces)
        self.faces    = self.faces[nondegenerate]

        if np.shape(self.face_normals) == np.shape(self.faces):
            self.face_normals = self.face_normals[nondegenerate]

        log.debug('%i/%i faces were degenerate and have been removed',
                  np.sum(np.logical_not(nondegenerate)),
                  len(nondegenerate))

    def facets(self, return_area=True):
        '''
        Return a list of face indices for coplanar adjacent faces
        '''
        facet_list = graph_ops.facets(self)
        if return_area:
            facets_area = [triangles.area(self.vertices[[self.faces[i]]]) for i in facet_list]
            return facet_list, facets_area
        return facet_list

    @log_time    
    def fix_normals(self):
        '''
        Find and fix problems with self.face_normals and self.faces winding direction.
        
        For face normals ensure that vectors are consistently pointed outwards,
        and that self.faces is wound in the correct direction for all connected components.
        '''
        graph_ops.fix_normals(self)

    def verify_normals(self):
        '''
        Check to make sure both face and vertex normals are defined. 
        '''
        self.verify_face_normals()
        self.verify_vertex_normals()

    def verify_vertex_normals(self):
        if np.shape(self.vertex_normals) != np.shape(self.vertices): 
            self.generate_vertex_normals()
            
    def verify_face_normals(self):
        '''
        Check to make sure face normals are defined. 
        '''
        if np.shape(self.face_normals) != np.shape(self.faces):
            log.debug('Generating face normals as shape check failed')
            self.generate_face_normals()
        else:
            self.face_normals, valid = geometry.unitize(self.face_normals, check_valid=True)
            if not np.all(valid):  self.generate_face_normals()

    def cross_section(self,
                      normal,
                      origin        = None,
                      return_planar = True):
        '''
        Returns a cross section of the current mesh and plane defined by
        origin and normal.
        
        Arguments
        ---------

        normal:        (3) vector for plane normal
        origin:        (3) vector for plane origin. If None, will use [0,0,0]
        return_planar: boolean, whether to project cross section to plane or not
                        If return_planar is True,  returned shape is (n, 2, 2) 
                        If return_planar is False, returned shape is (n, 2, 3)
                        
        Returns
        ---------
        intersections: (n, 2, [2|3]) line segments where plane intersects triangles in mesh
                       
        '''
        
        return geometry.cross_section(mesh          = self, 
                                      plane_normal  = normal, 
                                      plane_origin  = origin,
                                      return_planar = return_planar)
    
    @log_time   
    def convex_hull(self):
        '''
        Get a new Trimesh object representing the convex hull of the 
        current mesh. Requires scipy >.12.
  
        '''
        from scipy.spatial import ConvexHull
        faces  = ConvexHull(self.vertices).simplices
        convex = Trimesh(vertices=self.vertices.copy(), faces=faces)
        # the normals and triangle winding returned by scipy/qhull's
        # ConvexHull are apparently random, so we need to completely fix them
        convex.fix_normals()
        # since we just copied all the vertices over, we will have a bunch
        # of unreferenced vertices, so it is best to remove them
        convex.remove_unreferenced_vertices()
        return convex

    def merge_vertices(self, angle_max=None):
        if not (angle_max is None): grouping.merge_vertices_kdtree(self)
        else:                       grouping.merge_vertices_hash(self)

    @log_time
    def remove_duplicate_faces(self):
        unique = grouping.unique_rows(np.sort(self.faces, axis=1), digits=0)
        log.debug('%i/%i faces were duplicate and have been removed',
                  len(self.faces) - len(unique),
                  len(self.faces))
        self.faces        = self.faces[[unique]]
        if np.shape(self.face_normals) == np.shape(self.faces):
            self.face_normals = self.face_normals[[unique]]


    def remove_unreferenced_vertices(self):
        unique, inverse = np.unique(self.faces.reshape(-1), return_inverse=True)
        self.faces      = inverse.reshape((-1,3))          
        self.vertices   = self.vertices[unique]

    @log_time
    def unmerge_vertices(self):
        '''
        Removes all face references, so that every face contains
        three unique vertex indices.
        '''
        self.vertices = self.vertices[[self.faces]].reshape((-1,3))
        self.faces    = np.arange(len(self.vertices)).reshape((-1,3))
        
    @log_time    
    def generate_face_normals(self):
        '''
        If no normal information is loaded, we can get it from cross products
        Normal direction will be incorrect if mesh faces aren't ordered (right-hand rule)
        '''
        self.face_normals = triangles.normals(self.vertices[[self.faces]])
        
    @log_time    
    def generate_vertex_normals(self):
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
        self.vertex_normals = geometry.unitize(np.mean(vertex_normals, axis=1))
        
    def verify_face_colors(self):
        if np.shape(self.face_colors) != np.shape(self.faces):
            self.generate_face_colors()
    
    def generate_face_colors(self, assign_color=None):
        '''
        Apply face colors. If none are defined, set to default color
        '''
        if assign_color is None: assign_color = color.DEFAULT_COLOR
        self.face_colors = np.tile(assign_color, (len(self.faces), 1))

    def generate_vertex_colors(self):
        '''
        Populate self.vertex_colors
        If self.face_colors are defined, we use those values to generate
        vertex colors. If not, we just set them to the DEFAULT_COLOR
        '''
        if np.shape(self.vertex_colors) == (len(self.vertices), 3): 
            return
        elif np.shape(self.face_colors) == (len(self.faces), 3):
            # case where face_colors is populated, but vertex_colors isn't
            # we then generate vertex colors from the face colors
            vertex_colors = np.zeros((len(self.vertices), 3,3))
            vertex_colors[[self.faces[:,0],0]] = self.face_colors
            vertex_colors[[self.faces[:,1],1]] = self.face_colors
            vertex_colors[[self.faces[:,2],2]] = self.face_colors
            vertex_colors  = geometry.unitize(np.mean(vertex_colors, axis=1))
            vertex_colors *= (255.0 / np.max(vertex_colors, axis=1).reshape((-1,1)))
            self.vertex_colors = vertex_colors.astype(int)
        else:
            self.vertex_colors = np.tile(color.DEFAULT_COLOR, (len(self.vertices), 1))
        
    def transform(self, matrix):
        '''
        Transform mesh vertices by matrix
        '''
        self.vertices = transform_points(self.vertices, matrix)

    @property
    def area(self):
        '''
        Summed area of all triangles in the current mesh.
        '''
        return triangles.area(self.vertices[self.faces])
        
    @property
    def bounds(self):
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
        return np.min(self.box_size)
        
    @log_time  
    def mass_properties(self, density = 1.0, skip_inertia=False):
        '''
        Returns the mass properties of the current mesh.
        
        Assumes uniform density, and result is probably garbage if mesh
        isn't watertight. 

        Returns dictionary with keys: 
            'volume'      : in global units^3
            'mass'        : From specified density
            'density'     : Included again for convenience (same as kwarg density)
            'inertia'     : Taken at the center of mass and aligned with global coordinate system
            'center_mass' : Center of mass location, in global coordinate system
        '''
        return triangles.mass_properties(triangles    = self.vertices[[self.faces]], 
                                         density      = density,
                                         skip_inertia = skip_inertia)

    def show(self, smooth = None, smooth_angle = np.radians(20)):
        '''
        Render the mesh in an opengl window. Requires pyglet.
        Smooth will re-merge vertices to fix the shading, but can be slow
        on larger meshes. 
        '''
        from .mesh_render import MeshRender
        MeshRender(self, smooth=smooth, smooth_angle=smooth_angle)

    def boolean(self, other_mesh, operation='union'):
        '''
        Use cork with python bindings (https://github.com/stevedh/cork)
        for boolean operations.

        Arguments
        -----------
        other_mesh: Trimesh object of other mesh
        operation:  Which boolean operation to do. Cork supports:
                    'union'
                    'diff'
                    'isct' (intersection)
                    'xor'
                    'resolve'

        Returns
        -----------
        Trimesh object of result 
        '''
        def astuple(mesh):
            return (mesh.faces.astype(np.uint32), mesh.vertices.astype(np.float32))
            
        import cork
        operations = {'union' : cork.computeUnion,
                      'isct'  : cork.computeIntersection,
                      'diff'  : cork.computeDifference}
        
        result      = operations[operation](astuple(self), astuple(other_mesh))
        result_mesh = Trimesh(vertices = result[1], faces=result[0])
        result_mesh.verify_normals()
        return result_mesh

    def __add__(self, other):
        new_faces    = np.vstack((self.faces, (other.faces + len(self.vertices))))
        new_vertices = np.vstack((self.vertices, other.vertices))

        self.verify_face_colors()
        other.verify_face_colors()
        new_colors   = np.vstack((self.face_colors, other.face_colors))

        return Trimesh(vertices    = new_vertices, 
                       faces       = new_faces, 
                       face_colors = new_colors)

    def export(self, filename):
        from .mesh_io import export_stl
        export_stl(self, filename)
        
if __name__ == '__main__':
    formatter = logging.Formatter("[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s", 
                                  "%Y-%m-%d %H:%M:%S")
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    handler_stream.setLevel(logging.DEBUG)
    log.setLevel(logging.DEBUG)
    log.addHandler(handler_stream)
    np.set_printoptions(precision=6, suppress=True)

    m = load_mesh('models/featuretype.stl')
    

