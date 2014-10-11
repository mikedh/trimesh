'''
trimesh.py

Library for importing and doing simple operations on triangular meshes.
'''

import numpy as np
import struct
import os
from collections import deque
import sys


from time import time as time_function
from string import Template
import networkx as nx

try: 
    from graph_tool import Graph
    from graph_tool.topology import label_components
    has_gt = True
except: 
    has_gt = False

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

PY3 = sys.version_info.major >= 3
if not PY3: 
    range = xrange

TOL_ZERO  = 1e-12
TOL_MERGE = 1e-9

PASTELS = {'red'    : [194,59,34],
           'purple' : [150,111,214],
           'blue'   : [119,158,203],
           'brown'  : [160,85,45]}

DEFAULT_COLOR  = PASTELS['blue']

def available_formats():
    return _MESH_LOADERS.keys()

def log_time(method):
    def timed(*args, **kwargs):
        tic    = time_function()
        result = method(*args, **kwargs)
        log.debug('%s executed in %.4f seconds.',
                  method.__name__,
                  time_function()-tic)
        return result
    return timed

@log_time
def load_mesh(file_obj, file_type=None):
    '''
    Load a mesh file into a Trimesh object

    file_obj: a filename string, or a file object
    '''

        
    if not hasattr(file_obj, 'read'):
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj  = open(file_obj, 'rb')

    mesh = _MESH_LOADERS[file_type](file_obj, file_type)
    file_obj.close()
    log.info('loaded mesh using function %s, containing %i faces', 
             _MESH_LOADERS[file_type].__name__, 
             len(mesh.faces))
    return mesh

def load_assimp(file_obj, file_type=None):
    '''
    Use the assimp library to load a mesh, from a file object and type,
    or filename (if file_obj is a string)

    Assimp supports a huge number of mesh formats.

    Performance notes: in tests on binary STL pyassimp was ~10x 
    slower than the native loader included in this package. 
    This is probably due to their recursive prettifying of the data structure.
    
    Also, you need a very recent version of PyAssimp for this function to work 
    (the commit was merged into the assimp github master on roughly 9/5/2014)
    '''

    def LPMesh_to_Trimesh(lp):
        return Trimesh(vertices       = lp.vertices,
                       vertex_normals = lp.normals,
                       faces          = lp.faces)

    if not hasattr(file_obj, 'read'):
        # if there is no read attribute, we assume we've been passed a file name
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj  = open(file_obj, 'rb')

    scene  = pyassimp.load(file_obj, file_type=file_type)
    meshes = list(map(LPMesh_to_Trimesh, scene.meshes))
    pyassimp.release(scene)

    if len(meshes) == 1: 
        return meshes[0]
    return meshes

class Trimesh():
    def __init__(self, 
                 vertices        = None, 
                 faces           = None, 
                 face_normals    = None, 
                 vertex_normals  = None,
                 face_colors     = None,
                 vertex_colors   = None,
                 metadata        = None):

        self.vertices        = np.array(vertices)
        self.faces           = np.array(faces)
        self.face_normals    = np.array(face_normals)
        self.vertex_normals  = np.array(vertex_normals)
        self.face_colors     = np.array(face_colors)
        self.vertex_colors   = np.array(vertex_colors)
        self.metadata        = dict()
        
        if isinstance(metadata, dict):
            self.metadate.update(metadata)
            
    def process(self):
        '''
        Convenience function to do basic processing on a raw mesh
        '''
        self.merge_vertices_hash()
        self.verify_face_normals()
        self.remove_duplicate_faces()
        self.remove_degenerate_faces()
        return self

    @log_time
    def split(self):
        '''
        Returns a list of Trimesh objects, based on face connectivity.
        Splits into individual components, sometimes referred to as 'bodies'
        '''
        if has_gt: return split_gt(self)
        return split_by_face_connectivity(self)

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
        edges = faces_to_edges(self.faces, sort=True)
        # this will return the indices for duplicate edges
        # every edge appears twice in a well constructed mesh
        # so for every row in edge_idx, edges[edge_idx[*][0]] == edges[edge_idx[*][1]]
        # in this call to group rows, we discard edges which don't occur twice
        edge_idx = group_rows(edges, require_count=2)

        if len(edge_idx) == 0:
            log.error('No adjacent faces detected! Did you merge vertices?')
        # returns the pairs of all adjacent faces
        # so for every row in face_idx, self.faces[face_idx[*][0]] and self.faces[face_idx[*][1]]
        # will share an edge
        face_idx = np.tile(np.arange(len(self.faces)), (3,1)).T.reshape(-1)[[edge_idx]]
        return face_idx
        
    @log_time
    def is_watertight(self):
        '''
        Check if a mesh is watertight. 
        This currently only checks to see if every face has three adjacent faces
        '''
        adjacency  = nx.from_edgelist(self.face_adjacency())
        watertight = np.equal(list(adjacency.degree().values()), 3).all()
        return watertight

    @log_time
    def remove_degenerate_faces(self):
        '''
        Removes degenerate faces, or faces that have zero area.
        This function will only work if vertices are merged. 
        '''
        nondegenerate = nondegenerate_faces(self.faces)
        self.faces    = self.faces[nondegenerate]
        log.debug('%i/%i faces were degenerate and have been removed',
                  np.sum(np.logical_not(nondegenerate)),
                  len(nondegenerate))

    def facets(self):
        '''
        Return a list of face indices for coplanar adjacent faces
        '''
        return facets(self)

    @log_time    
    def fix_normals(self):
        '''
        Find and fix problems with self.face_normals and self.faces winding direction.
        
        For face normals ensure that vectors are consistently pointed outwards,
        and that self.faces is wound in the correct direction for all connected components.
        '''
        self.generate_face_normals()
        # we create the face adjacency graph: 
        # every node in g is an index of mesh.faces
        # every edge in g represents two faces which are connected
        graph = nx.from_edgelist(self.face_adjacency())
        
        # we are going to traverse the graph using BFS, so we have to start
        # a traversal for every connected component
        for connected in nx.connected_components(graph):
            # we traverse every pair of faces in the graph
            # we modify self.faces and self.face_normals in place 
            for face_pair in nx.bfs_edges(graph, connected[0]):
                # for each pair of faces, we convert them into edges,
                # find the edge that both faces share, and then see if the edges
                # are reversed in order as you would expect in a well constructed mesh
                pair      = self.faces[[face_pair]]
                edges     = faces_to_edges(pair, sort=False)
                overlap   = group_rows(np.sort(edges,axis=1), require_count=2)
                edge_pair = edges[[overlap[0]]]
                reversed  = edge_pair[0][0] != edge_pair[1][0]
                if reversed: continue
                # if the edges aren't reversed, invert the order of one of the faces
                # and negate its normal vector
                self.faces[face_pair[1]] = self.faces[face_pair[1]][::-1]
                self.face_normals[face_pair[1]] *= (reversed*2) - 1
                self.vertices[[self.faces[face_pair[1]]]]
            # the normals of every connected face now all pointed in 
            # the same direction, but there is no guarantee that they aren't all
            # pointed in the wrong direction
            # NOTE this check appears to have an issue and normals will sometimes
            # be incorrectly directed. 
            faces     = self.faces[[connected]]
            leftmost  = np.argmin(np.min(self.vertices[:,0][[faces]], axis=1))
            backwards = np.dot([-1.0,0,0], self.face_normals[leftmost]) > 0.0
            if backwards: self.face_normals[[connected]] *= -1.0
            
            winding_tri  = connected[0]
            winding_test = np.diff(self.vertices[[self.faces[winding_tri]]], axis=0)
            winding_dir  = np.dot(unitize(np.cross(*winding_test)), self.face_normals[winding_tri])
            if winding_dir < 0: self.faces[[connected]] = np.fliplr(self.faces[[connected]])
            
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
        if (np.shape(self.face_normals) != np.shape(self.faces)):
            log.debug('Generating face normals for faces %s and passed face normals %s',
                      str(np.shape(self.faces)),
                      str(np.shape(self.face_normals)))
            self.generate_face_normals()

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
        if np.shape(origin) != (3):
            origin = self.centroid
        intersections = cross_section(mesh         = self, 
                                      plane_normal = normal, 
                                      plane_origin = origin)
        if not return_planar: return intersections                     
        return  project_to_plane(intersections.reshape((-1,3)), 
                                 normal = normal, 
                                 origin = origin).reshape((-1,2,2))

    @log_time   
    def convex_hull(self):
        '''
        Get a new Trimesh object representing the convex hull of the 
        current mesh. Requires scipy >.12.
  
        '''
        from scipy.spatial import ConvexHull
        faces = ConvexHull(self.vertices).simplices
        mesh  = Trimesh(vertices = self.vertices, faces = faces)
        # the normals and triangle winding returned by scipy/qhull's
        # ConvexHull are apparently random, so we need to completely fix them
        mesh.fix_normals()
        return mesh

    def merge_vertices(self, angle_max=None):
        if angle_max != None: self.merge_vertices_kdtree(angle_max)
        else:                 self.merge_vertices_hash()

    @log_time
    def merge_vertices_kdtree(self, angle_max=None):
        '''
        Merges vertices which are identical, AKA within 
        Cartesian distance TOL_MERGE of each other.  
        Then replaces references in self.faces
        
        If angle_max == None, vertex normals won't be looked at. 
        if angle_max has a value, vertices will only be considered identical
        if they are within TOL_MERGE of each other, and the angle between
        their normals is less than angle_max

        Performance note:
        cKDTree requires scipy >= .12 for this query type and you 
        probably don't want to use plain python KDTree as it is crazy slow (~1000x in tests)
        '''
        from scipy.spatial import cKDTree as KDTree
        
        tic         = time_function()
        tree        = KDTree(self.vertices)
        used        = np.zeros(len(self.vertices), dtype=np.bool)
        unique      = deque()
        replacement = dict()
        
        if angle_max != None: self.verify_normals()

        for index, vertex in enumerate(self.vertices):
            if used[index]: continue
            neighbors = np.array(tree.query_ball_point(self.vertices[index], TOL_MERGE))
            used[[neighbors]] = True
            if angle_max != None:
                normals, aligned = group_vectors(self.vertex_normals[[neighbors]], TOL_ANGLE = angle_max)
                for group in aligned:
                    vertex_indices = neighbors[[group]]
                    replacement.update(np.column_stack((vertex_indices, [len(unique)] * len(group))))
                    unique.append(vertex_indices[0])
            else:
                replacement.update(np.column_stack((neighbors, [len(unique)]*len(neighbors))))
                unique.append(neighbors[0])

        self.vertices = self.vertices[[unique]]
        self.faces    = replace_references(self.faces, replacement)
        if angle_max != None: self.generate_vertex_normals()
       
        log.debug('merge_vertices_kdtree reduced vertex count from %i to %i', 
                  len(used),
                  len(unique))
                  
    @log_time
    def merge_vertices_hash(self):
        '''
        Removes duplicate vertices, based on integer hashes.
        This is roughly 20x faster than querying a KD tree in a loop
        '''
        
        tic       = time_function()
        pre_merge = len(self.vertices)

        unique, inverse = unique_rows(self.vertices, return_inverse=True)        
        self.faces      = inverse[[self.faces.reshape(-1)]].reshape((-1,3))
        self.vertices   = self.vertices[[unique]]

        log.debug('merge_vertices_hash reduced vertex count from %i to %i.',
                  pre_merge,
                  len(self.vertices))

    @log_time
    def remove_duplicate_faces(self):
        unique = unique_rows(np.sort(self.faces, axis=1), digits=0)
        log.debug('%i/%i faces were duplicate and have been removed',
                  len(self.faces) - len(unique),
                  len(self.faces))
        self.faces        = self.faces[[unique]]
        self.face_normals = self.face_normals[[unique]]
        
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
        self.face_normals = triangles_normal(self.vertices[[self.faces]])
        
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
        self.vertex_normals = unitize(np.mean(vertex_normals, axis=1))
        
    @log_time   
    def generate_face_colors(self, force_default=False):
        if (np.shape(self.face_colors) == (len(self.faces), 3)) and (not force_default): 
            return
        self.face_colors = np.tile(DEFAULT_COLOR, (len(self.faces), 1))
        
    @log_time       
    def generate_vertex_colors(self):
        '''
        Populate self.vertex_colors
        If self.face_colors are defined, we use those values to generate
        vertex colors. If not, we just set them to the DEFAULT_COLOR
        '''
        if np.shape(self.vertex_colors) == (len(self.vertices), 3): 
            return
        if np.shape(self.face_colors) == (len(self.faces), 3):
            # case where face_colors is populated, but vertex_colors isn't
            # we then generate vertex colors from the face colors
            vertex_colors = np.zeros((len(self.vertices), 3,3))
            vertex_colors[[self.faces[:,0],0]] = self.face_colors
            vertex_colors[[self.faces[:,1],1]] = self.face_colors
            vertex_colors[[self.faces[:,2],2]] = self.face_colors
            vertex_colors  = unitize(np.mean(vertex_colors, axis=1))
            vertex_colors *= (255.0 / np.max(vertex_colors, axis=1).reshape((-1,1)))
            self.vertex_colors = vertex_colors.astype(int)
            return
        self.vertex_colors = np.tile(DEFAULT_COLOR, (len(self.vertices), 1))
        
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
        return triangles_area(self.vertices[[self.faces]])
        
    @property
    def bounds(self):
        return np.vstack((np.min(self.vertices, axis=0),
                          np.max(self.vertices, axis=0)))
                          
    @property                 
    def centroid(self):
        return np.mean(self.vertices, axis=0)
        
    @property
    def box_size(self):
        return np.diff(self.bounds, axis=0)[0]
        
    @property
    def scale(self):
        return np.min(self.box_size)
        
    @log_time  
    def mass_properties(self, density = 1.0):
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
        return triangles_mass_properties(triangles = self.vertices[[self.faces]], 
                                         density   = density)

    def show(self, smooth = None, smooth_angle = np.radians(20)):
        '''
        Render the mesh in an opengl window. Requires pyglet.
        Smooth will re-merge vertices to fix the shading, but can be slow
        on larger meshes. 
        '''
        from mesh_render import MeshRender
        MeshRender(self, smooth=smooth, smooth_angle=smooth_angle)

    def export(self, filename):
        export_stl(self, filename)

def faces_to_edges(faces, sort=True):
    '''
    Given a list of faces (n,3), return a list of edges (n*3,2)
    '''
    edges = np.column_stack((faces[:,(0,1)],
                             faces[:,(1,2)],
                             faces[:,(2,0)])).reshape(-1,2)
    if sort: edges.sort(axis=1)
    return edges
        
def nondegenerate_faces(faces):
    '''
    Returns a 1D boolean array where non-degenerate faces are 'True'
    Faces should be (n, m) where for Trimeshes m=3. Returns (n) array
    '''
    nondegenerate = np.all(np.diff(np.sort(faces, axis=1), axis=1) != 0, axis=1)
    return nondegenerate 
        
def replace_references(data, reference_dict):
    '''
    Replace elements in an array as per a dictionary of replacement values

    data:           numpy array
    reference_dict: dictionary of replacements. example:
                       {2:1, 3:1, 4:5}
    '''
    shape = np.shape(data)
    view  = np.array(data).view().reshape((-1))
    for i, value in enumerate(view):
        if value in reference_dict:
            view[i] = reference_dict[value]
    return view.reshape(shape)

def detect_binary_file(file_obj):
    '''
    Returns True if file has non-ASCII characters (> 0x7F, or 127)
    Should work in both Python 2 and 3
    '''
    start  = file_obj.tell()
    fbytes = file_obj.read(1024)
    file_obj.seek(start)
    is_str = isinstance(fbytes, str)
    for fbyte in fbytes:
        if is_str: code = ord(fbyte)
        else:      code = fbyte
        if code > 127: return True
    return False
    
def cross_section(mesh, 
                  plane_origin  = [0,0,0], 
                  plane_normal  = [0,0,1]):
    '''
    Return a cross section of the trimesh based on plane origin and normal. 
    Basically a bunch of plane-line intersection queries

    origin:        (3) array of plane origin
    normal:        (3) array for plane normal
    return_planar: bool, True returns:
                         (m,2,2) list of 2D line segments
                         False returns:
                         (m,2,3) list of 3D line segments
    '''
    if len(mesh.faces) == 0: 
        raise NameError("Cannot compute cross section of empty mesh.")
    tic = time_function()
    edges = faces_to_edges(mesh.faces, sort=True)
    intersections, valid  = plane_line_intersection(plane_origin, 
                                                    plane_normal, 
                                                    mesh.vertices[[edges.T]],
                                                    line_segments = True)
    toc = time_function()
    log.debug('mesh_cross_section found %i intersections in %fs.', np.sum(valid), toc-tic)
    return intersections.reshape(-1,2,3)
    
def plane_line_intersection(plane_ori, 
                            plane_dir, 
                            endpoints,                            
                            line_segments = True):
    '''
    Calculates plane-line intersections

    Arguments
    ---------
    plane_ori: plane origin, (3) list
    plane_dir: plane direction (3) list
    endpoints: points defining lines to be intersected, (2,n,3)
    line_segments: if True, only returns intersections as valid if
                   vertices from endpoints are on different sides
                   of the plane.

    Returns
    ---------
    intersections: (m, 3) list of cartesian intersection points
    valid        : (n, 3) list of booleans indicating whether a valid
                   intersection occurred
    '''
    endpoints = np.array(endpoints)
    line_dir  = unitize(endpoints[1] - endpoints[0])
    plane_dir = unitize(plane_dir)

    t = np.dot(plane_dir, np.transpose(plane_ori - endpoints[0]))
    b = np.dot(plane_dir, np.transpose(line_dir))
    
    # If the plane normal and line direction are perpendicular, it means
    # the vector is 'on plane', and there isn't a valid intersection.
    # We discard on-plane vectors by checking that the dot product is nonzero
    valid = np.abs(b) > TOL_ZERO
    if line_segments:
        test = np.dot(plane_dir, np.transpose(plane_ori - endpoints[1]))
        different_sides = np.sign(t) != np.sign(test)
        valid           = np.logical_and(valid, different_sides)
        
    d  = np.divide(t[valid], b[valid])
    intersection  = endpoints[0][valid]
    intersection += np.reshape(d, (-1,1)) * line_dir[valid]
    return intersection, valid
    
def point_plane_distance(points, plane_normal, plane_origin=[0,0,0]):
    w         = np.array(points) - plane_origin
    distances = np.abs(np.dot(plane_normal, w.T) / np.linalg.norm(plane_normal))
    return distances

def unitize(points, check_valid=False):
    '''
    Flexibly turn a list of vectors into a list of unit vectors.
    
    Arguments
    ---------
    points:       (n,m) or (j) input array of vectors. 
                  For 1D arrays, points is treated as a single vector
                  For 2D arrays, each row is treated as a vector
    check_valid:  boolean, if True enables valid output and checking

    Returns
    ---------
    unit_vectors: (n,m) or (j) length array of unit vectors

    valid:        (n) boolean array, output only if check_valid.
                   True for all valid (nonzero length) vectors, thus m=sum(valid)
    '''
    points       = np.array(points)
    axis         = len(points.shape) - 1
    length       = np.sum(points ** 2, axis=axis) ** .5
    if check_valid:
        valid = np.greater(length, TOL_ZERO)
        if axis == 1: unit_vectors = (points[valid].T / length[valid]).T
        elif valid:   unit_vectors = points / length
        else:         unit_vectors = []
        return unit_vectors, valid
        
    unit_vectors = (points.T / length).T
    return unit_vectors
    
def major_axis(points):
    '''
    Returns an approximate vector representing the major axis of points
    '''
    sq = np.dot(np.transpose(points), points)
    d, v = np.linalg.eig(sq)
    return v[np.argmax(d)]
        
def surface_normal(points):
    '''
    Returns a normal estimate using SVD http://www.lsr.ei.tum.de/fileadmin/publications/KlasingAlthoff-ComparisonOfSurfaceNormalEstimationMethodsForRangeSensingApplications_ICRA09.pdf

    points: (n,m) set of points

    '''
    return np.linalg.svd(points)[2][-1]

def radial_sort(points, 
                origin = None, 
                normal = None):
    '''
    Sorts a set of points radially (by angle) around an origin/normal.
    If origin/normal aren't specified, it sorts around centroid
    and the approximate plane the points lie in. 

    points: (n,3) set of points
    '''
    #if origin and normal aren't specified, generate one at the centroid
    if origin==None: origin = np.average(points, axis=0)
    if normal==None: normal = surface_normal(points)
    
    #create two axis perpendicular to each other and the normal, and project the points onto them
    axis0 = [normal[0], normal[2], -normal[1]]
    axis1 = np.cross(normal, axis0)
    ptVec = points - origin
    pr0 = np.dot(ptVec, axis0)
    pr1 = np.dot(ptVec, axis1)

    #calculate the angles of the points on the axis
    angles = np.arctan2(pr0, pr1)

    #return the points sorted by angle
    return points[[np.argsort(angles)]]
               
def project_to_plane(points, 
                     normal           = [0,0,1], 
                     origin           = [0,0,0], 
                     return_transform = False):
    '''
    Projects a set of (n,3) points onto a plane, returning (n,2) points
    '''

    if np.all(np.abs(normal) < TOL_ZERO):
        raise NameError('Normal must be nonzero!')
    T        = align_vectors(normal, [0,0,1])
    T[0:3,3] = -np.array(origin) 
    xy       = transform_points(points, T)[:,0:2]
    if return_transform: return xy, T
    return xy

def planar_hull(mesh, 
                normal           = [0,0,1], 
                origin           = [0,0,0], 
                return_transform = False):
    from scipy.spatial import ConvexHull
    planar , T = project_to_plane(mesh.vertices,
                                  normal = normal,
                                  origin = origin,
                                  return_transform = True)
    hull_edges = ConvexHull(planar).simplices
    if return_transform:
        return planar[[hull_edges]], T
    return planar[[hull_edges]]
    
def counterclockwise_angles(vector, vectors):
    dots    = np.dot(vector, np.array(vectors).T)
    dets    = np.cross(vector, vectors)
    angles  = np.arctan2(dets, dots)
    angles += (angles < 0.0)*np.pi*2
    return angles

def hashable_rows(data, digits=None):
    '''
    We turn our array into integers, based on the precision 
    given by digits, and then put them in a hashable format. 
    
    Arguments
    ---------
    data:    (n,m) input array
    digits:  how many digits to add to hash, if data is floating point
             If none, TOL_MERGE will be turned into a digit count and used. 
    
    Returns
    ---------
    hashable:  (n) length array of custom data which can be sorted 
                or used as hash keys
    '''
    data = np.array(data)   
    if digits == None: digits = abs(int(np.log10(TOL_MERGE)))
     
    if data.dtype.kind in 'ib':
        #if data is an integer or boolean, don't bother multiplying by precision
        as_int = data
    else:
        as_int = ((data+10**-(digits+1))*10**digits).astype(np.int64)    
    hashable = np.ascontiguousarray(as_int).view(np.dtype((np.void, 
                                                         as_int.dtype.itemsize * as_int.shape[1]))).reshape(-1)
    return hashable
    
def unique_rows(data, return_inverse=False, digits=None):
    '''
    Returns indices of unique rows. It will return the 
    first occurrence of a row that is duplicated:
    [[1,2], [3,4], [1,2]] will return [0,1]
    '''
    hashes                   = hashable_rows(data, digits=digits)
    garbage, unique, inverse = np.unique(hashes, 
                                         return_index   = True, 
                                         return_inverse = True)
    if return_inverse: 
        return unique, inverse
    return unique
    
def group_rows(data, require_count=None, digits=None):
    '''
    Returns index groups of duplicate rows, for example:
    [[1,2], [3,4], [1,2]] will return [[0,2], [1]]
    
    Arguments
    ----------
    data:          (n,m) array
    require_count: only returns groups of a specified length, eg:
                   require_count =  2
                   [[1,2], [3,4], [1,2]] will return [[0,2]]
    
                   Note that using require_count allows numpy advanced indexing
                   to be used in place of looping and checking hashes, and as a
                   consequence is ~10x faster. 
                   
    digits:        If data is floating point, how many decimals to look at.
                   If this is None, the value in TOL_MERGE will be turned into a 
                   digit count and used. 

    Returns
    ----------
    groups:        List or sequence of indices from data indicating identical rows.
                   If require_count != None, shape will be (j, require_count)
                   If require_count == None, shape will be irregular (AKA a sequence)
    '''
    
    def group_dict():
        observed = dict()
        hashable = hashable_rows(data, digits=digits)
        for index, key in enumerate(hashable):
            key_string = key.tostring()
            if key_string in observed: observed[key_string].append(index)
            else:                      observed[key_string] = [index]
        return np.array(list(observed.values()))
        
    def group_slice():
        # create a representation of the rows that can be sorted
        hashable = hashable_rows(data, digits=digits)
        # record the order of the rows so we can get the original indices back later
        order    = np.argsort(hashable)
        # but for now, we want our hashes sorted
        hashable = hashable[order]
        # this is checking each neighbour for equality, example: 
        # example: hashable = [1, 1, 1]; dupe = [0, 0]
        dupe     = hashable[1:] != hashable[:-1]
        # we want the first index of a group, so we can slice from that location
        # example: hashable = [0 1 1]; dupe = [1,0]; dupe_idx = [0,1]
        dupe_idx = np.append(0, np.nonzero(dupe)[0] + 1)
        # if you wanted to use this one function to deal with non- regular groups
        # you could use: np.array_split(dupe_idx)
        # this is roughly 3x slower than using the group_dict method above. 
        start_ok   = np.diff(np.hstack((dupe_idx, len(hashable)))) == require_count
        groups     = np.tile(dupe_idx[start_ok].reshape((-1,1)), 
                             require_count) + np.arange(require_count)
        groups_idx = order[groups]
        if require_count == 1: 
            return groups_idx.reshape(-1)
        return groups_idx

    if require_count == None: return group_dict()
    else:                     return group_slice()

def group_vectors(vectors, 
                  TOL_ANGLE        = np.radians(10), 
                  include_negative = False):
    '''
    Group vectors based on an angle tolerance, with the option to 
    include negative vectors. 
    
    This is very similar to a group_rows(stack_negative(rows))
    The main difference is that TOL_ANGLE can be much looser, as we
    are doing actual distance queries. 
    '''
    from scipy.spatial import cKDTree as KDTree
    TOL_END             = np.tan(TOL_ANGLE)
    unit_vectors, valid = unitize(vectors, check_valid = True)
    valid_index         = np.nonzero(valid)[0]
    consumed            = np.zeros(len(unit_vectors), dtype=np.bool)
    tree                = KDTree(unit_vectors)
    unique_vectors      = deque()
    aligned_index       = deque()
    
    for index, vector in enumerate(unit_vectors):
        if consumed[index]: continue
        aligned = np.array(tree.query_ball_point(vector, TOL_END))        
        if include_negative:
            aligned = np.append(aligned, tree.query_ball_point(-1*vector, TOL_END))
        aligned = aligned.astype(int)
        consumed[[aligned]] = True
        unique_vectors.append(unit_vectors[aligned[-1]])
        aligned_index.append(valid_index[[aligned]])
    return np.array(unique_vectors), np.array(aligned_index)
    
def stack_negative(rows):
    '''
    Given an input of rows (n,d), return an array which is (n,2*d)
    Which is sign- independent
    '''
    rows     = np.array(rows)
    width    = rows.shape[1]
    stacked  = np.column_stack((rows, rows*-1))
    negative = rows[:,0] < 0
    stacked[negative] = np.roll(stacked[negative], 3, axis=1)
    return stacked
    
def connected_edges(G, nodes):
    '''
    Given graph G and list of nodes, return the list of edges that 
    are connected to nodes
    '''
    nodes_in_G = deque()
    for node in nodes:
        if not G.has_node(node): continue
        nodes_in_G.extend(nx.node_connected_component(G, node))
    edges = G.subgraph(nodes_in_G).edges()
    return edges

def facets(mesh, return_area = True):
    '''
    Returns lists of facets of a mesh. 
    Facets are defined as groups of faces which are both adjacent and parallel
    
    facets returned reference indices in mesh.faces
    If return_area is True, both the list of facets and their area are returned. 
    '''
    face_idx       = mesh.face_adjacency()
    normal_pairs   = mesh.face_normals[[face_idx]]
    parallel       = np.abs(np.sum(normal_pairs[:,0,:] * normal_pairs[:,1,:], axis=1) - 1) < TOL_ZERO
    graph_parallel = nx.from_edgelist(face_idx[parallel])
    facets         = list(nx.connected_components(graph_parallel))
    
    if not return_area: 
        return facets
    facets_area    = [triangles_area(mesh.vertices[[mesh.faces[facet]]]) for facet in facets]
    return facets, facets_area

def split_by_face_connectivity(mesh, check_watertight=True):
    '''
    Given a mesh, will split it up into a list of meshes based on face connectivity
    If check_watertight is true, it will only return meshes where each face has
    exactly 3 adjacent faces, which is a simple metric for being watertight.
    '''
    def mesh_from_components(connected_faces):
        if check_watertight:
            subgraph   = nx.subgraph(face_adjacency, connected_faces)
            watertight = np.equal(list(subgraph.degree().values()), 3).all()
            if not watertight: return
        faces  = mesh.faces[[connected_faces]]
        unique = np.unique(faces.reshape(-1))
        replacement = dict()
        replacement.update(np.column_stack((unique, np.arange(len(unique)))))
        faces = replace_references(faces, replacement).reshape((-1,3))
        new_meshes.append(Trimesh(vertices     = mesh.vertices[[unique]],
                                  faces        = faces,
                                  face_normals = mesh.face_normals[[connected_faces]]))
    face_adjacency = nx.from_edgelist(mesh.face_adjacency())
    new_meshes     = deque()
    list(map(mesh_from_components, nx.connected_components(face_adjacency)))
    log.info('split mesh into %i components.',
             len(new_meshes))
    return list(new_meshes)

def split_gt(mesh, check_watertight=True):
    g = Graph()
    g.add_edge_list(mesh.face_adjacency())
    
    p = label_components(g, directed=False)[0].a
    if check_watertight: d = g.degree_property_map('total').a
    meshes = deque()
    for component_id in np.unique(p):
        current  = np.equal(p, component_id)
        if (check_watertight) and (d[current] != 3).any(): continue
        vertices = mesh.vertices[mesh.faces[current]].reshape((-1,3))
        faces    = np.arange(len(vertices)).reshape((-1,3))
        meshes.append(Trimesh(faces=faces, vertices=vertices).process())
    return list(meshes)

def transform_points(points, matrix):
    '''
    Returns points, rotated by transformation matrix 
    If points is (n,2), matrix must be (3,3)
    if points is (n,3), matrix must be (4,4)
    '''
    dimension   = np.shape(points)[1]
    stacked     = np.column_stack((points, np.ones(len(points))))
    transformed = np.dot(matrix, stacked.T).T[:,0:dimension]
    return transformed

def align_vectors(vector_start, vector_end):
    '''
    Returns the 4x4 transformation matrix which will rotate from 
    vector_start (3,) to vector end (3,) 
    '''
    import transformations as tr
    vector_start = unitize(vector_start)
    vector_end   = unitize(vector_end)
    cross        = np.cross(vector_start, vector_end)
    # we clip the norm to 1, as otherwise floating point bs
    # can cause the arcsin to error
    norm         = np.clip(np.linalg.norm(cross), -TOL_ZERO, 1)
    if norm < TOL_ZERO:
        # if the norm is zero, the vectors are the same
        # and no rotation is needed
        return np.eye(4)
    angle = np.arcsin(norm) 
    T     = tr.rotation_matrix(angle, cross)
    return T

def triangles_cross(triangles):
    '''
    Returns the cross product of two edges from input triangles 

    triangles: vertices of triangles (n,3,3)
    returns:   cross product of two edge vectors (n,3)
    '''
    vectors = np.diff(triangles, axis=1)
    crosses = np.cross(vectors[:,0], vectors[:,1])
    return crosses
    
def triangles_area(triangles):
    '''
    Calculates the sum area of input triangles 

    triangles: vertices of triangles (n,3,3)
    returns:   area, (n)
    '''
    crosses = triangles_cross(triangles)
    area    = np.sum(np.sum(crosses**2, axis=1)**.5)*.5
    return area
    
def triangles_normal(triangles):
    '''
    Calculates the normals of input triangles 
    
    triangles: vertices of triangles, (n,3,3)
    returns:   normal vectors, (n,3)
    '''
    normals = unitize(triangles_cross(triangles))
    return normals
    
def triangles_all_coplanar(triangles):
    '''
    Given a list of triangles, return True if they are all coplanar, and False if not.
  
    triangles: vertices of triangles, (n,3,3)
    returns:   all_coplanar, bool
    '''
    test_normal  = triangles_normal(triangles)[0]
    test_vertex  = triangles[0][0]
    distances    = point_plane_distance(points       = triangles[1:].reshape((-1,3)),
                                        plane_normal = test_normal,
                                        plane_origin = test_vertex)
    all_coplanar = np.all(np.abs(distances) < TOL_ZERO)
    return all_coplanar
    
def triangles_any_coplanar(triangles):
    '''
    Given a list of triangles, if the first triangle is coplanar with ANY
    of the following triangles, return True.
    Otherwise, return False. 
    '''
    test_normal  = triangles_normal(triangles)[0]
    test_vertex  = triangles[0][0]
    distances    = point_plane_distance(points       = triangles[1:].reshape((-1,3)),
                                        plane_normal = test_normal,
                                        plane_origin = test_vertex)
    any_coplanar = np.any(np.all(np.abs(distances.reshape((-1,3)) < TOL_ZERO), axis=1))
    return any_coplanar
    
def triangles_mass_properties(triangles, density = 1.0):
    '''
    Calculate the mass properties of a group of triangles
    
    http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf
    '''
    crosses = triangles_cross(triangles)
    # thought vectorizing would make this slightly nicer, though it's still pretty ugly
    # these are the sub expressions of the integral 
    f1 = triangles.sum(axis=1)
    
    # triangles[:,0,:] will give rows like [[x0, y0, z0], ...] (the first vertex of every triangle)
    # triangles[:,:,0] will give rows like [[x0, x1, x2], ...] (the x coordinates of every triangle)
    f2 = (triangles[:,0,:]**2 +
          triangles[:,1,:]**2 +    
          triangles[:,0,:]*triangles[:,1,:] + 
          triangles[:,2,:]*f1)
          
    f3 = ((triangles[:,0,:]**3) + 
          (triangles[:,0,:]**2) * (triangles[:,1,:]) + 
          (triangles[:,0,:])    * (triangles[:,1,:]**2) +
          (triangles[:,1,:]**3) + 
          (triangles[:,2,:]*f2))
          
    g0 = (f2 + (triangles[:,0,:] + f1)*triangles[:,0,:])
    g1 = (f2 + (triangles[:,1,:] + f1)*triangles[:,1,:])
    g2 = (f2 + (triangles[:,2,:] + f1)*triangles[:,2,:])
    
    integral      = np.zeros((10, len(f1)))
    integral[0]   = crosses[:,0] * f1[:,0]
    integral[1:4] = (crosses * f2).T
    integral[4:7] = (crosses * f3).T
    
    for i in range(3):
        triangle_i    = np.mod(i+1, 3)
        integral[i+7] = crosses[:,i] * ((triangles[:,0, triangle_i] * g0[:,i]) + 
                                        (triangles[:,1, triangle_i] * g1[:,i]) + 
                                        (triangles[:,2, triangle_i] * g2[:,i]))
                                        
    coefficents = 1.0 / np.array([6,24,24,24,60,60,60,120,120,120])
    integrated  = integral.sum(axis=1) * coefficents
    
    volume      = integrated[0]
    center_mass = integrated[1:4] / volume

    inertia = np.zeros((3,3))
    inertia[0,0] = integrated[5] + integrated[6] - (volume * (center_mass[[1,2]]**2).sum())
    inertia[1,1] = integrated[4] + integrated[6] - (volume * (center_mass[[0,2]]**2).sum())
    inertia[2,2] = integrated[4] + integrated[5] - (volume * (center_mass[[0,1]]**2).sum())
    inertia[0,1] = (integrated[7] - (volume * np.product(center_mass[[0,1]])))
    inertia[1,2] = (integrated[8] - (volume * np.product(center_mass[[1,2]])))
    inertia[0,2] = (integrated[9] - (volume * np.product(center_mass[[0,2]])))
    inertia[2,0] = inertia[0,2]
    inertia[2,1] = inertia[1,2]
    inertia *= density
    
    # lists instead of numpy arrays, in case we want to serialize
    result = {'density'     : density,
              'volume'      : volume,
              'mass'        : density * volume,
              'center_mass' : center_mass.tolist(),
              'inertia'     : inertia.tolist()}
    return result
    
def load_stl(file_obj, file_type=None):
    if detect_binary_file(file_obj): return load_stl_binary(file_obj)
    else:                            return load_stl_ascii(file_obj)
        
def load_stl_binary(file_obj):
    '''
    Load a binary STL file into a trimesh object. 
    Uses a single main struct.unpack call, and is significantly faster
    than looping methods or ASCII STL. 
    '''
    # get the file_obj header
    header = file_obj.read(80)
    
    # get the file information about the number of triangles
    tri_count    = int(struct.unpack("@i", file_obj.read(4))[0])
    
    # now we check the length from the header versus the length of the file
    # data_start should always be position 84, but hard coding that felt ugly
    data_start = file_obj.tell()
    # this seeks to the end of the file (position 0, relative to the end of the file 'whence=2')
    file_obj.seek(0, 2)
    # we save the location of the end of the file and seek back to where we started from
    data_end = file_obj.tell()
    file_obj.seek(data_start)
    # the binary format has a rigidly defined structure, and if the length
    # of the file doesn't match the header, the loaded version is almost
    # certainly going to be garbage. 
    data_ok = (data_end - data_start) == (tri_count * 50)
   
    # this check is to see if this really is a binary STL file. 
    # if we don't do this and try to load a file that isn't structured properly 
    # the struct.unpack call uses 100% memory until the whole thing crashes, 
    # so it's much better to raise an exception here. 
    if not data_ok:
        raise NameError('Attempted to load binary STL with incorrect length in header!')
    
    # all of our vertices will be loaded in order due to the STL format, 
    # so faces are just sequential indices reshaped. 
    faces        = np.arange(tri_count*3).reshape((-1,3))

    # this blob extracts 12 float values, with 2 pad bytes per face
    # the first three floats are the face normal
    # the next 9 are the three vertices 
    blob = np.array(struct.unpack("<" + "12fxx"*tri_count, 
                                  file_obj.read())).reshape((-1,4,3))

    face_normals = blob[:,0]
    vertices     = blob[:,1:].reshape((-1,3))
    
    return Trimesh(vertices     = vertices,
                   faces        = faces, 
                   face_normals = face_normals)

def load_stl_ascii(file_obj):
    '''
    Load an ASCII STL file.
    
    Should be pretty robust to whitespace changes due to the use of split()
    '''
    
    header = file_obj.readline()
    blob   = np.array(file_obj.read().split())
    
    # there are 21 'words' in each face
    face_len     = 21
    face_count   = float(len(blob) - 1) / face_len
    if (face_count % 1) > TOL_ZERO:
        raise NameError('Incorrect number of values in STL file!')
    face_count   = int(face_count)
    # this offset is to be added to a fixed set of indices that is tiled
    offset       = face_len * np.arange(face_count).reshape((-1,1))
    # these hard coded indices will break if the exporter adds unexpected junk
    # but then it wouldn't really be an STL file... 
    normal_index = np.tile([2,3,4], (face_count, 1)) + offset
    vertex_index = np.tile([8,9,10,12,13,14,16,17,18], (face_count, 1)) + offset
    
    # faces are groups of three sequential vertices, as vertices are not references
    faces        = np.arange(face_count*3).reshape((-1,3))
    face_normals = blob[normal_index].astype(float)
    vertices     = blob[vertex_index.reshape((-1,3))].astype(float)
    
    return Trimesh(vertices     = vertices,
                   faces        = faces, 
                   face_normals = face_normals)
                   
def load_wavefront(file_obj, file_type=None):
    '''
    Loads a Wavefront .obj file_obj into a Trimesh object
    Discards texture normals and vertex color information
    https://en.wikipedia.org/wiki/Wavefront_.obj_file
    '''
    def parse_face(line):
        #faces are vertex/texture/normal and 1-indexed
        face = [None]*3
        for i in range(3):
            face[i] = int(line[i].split('/')[0]) - 1
        return face
    vertices = deque()
    faces    = deque()
    normals  = deque()
    line_key = {'vn': normals, 'v': vertices, 'f':faces}

    for raw_line in file_obj:
        line = raw_line.strip().split()
        if len(line) == 0: continue
        if line[0] ==  'v': vertices.append(list(map(float, line[-3:]))); continue
        if line[0] == 'vn': normals.append(list(map(float, line[-3:]))); continue
        if line[0] ==  'f': faces.append(parse_face(line[-3:]));
    mesh = Trimesh(vertices       = np.array(vertices, dtype=float),
                   faces          = np.array(faces,    dtype=int),
                   vertex_normals = np.array(normals,  dtype=float))
    mesh.generate_face_normals()
    return mesh
    
def export_stl(mesh, filename):
    '''
    Saves a Trimesh object as a binary STL file.
    '''
    def write_face(file_object, vertices, normal):
        #vertices: (3,3) array of floats
        #normal:   (3) array of floats
        file_object.write(struct.pack('<3f', *normal))
        for vertex in vertices: 
            file_object.write(struct.pack('<3f', *vertex))
        file_object.write(struct.pack('<h', 0))
    if len(mesh.face_normals) == 0: mesh.generate_normals()
    with open(filename, 'wb') as file_object:
        #write a blank header
        file_object.write(struct.pack("<80x"))
        #write the number of faces
        file_object.write(struct.pack("@i", len(mesh.faces)))
        # write the faces
        # TODO: remove the for loop and do this as a single struct.pack operation
        # like we do in the loader, as it is way, way faster.
        for index in range(len(mesh.faces)):
            write_face(file_object, 
                       mesh.vertices[[mesh.faces[index]]], 
                       mesh.face_normals[index])

def export_collada(mesh, filename):
    '''
    Export a mesh as collada, to filename
    '''
    import inspect
    MODULE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    template = Template(open(os.path.join(MODULE_PATH, 
                                          'templates', 
                                          'collada_template.dae'), 'rb').read())

    # we bother setting this because np.array2string uses these printoptions 
    np.set_printoptions(threshold=np.inf, precision=5, linewidth=np.inf)

    replacement = dict()
    replacement['VERTEX']   = np.array2string(mesh.vertices.reshape(-1))[1:-1]
    replacement['FACES']    = np.array2string(mesh.faces.reshape(-1))[1:-1]
    replacement['NORMALS']  = np.array2string(mesh.vertex_normals.reshape(-1))[1:-1]
    replacement['VCOUNT']   = str(len(mesh.vertices))
    replacement['VCOUNTX3'] = str(len(mesh.vertices) * 3)
    replacement['FCOUNT']   = str(len(mesh.faces))
    with open(filename, 'wb') as outfile:
        outfile.write(template.substitute(replacement))

_MESH_LOADERS   = {'stl': load_stl, 
                  'obj': load_wavefront}

_ASSIMP_FORMATS = ['dae', 
                   'blend', 
                   '3ds', 
                   'ase', 
                   'obj', 
                   'ifc', 
                   'xgl', 
                   'zgl',
                   'ply',
                   'lwo',
                   'lxo',
                   'x',
                   'ac',
                   'ms3d',
                   'cob',
                   'scn']
try: 
    import pyassimp
    _MESH_LOADERS.update(zip(_ASSIMP_FORMATS, 
                             [load_assimp]*len(_ASSIMP_FORMATS)))
except:
    log.debug('No pyassimp, only native loaders available!')
        
if __name__ == '__main__':
    formatter = logging.Formatter("[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s", 
                                  "%Y-%m-%d %H:%M:%S")
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    handler_stream.setLevel(logging.DEBUG)
    log.setLevel(logging.DEBUG)
    log.addHandler(handler_stream)
    np.set_printoptions(precision=6, suppress=True)
    
    m = load_mesh('models/kinematic_tray.STL').process()
    #mesh.show()
    m = load_mesh('models/src_bot.STL').process()
    
    s = m.split()
