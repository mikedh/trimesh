'''
trimesh.py

Library for importing and doing simple operations on triangular meshes
Styled after transformations.py
'''

import numpy as np
import struct
from collections import deque
import logging
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import ConvexHull
from time import time as time_func
from string import Template
from copy import deepcopy
import networkx as nx
import os
import inspect

from StringIO import StringIO

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

MODULE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

TOL_ZERO  = 1e-12
TOL_MERGE = 1e-9

def load_mesh(file_obj, type=None):
    '''
    Load a mesh file into a Trimesh object

    file_obj: a filename string, or a file object
    '''

    mesh_loaders = {'stl': load_stl, 
                    'obj': load_wavefront}
    if type == None and file_obj.__class__.__name__ == 'str':
        type     = (str(file_obj).split('.')[-1]).lower()
        file_obj = open(file_obj, 'rb')

    tic = time_func()
    mesh = mesh_loaders[type](file_obj)
    file_obj.close()
    toc = time_func()

    log.info('loaded mesh from %s container, with %i faces and %i vertices in %fs.', 
             type, 
             len(mesh.faces),
             len(mesh.vertices),
             toc-tic)

    return mesh


class Trimesh():
    def __init__(self, 
                 vertices        = None, 
                 faces           = None, 
                 face_normals    = None, 
                 vertex_normals  = None,
                 edges           = None, 
                 color_faces     = None,
                 color_vertices  = None):
        self.vertices        = vertices
        self.faces           = faces
        self.face_normals    = face_normals
        self.vertex_normals  = vertex_normals
        self.edges           = edges
        self.color_faces     = color_faces
        self.color_vertices  = color_vertices

        self.generate_bounds()
        #self.verify_normals()
        
    def __add__(self, other):
        other_faces    = np.array(other.faces).astype(int) + len(self.vertices)
        other_vertices = np.array(other.vertices).astype(float)        
        other_normals  = np.array(other.face_normals).astype(float)
        
        new_vertices = np.vstack((self.vertices, other_vertices))
        new_faces    = np.vstack((self.faces, other_faces))
        new_normals  = np.vstack((self.face_normals, other_normals))
        
        return Trimesh(vertices = new_vertices, 
                       faces    = new_faces, 
                       face_normals = new_normals)

    def area(self):
        area_sum = 0.0

        for face in self.faces:
            vertices = self.vertices[[face]]
            cross = np.cross(vertices[0] - vertices[1],
                             vertices[0] - vertices[2])
            area_sum += .5*np.linalg.norm(cross)
        return area_sum
            


    def verify_normals(self):
        if ((self.face_normals == None) or
            (np.shape(self.face_normals) <> np.shape(self.faces))):
            log.debug('generating face normals for faces %s and passed face normals %s',
                      str(np.shape(self.faces)),
                      str(np.shape(self.face_normals)))
            self.generate_face_normals()

        if ((self.vertex_normals == None) or 
            (np.shape(self.vertex_normals) <> np.shape(self.vertices))): 
            self.generate_vertex_normals()

    def convex_hull(self, merge_radius=1e-3):
        '''
        Get a new Trimesh object representing the convex hull of the 
        current mesh. Requires scipy >.12, and doesn't produce properly directed normals

        merge_radius: when computing a complex hull, at what distance do we merge close vertices 
        '''
        from scipy.spatial import ConvexHull
        mesh = Trimesh()
        mesh.vertices = self.vertices
        mesh.faces = np.array([])
        mesh.merge_vertices()
        mesh.faces = ConvexHull(mesh.vertices).simplices
        mesh.remove_unreferenced()
        return mesh

    def merge_vertices(self, angle_max=None):
        if angle_max <> None:
            self.merge_vertices_kdtree(angle_max)
        else:
            self.merge_vertices_hash()

    def merge_vertices_kdtree(self, angle_max=None):
        '''
        Merges vertices which are identical, AKA within 
        cartesian distance TOL_MERGE of each other.  
        Then replaces references in self.faces
        
        If angle_max == None, vertex normals won't be looked at. 
        if angle_max has a value, vertices will only be considered identical
        if they are within TOL_MERGE of each other, and the angle between
        their normals is less than angle_max

        Performance note:
        cKDTree requires scipy >= .12 for this query type and you 
        probably don't want to use plain python KDTree as it is crazy slow (~1000x in tests)
        '''
        tic         = time_func()
        tree        = KDTree(self.vertices)
        used        = np.zeros(len(self.vertices), dtype=np.bool)
        unique      = deque()
        replacement = dict()
        
        if angle_max <> None: self.verify_normals()

        for index, vertex in enumerate(self.vertices):
            if used[index]: continue
            neighbors = np.array(tree.query_ball_point(self.vertices[index], TOL_MERGE))
            used[[neighbors]] = True
            if angle_max <> None:
                normals, aligned = group_vectors(self.vertex_normals[[neighbors]],
                                                 TOL_ANGLE = angle_max)
                for group in aligned:
                    vertex_indices = neighbors[[group]]
                    replacement.update(np.column_stack((vertex_indices,
                                                        [len(unique)] * len(group))))
                    unique.append(vertex_indices[0])
            else:
                replacement.update(np.column_stack((neighbors,
                                                    [len(unique)]*len(neighbors))))
                unique.append(neighbors[0])

     
        self.vertices = self.vertices[[unique]]
        replace_references(self.faces, replacement)
        if angle_max <> None: self.generate_vertex_normals()
       
        log.debug('merge_vertices_kdtree reduced vertex count from %i to %i in %.4fs.',
                  len(used),
                  len(unique),
                  time_func()-tic)
                  

    def merge_vertices_hash(self):
        '''
        Removes duplicate vertices, based on integer hashes.
        This is roughly 20x faster than querying a KD tree in a loop
        '''
        tic         = time_func()
        digits = abs(int(np.log10(TOL_MERGE)))
     
        # we turn our array into integers, based on the precision given by 
        # TOL_MERGE (which we turn into a digit count)
   

        as_int = ((self.vertices+10**-(digits+1))*10**digits).astype(np.int64)
    
        hashes = as_int.view(np.dtype((np.void, 
                                       as_int.dtype.itemsize * as_int.shape[1])))
      
        garbage, unique, inverse = np.unique(hashes, 
                                             return_index   = True, 
                                             return_inverse = True)
        
        self.faces    = inverse[[self.faces.reshape(-1)]].reshape((-1,3))
        self.vertices = self.vertices[[unique]]

        log.debug('merge_vertices_hash reduced vertex count from %i to %i in %.4fs.',
                  len(hashes),
                  len(unique),
                  time_func()-tic)

    def unmerge_vertices(self):
        '''
        Removes all face references, so that every face contains
        three unique vertex indices.
        '''
        self.vertices = self.vertices[[self.faces]].reshape((-1,3))
        self.faces    = np.arange(len(self.vertices)).reshape((-1,3))
        
    def remove_unreferenced(self):
        '''
        Removes all vertices which aren't in a face
        Reindexes vertices from zero and replaces face references
        '''
        referenced = self.faces.view().reshape(-1)
        unique_ref = np.unique(referenced).astype(int)
        replacement = dict()
        replacement.update(np.column_stack((unique_ref,
                                            range(len(unique_ref)))))                            
        replace_references(self.faces, replacement)
        self.vertices = self.vertices[[unique_ref]]

    def generate_edges(self):
        '''
        Populate self.edges from face information.
        '''
        self.edges = np.column_stack((self.faces[:,(0,1)],
                                      self.faces[:,(1,2)],
                                      self.faces[:,(2,0)])).reshape(-1,2)
        self.edges.sort(axis=1)
        
    def generate_face_normals(self):
        '''
        If no normal information is loaded, we can get it from cross products
        Normal direction will be incorrect if mesh faces aren't ordered (right-hand rule)
        '''
        self.face_normals = np.zeros((len(self.faces),3))
        self.vertices = np.array(self.vertices)
        for index, face in enumerate(self.faces):
            v0 = (self.vertices[face[2]] - self.vertices[face[1]])
            v1 = (self.vertices[face[0]] - self.vertices[face[1]])
            self.face_normals[index] = unitize(np.cross(v0, v1))
        
    def generate_vertex_normals(self):
        '''
        If face normals are defined, produce approximate vertex normals based on the
        average of the adjacent faces.
        
        If vertices are merged with no regard to normal angle, this is
        going to render super weird. 
        '''
        vertex_normals = np.zeros((len(self.vertices), 3,3))
        vertex_normals[[self.faces[:,0],0]] = self.face_normals
        vertex_normals[[self.faces[:,1],1]] = self.face_normals
        vertex_normals[[self.faces[:,2],2]] = self.face_normals
        self.vertex_normals = unitize(np.mean(vertex_normals, axis=1))

    def transform(self, matrix):
        stacked = np.column_stack((self.vertices, np.ones(len(self.vertices))))
        self.vertices = np.dot(matrix, stacked.T)[:,0:3]

    def generate_bounds(self):
        self.bounds   = np.vstack((np.min(self.vertices, axis=0),
                                   np.max(self.vertices, axis=0)))
        self.centroid = np.mean(self.vertices, axis=0)
        self.box_size = np.diff(self.bounds, axis=0)[0]
        self.scale    = np.min(self.box_size)
  
    def show(self, smooth = False):
        from mesh_render import MeshRender
        MeshRender(self, smooth=smooth)

    def export(self, filename):
        export_stl(self, filename)

def replace_references(data, reference_dict, return_array=False):
    '''
    Replace elements in an array as per a dictionary of replacement values

    data:           numpy array
    reference_dict: dictionary of replacements. example:
                       {2:1, 3:1, 4:5}

    return_array: if False, replaces references in place and returns nothing
    '''
    dv = data.view().reshape((-1))
    for i in xrange(len(dv)):
        if dv[i] in reference_dict:
            dv[i] = reference_dict[dv[i]]
    if return_array: return dv

def detect_binary_file(file_obj):
    '''
    Returns True if file has non-ascii charecters
    http://stackoverflow.com/questions/898669/how-can-i-detect-if-a-file-is-binary-non-text-in-python
    '''
    textchars = ''.join(map(chr, [7,8,9,10,12,13,27] + range(0x20, 0x100)))
    start     = file_obj.tell()
    fbytes    = file_obj.read(1024)
    file_obj.seek(start)
    return bool(fbytes.translate(None, textchars))

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
        different_sides = np.sign(t) <> np.sign(test)
        valid           = np.logical_and(valid, different_sides)
        
    d  = np.divide(t[valid], b[valid])
    intersection  = endpoints[0][valid]
    intersection += np.reshape(d, (-1,1)) * line_dir[valid]
    return intersection, valid
    
def point_plane_distance(plane_ori, plane_dir, points):
    w = points - plane_ori
    return np.abs(np.dot(plane_dir, w.T) / np.linalg.norm(plane_dir))
    
def unitize(points, error_on_zero=False):
    '''
    Unitize vectors by row
    one vector (3,) gets vectorized correctly, 
    as well as 10 row vectors (10,3)

    points: numpy array/list of points to be unit vector'd
    error_on_zero: if zero magnitude vectors exist, throw an error
    '''
    points = np.array(points)
    axis   = len(points.shape)-1
    norms  = np.sum(points ** 2, axis=axis) ** .5
    nonzero = norms > TOL_ZERO
    if (error_on_zero and
        not np.all(nonzero)):
        raise NameError('Unable to unitize zero length vector!')
    if axis==1:
        points[nonzero] =  (points[nonzero].T / norms[nonzero]).T
        return points
    return (points.T / norms).T
    
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
               
def project_to_plane(points, origin=[0,0,0], normal=[0,0,1]):
    '''
    projects a set of (n,3) points onto a plane, returning (n,2) points
    '''
    # we first establish a set of perpendicular axis. 
    
    if np.all(np.abs(normal) < TOL_ZERO):
        raise NameError('Normal must be nonzero!')
    for i in xrange(3):
        test_axis = np.abs(np.cross(normal, np.roll([0,1,0], i)))
        if np.linalg.norm(test_axis) < TOL_ZERO: continue
        test_axis = unitize(test_axis)
        if np.any(np.abs(test_axis) > TOL_ZERO): break
    if np.all(np.abs(test_axis) < TOL_ZERO): 
        raise NameError('Unable to find projection axis!')
    axis0 = test_axis
    axis1 = unitize(np.cross(normal, axis0))
  
    log.debug('Projecting points to axis %s, %s', 
              str(axis0), 
              str(axis1))
    
    pt_vec = np.array(points) - origin
    pr0 = np.dot(axis0, pt_vec.T)
    pr1 = np.dot(axis1, pt_vec.T)
    return np.column_stack((pr0, pr1))
 
def cross_section(mesh, 
                  plane_origin  = [0,0,0], 
                  plane_normal  = [0,0,1], 
                  return_planar = True):
    '''
    Return a cross section of the trimesh based on plane origin and normal. 
    Basically a bunch of plane-line intersection queries
    Depends on properly ordered edge information, as done by generate_edges

    origin:        (3) array of plane origin
    normal:        (3) array for plane normal
    return_planar: bool, True returns:
                         (m,2,2) list of 2D line segments
                         False returns:
                         (m,2,3) list of 3D line segments
    '''
    if len(mesh.faces) == 0: 
        raise NameError("Cannot compute cross section of empty mesh.")
    tic = time_func()
    mesh.generate_edges()
    intersections, valid  = plane_line_intersection(plane_origin, 
                                                    plane_normal, 
                                                    mesh.vertices[[mesh.edges.T]],
                                                    line_segments = True)
    toc = time_func()
    log.debug('mesh_cross_section found %i intersections in %fs.', np.sum(valid), toc-tic)
    
    if return_planar: 
        planar = project_to_plane(intersections, 
                                  plane_origin, 
                                  plane_normal).reshape((-1,2,2))
        return planar
    else: 
        return intersections.reshape(-1,2,3)

def planar_outline(mesh, 
                   plane_origin = [0,0,0],
                   plane_normal = [0,0,-1]):
    '''
    Arguments
    ---------
    mesh: Trimesh object
    plane_origin: plane origin, (3) list
    plane_normal: plane normal, (3) list
                  Note that if plane_normal is NOT aligned with an axis,
                  The R-tree based triangle picking we're using isn't going to do much
                  and tracing a ray becomes very, very n^2
    output:
    (n,2,2) vertices of a closed polygon representing the outline of the mesh
          projected onto the specified plane. 
    '''
    from raytracing import RayTracer
    mesh.merge_vertices()
    # only work with faces pointed towards us (AKA discard back-faces)
    visible_faces = mesh.faces[np.dot(plane_normal, 
                                      mesh.face_normals.T) > -TOL_ZERO]
                                      
    # create a raytracer for only visible faces
    r = RayTracer(Trimesh(vertices = mesh.vertices, faces = visible_faces))
    visible_edges = np.column_stack((visible_faces[:,(0,1)],
                                     visible_faces[:,(1,2)],
                                     visible_faces[:,(2,0)])).reshape(-1,2)
    visible_edges.sort(axis=1)
    # the edges that make up the outline of the mesh will only appear once
    # however this doesn't remove internal geometry or local edges. 
    # in order to only get the contour outline, we need a visibility check
    contour_edges = visible_edges[unique_rows(visible_edges)]
    
    # the rays are coming towards the projection plane
    ray_dir    = unitize(plane_normal) * - 1
    # we start the rays just past the mesh bounding box
    ray_offset = unitize(plane_normal) * np.max(np.ptp(mesh.bounds, axis=0))*1.25
    offset_max = 1e-3 * np.clip(mesh.scale, 0.0, 1.0)
    edge_thru  = deque()
    for edge in contour_edges:
        edge_center = np.mean(mesh.vertices[[edge]], axis=0)
        edge_vector = np.diff(mesh.vertices[[edge]], axis=0)
        edge_length = np.linalg.norm(edge_vector)
        try: 
            edge_normal = unitize(np.cross(ray_dir, edge_vector)[0], error_on_zero=True)
        except:  continue
        ray_perp = edge_normal * np.clip(edge_length * .5, 0, offset_max)
        ray_0 = edge_center + ray_offset + ray_perp 
        ray_1 = edge_center + ray_offset - ray_perp
        inersections_0 = len(r.intersect_ray(ray_0, ray_dir))
        inersections_1 = len(r.intersect_ray(ray_1, ray_dir))
        if (inersections_0 == 0) <> (inersections_1 == 0):
            edge_thru.append(edge)

    edge_thru = np.array(edge_thru).reshape(-1)
    contour_lines = project_to_plane(mesh.vertices[[edge_thru]],
                                     origin = plane_origin,
                                     normal = plane_normal).reshape((-1,2,2))
    return contour_lines
    
def planar_hull(mesh, plane_normal=[0,0,1]):
    planar = project_to_plane(mesh.vertices,
                              normal = plane_normal)
    hull_edges = ConvexHull(planar).simplices
    return planar[[hull_edges]]
    
def counterclockwise_angles(vector, vectors):
    dots    = np.dot(vector, np.array(vectors).T)
    dets    = np.cross(vector, vectors)
    angles  = np.arctan2(dets, dots)
    angles += (angles < 0.0)*np.pi*2
    return angles

def unique_rows(data, return_first=False, decimals=6):
    '''
    Returns unique rows of an array, using string hashes. 
    '''
    def row_to_string(row):
        result = ''
        for i in row: 
            result += format(i, format_str)
        return result
        
    first_occur = dict()
    format_str  = '.' + str(decimals) + 'f'
    unique      = np.ones(len(data), dtype=np.bool)
    for index, row in enumerate(data):
        hashable = row_to_string(row)
        if hashable in first_occur:
            unique[index]                     = False
            if not return_first:
                unique[first_occur[hashable]] = False
        else:
            first_occur[hashable] = index
    return unique
    
def group_rows(data, decimals=6):
    '''
    Returns unique rows of an array, using string hashes. 
    '''
    def row_to_string(row):
        result = ''
        for i in row: 
            result += format(i, format_str)
        return result
        
    observed = dict()
    format_str  = '.' + str(decimals) + 'f'
    for index, row in enumerate(data):
        hashable = row_to_string(row)
        if hashable in observed:
            observed[hashable] = np.append(observed[hashable], index)
        else:
            observed[hashable] = np.array([index])
    return np.array(observed.values())

def group_vectors(vectors, TOL_ANGLE=np.radians(10), include_negative=False):
    TOL_END        = np.tan(TOL_ANGLE)
    unit_vectors   = unitize(vectors)
    tree           = KDTree(unit_vectors)

    unique_vectors = deque()
    aligned_index  = deque()
    consumed       = np.zeros(len(unit_vectors), dtype=np.bool)

    for index, vector in enumerate(unit_vectors):
        if consumed[index]: continue
        aligned = np.array(tree.query_ball_point(vector, TOL_END))        
        if include_negative:
            aligned = np.append(aligned, 
                                tree.query_ball_point(-1*vector, TOL_END))

        aligned = aligned.astype(int)
        consumed[[aligned]] = True
        test = np.sum((unit_vectors[[aligned]] - vector)**2, axis=1)**.5
        #unique_vectors.append(vector)
        unique_vectors.append(np.percentile(vectors[[aligned]], q=75, axis=0))
        aligned_index.append(aligned)
    return np.array(unique_vectors), np.array(aligned_index)
    
def load_stl(file_obj):
    if detect_binary_file(file_obj): return load_stl_binary(file_obj)
    else:                            return load_stl_ascii(file_obj)
        
def load_stl_binary(file_obj):
    #get the file_obj header
    header = file_obj.read(80)
    
    #use the header information about number of triangles
    tri_count    = int(struct.unpack("@i", file_obj.read(4))[0])
    faces        = np.arange(tri_count*3).reshape((-1,3))
    
    # this blob extracts 12 float values, with 2 pad bytes per face
    # the first three floats are the face normal
    # the next 9 are the three vertices 
    blob = np.array(struct.unpack("<" + "12fxx"*tri_count, file_obj.read(50*tri_count))).reshape((-1,4,3))

    face_normals = blob[:,0]
    vertices     = blob[:,1:].reshape((-1,3))
    
    return Trimesh(vertices     = vertices,
                   faces        = faces, 
                   face_normals = face_normals)

def load_stl_ascii(file_obj):
    def parse_line(line):
        return map(float, line.strip().split(' ')[-3:])
    def read_face(file_obj):
        normals.append(parse_line(file_obj.readline()))
        faces.append(np.arange(0,3) + len(vertices))
        file_obj.readline()
        for i in xrange(3):      
            vertices.append(parse_line(file_obj.readline()))
        file_obj.readline(); file_obj.readline()
    faces    = deque() 
    normals  = deque()
    vertices = deque()

    #get the file header
    header = file_obj.readline()
    while True:
        try: read_face(file_obj)
        except: break
    return Trimesh(faces       = np.array(faces,    dtype=np.int),
                   face_normals = np.array(normals, dtype=np.float),
                   vertices    = np.array(vertices, dtype=np.float))

def load_wavefront(file_obj):
    '''
    Loads a Wavefront .obj file_obj into a Trimesh object
    Discards texture normals and vertex color information
    https://en.wikipedia.org/wiki/Wavefront_.obj_file
    '''
    def parse_face(line):
        #faces are vertex/texture/normal and 1-indexed
        face = [None]*3
        for i in xrange(3):
            face[i] = int(line[i].split('/')[0]) - 1
        return face
    vertices = deque()
    faces    = deque()
    normals  = deque()
    line_key = {'vn': normals, 'v': vertices, 'f':faces}

    for raw_line in file_obj:
        line = raw_line.strip().split()
        if len(line) == 0: continue
        if line[0] ==  'v': vertices.append(map(float, line[-3:])); continue
        if line[0] == 'vn': normals.append(map(float, line[-3:])); continue
        if line[0] ==  'f': faces.append(parse_face(line[-3:]));
    mesh = Trimesh(vertices      = np.array(vertices, dtype=float),
                   faces         = np.array(faces,    dtype=int),
                   vertex_normals = np.array(normals,  dtype=float))
    mesh.generate_face_normals()
    return mesh
    
def export_stl(mesh, filename):
    #Saves a Trimesh object as a binary STL file. 
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
        #write the faces
        for index in xrange(len(mesh.faces)):
            write_face(file_object, 
                       mesh.vertices[[mesh.faces[index]]], 
                       mesh.face_normals[index])

def export_collada(mesh, filename):
    template = Template(open(os.path.join(MODULE_PATH, 
                                          'templates', 
                                          'collada_template.dae'), 'rb').read())
    
    #np.array2string uses the numpy printoptions
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

def connected_edges(G, nodes):

    #Given graph G and list of nodes, return the list of edges that 
    #are connected to nodes

    nodes_in_G = deque()
    for node in nodes:
        if not G.has_node(node): continue
        nodes_in_G.extend(nx.node_connected_component(G, node))
    edges = G.subgraph(nodes_in_G).edges()
    return edges
 
def mesh_facets(mesh, angle_max=np.radians(50)):

    mesh.merge_vertices(angle_max = angle_max)
    mesh.generate_edges()
    g = nx.Graph()
    g.add_edges_from(mesh.edges)

    connected_vertices = nx.connected_components(g)
    facet_faces        = deque()
    for vertices in connected_vertices:
        facet = np.all(np.in1d(mesh.faces.reshape(-1), vertices).reshape((-1,3)), axis=1)
        facet_faces.append(mesh.faces[facet])
    return list(facet_faces)


def split_by_connectivity(mesh):
    tic = time_func()
    mesh.generate_edges()
    g = nx.Graph()
    g.add_edges_from(mesh.edges)

    components = nx.connected_components(g)
    new_meshes = [None] * len(components)

    for i, connected in enumerate(components): 
        mask = np.zeros(len(mesh.vertices))
        mask[[connected]] = 1
        face_subset = np.sum(mask[[mesh.faces]], axis=1) <> 0
        vertices = mesh.vertices[[mesh.faces[face_subset].reshape(-1)]]
        normals  = mesh.face_normals[face_subset]
        faces    = np.arange(len(vertices)).reshape((-1,3))

        new_meshes[i] = Trimesh(vertices     = vertices,
                                faces        = faces,
                                face_normals = normals)
    toc = time_func()
    log.info('split mesh into %i components in %fs',
             len(new_meshes),
             toc-tic)
    return new_meshes

def cut_axis(mesh):
    mesh.merge_vertices()
    mesh.generate_edges()

    # this will return the indices for duplicate edges
    # every edge appears twice in a well constructed mesh
    # so for every row in edge_idx, mesh.edges[edge_idx[*][0]] == mesh.edges[edge_idx[*][1]]
    edge_idx = group_rows(mesh.edges)

    # returns the pairs of all adjacent faces
    # so for every row in face_idx, mesh.faces[face_idx[*][0]] and mesh.faces[face_idx[*][1]]
    # will share an edge
    face_idx = np.tile(np.arange(len(mesh.faces)), (3,1)).T.reshape(-1)[[edge_idx]]
    normal_pairs = mesh.face_normals[[face_idx]]

    pair_axis   = np.cross(normal_pairs[:,0,:], normal_pairs[:,1,:])
    pair_norms  = np.sum(pair_axis ** 2, axis=(1)) ** .5
    parallel    = pair_norms < TOL_ZERO
    nonparallel = np.logical_not(parallel)
    pair_axis[nonparallel] *= 1.0 / pair_norms[nonparallel].reshape((-1,1))

    #remove duplicate pair axis vectors, so we have easier comparisons
    #while traversing
    pair_vectors, aligned = group_vectors(pair_axis, 
                                          TOL_ANGLE = np.radians(10), 
                                          include_negative=True)

    graph_parallel = nx.Graph()
    graph_parallel.add_edges_from(face_idx[parallel])

    for axis_vector, group in zip(pair_vectors, aligned):
        if np.linalg.norm(axis_vector) < TOL_ZERO: continue
        graph_group = nx.Graph()
        graph_group.add_edges_from(face_idx[[group]])

        parallel_connected_faces = connected_edges(graph_parallel, graph_group.nodes())
        face_dot = np.dot(mesh.face_normals[[parallel_connected_faces]].reshape((-1,3)), 
                          axis_vector).reshape((-1,2))
        along_axis = np.all(face_dot < TOL_ZERO, axis=1)
        parallel_along_axis = np.array(parallel_connected_faces)[along_axis]


        graph_group.add_edges_from(parallel_along_axis)


        cycles = nx.cycle_basis(graph_group)
        if len(cycles) == 0: continue
        cycle_faces = np.unique(np.hstack(cycles))
        graph_parallel.remove_nodes_from(cycle_faces)

        #print '\n\n', cycle_faces

        display = Trimesh(vertices     = mesh.vertices, 
                          faces        = mesh.faces[[cycle_faces]], 
                          face_normals = mesh.face_normals[[cycle_faces]])
        display.show()

if __name__ == '__main__':
    formatter = logging.Formatter("[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s", "%Y-%m-%d %H:%M:%S")
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    handler_stream.setLevel(logging.DEBUG)
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    log.addHandler(handler_stream)

    np.set_printoptions(precision=4, suppress=True)
        
    import matplotlib.pyplot as plt
    import time

    '''

    m = load_mesh('./src_bot.STL')

    import cProfile, pstats, StringIO
    pr = cProfile.Profile()
    pr.enable()

    m.merge_vertices()

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
    '''


    #m = Trimesh(vertices = np.random.random((100,2)),
    #            faces    = np.arange(12).reshape((-1,3)))
