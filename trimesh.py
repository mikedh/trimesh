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

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

TOL_ZERO  = 1e-12
TOL_MERGE = 1e-7

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
    
    if type in mesh_loaders:
        mesh = mesh_loaders[type](file_obj)
        file_obj.close()
        return mesh
    raise NameError('No mesh loader for files of type .' + type)

class Trimesh():
    def __init__(self, 
                 vertices      = None, 
                 faces         = None, 
                 normal_face   = None, 
                 normal_vertex = None,
                 edges         = None, 
                 color_face    = None,
                 color_vertex  = None):
        self.vertices      = vertices
        self.faces         = faces
        self.normal_face   = normal_face
        self.normal_vertex = normal_vertex
        self.edges         = edges
        self.color_face    = color_face
        self.color_vertex  = color_vertex

        self.merged = False
        
        self.generate_bounds()

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
        mesh.merge_vertices(merge_radius)
        mesh.faces = ConvexHull(mesh.vertices).simplices
        mesh.remove_unreferenced()
        return mesh

    def merge_vertices(self):
        '''
        Merges vertices which are identical and replaces references
        Does this by creating a KDTree.
        cKDTree requires scipy >= .12 for this query type and you 
        probably don't want to use plain python KDTree as it is crazy slow (~1000x in my tests)

        tolerance: to what precision do vertices need to be identical
        '''

        if self.merged: return 
        self.merged = True
        
        tree    = KDTree(self.vertices)
        used    = np.zeros(len(self.vertices), dtype=np.bool)
        unique  = []
        replacement_dict = dict()

        for index, vertex in enumerate(self.vertices):
            if used[index]: continue
            neighbors = tree.query_ball_point(self.vertices[index], TOL_MERGE)
            used[[neighbors]] = True
            replacement_dict.update(np.column_stack((neighbors,
                                                     [len(unique)]*len(neighbors))))
            unique.append(index)
        log.debug('merge_vertices reduced vertex count from %i to %i.',
                  len(self.vertices),
                  len(unique))
        self.vertices = self.vertices[[unique]]
        replace_references(self.faces, replacement_dict)

    def remove_unreferenced(self):
        '''
        Removes all vertices which aren't in a face
        Reindexes vertices from zero and replaces face references
        '''
        referenced = self.faces.view().reshape(-1)
        unique_ref = np.int_(np.unique(referenced))
        replacement_dict = dict()
        replacement_dict.update(np.column_stack((unique_ref,
                                                 range(len(unique_ref)))))                                         
        replace_references(self.faces, replacement_dict)
        self.vertices = self.vertices[[unique_ref]]

    def generate_edges(self):
        '''
        Populate self.edges from face information
        '''
        self.edges = np.column_stack((self.faces[:,(0,1)],
                                      self.faces[:,(1,2)],
                                      self.faces[:,(2,0)])).reshape(-1,2)
        self.edges.sort(axis=1)
    def generate_normals(self, fix_direction=False):
        '''
        If no normal information is loaded, we can get it from cross products
        Normal direction will be incorrect if mesh faces aren't ordered (right-hand rule)
        '''
        self.normal_face = np.zeros((len(self.faces),3))
        self.vertices = np.array(self.vertices)
        for index, face in enumerate(self.faces):
            v0 = (self.vertices[face[0]] - self.vertices[face[1]])
            v1 = (self.vertices[face[2]] - self.vertices[face[1]])
            self.normal_face[index] = np.cross(v0, v1)
        if fix_direction: self.fix_normals_direction()
            
    def fix_normals_direction(self):
        '''
        NONFUNCTIONAL
        Will eventually fix normals for a mesh. 
        '''
        visited_faces = np.zeros(len(self.faces))
        centroid = np.mean(self.vertices, axis=1)
        return None

    def transform(self, matrix):
        stacked = np.column_stack((self.vertices, np.ones(len(self.vertices))))
        self.vertices = np.dot(matrix, stacked.T)[:,0:3]

    def generate_bounds(self):
        self.bounds   = np.vstack((np.min(self.vertices, axis=0),
                                   np.max(self.vertices, axis=0)))
        self.centroid = np.mean(self.vertices, axis=0)
        self.box_size = np.diff(self.bounds, axis=0)[0]
        self.scale    = np.min(self.box_size)
  
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
    '''
    points = np.array(points)
    norms  = np.sum(points ** 2, axis=(len(points.shape)-1)) ** .5
    if (error_on_zero and
        np.any(norms < TOL_ZERO)):
        raise NameError('Unable to unitize zero length vector!')
    return (points.T/norms).T
    
def major_axis(points):
    '''
    Returns an approximate vector representing the major axis of points
    '''
    sq = np.dot(np.transpose(points), points)
    d, v = np.linalg.eig(sq)
    return v[np.argmax(d)]
        
def surface_normal(points):
    '''
    Returns a normal estimate:
    http://www.lsr.ei.tum.de/fileadmin/publications/KlasingAlthoff-ComparisonOfSurfaceNormalEstimationMethodsForRangeSensingApplications_ICRA09.pdf

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
        test_axis = np.cross(normal, np.roll([0,-1,0], i))
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
    mesh.generate_edges()
    intersections, valid  = plane_line_intersection(plane_origin, 
                                                    plane_normal, 
                                                    mesh.vertices[[mesh.edges.T]],
                                                    line_segments = True)
    log.debug('mesh_cross_section found %i intersections.', np.sum(valid))
    if return_planar: 
        planar = project_to_plane(intersections, 
                                  plane_origin, 
                                  plane_normal).reshape((-1,2,2))
        return planar
    else: 
        return intersections.reshape(-1,2,3)

def planar_outline(mesh, 
                   plane_origin = [0,0,0],
                   plane_normal = [0,0,1]):
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
    (n,2) vertices of a closed polygon representing the outline of the mesh
          projected onto the specified plane. 
    '''
    from raytracing import RayTracer
    mesh.merge_vertices()
    # only work with faces pointed towards us (AKA discard back-faces)
    visible_faces = mesh.faces[np.dot(plane_normal, 
                                      mesh.normal_face.T) > -TOL_ZERO]
                                      
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
    offset_max = 1e-3
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
    first_occur = dict()
    format_str  = '0.' + str(decimals) + 'f'
    unique      = np.ones(len(data), dtype=np.bool)
    for index, row in enumerate(data):
        hashable = row_to_string(row, format_str=format_str)
        if hashable in first_occur:
            unique[index]                     = False
            if not return_first:
                unique[first_occur[hashable]] = False
        else:
            first_occur[hashable] = index
    return unique

def row_to_string(row, format_str='0.6f'):
    result = ""
    for i in row:
        result += format(i, format_str)
    return result
    
def load_stl(file_obj):
    if detect_binary_file(file_obj): return load_stl_binary(file_obj)
    else:                            return load_stl_ascii(file_obj)
        
def load_stl_binary(file_obj):
    def read_face():
        normal_face[current[1]] = np.array(struct.unpack("<3f", file_obj.read(12)))
        for i in xrange(3):
            vertex = np.array(struct.unpack("<3f", file_obj.read(12)))               
            faces[current[1]][i] = current[0]
            vertices[current[0]] = vertex
            current[0] += 1
        #this field is occasionally used for color, but is usually just ignored.
        colors[current[1]] = int(struct.unpack("<h", file_obj.read(2))[0]) 
        current[1] += 1

    #get the file_obj header
    header = file_obj.read(80)
    #use the header information about number of triangles
    tri_count   = int(struct.unpack("@i", file_obj.read(4))[0])
    faces       = np.zeros((tri_count, 3),   dtype=np.int)  
    normal_face = np.zeros((tri_count, 3),   dtype=np.float) 
    colors      = np.zeros( tri_count,       dtype=np.int)
    vertices    = np.zeros((tri_count*3, 3), dtype=np.float) 
    #current vertex, face
    current   = [0,0]
    while True:
        try: read_face()
        except: break
    vertices = vertices[:current[0]]
    if current[1] <> tri_count: 
        raise NameError('Number of faces loaded is different than specified by header!')
    return Trimesh(vertices    = vertices, 
                   faces       = faces, 
                   normal_face = normal_face, 
                   color_face  = colors)

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
                   normal_face = np.array(normals,  dtype=np.float),
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
                   normal_vertex = np.array(normals,  dtype=float))
    mesh.generate_normals()
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
    if len(mesh.normals) == 0: mesh.generate_normals(fix_directions=True)
    with open(filename, 'wb') as file_object:
        #write a blank header
        file_object.write(struct.pack("<80x"))
        #write the number of faces
        file_object.write(struct.pack("@i", len(mesh.faces)))
        #write the faces
        for index in xrange(len(mesh.faces)):
            write_face(file_object, 
                       mesh.vertices[[mesh.faces[index]]], 
                       mesh.normals[index])
                       
if __name__ == '__main__':
    formatter = logging.Formatter("[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s", "%Y-%m-%d %H:%M:%S")
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    handler_stream.setLevel(logging.DEBUG)
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    log.addHandler(handler_stream)
    import matplotlib.pyplot as plt
    import time
    
    mesh = load_mesh('./models/octagonal_pocket.stl')
    mesh.merge_vertices()

    tic = [time.time()]
    contour = planar_outline(mesh)
    cross   = cross_section(mesh, plane_origin=mesh.centroid)

    for line in cross:
        plt.plot(*line.T)
    plt.show()
    for line in contour:
        plt.plot(*line.T)
    plt.show()
    
    
    
    
    
 