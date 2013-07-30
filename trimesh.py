'''
trimesh.py

Library for importing and doing simple operations on triangular meshes
Styled after transformations.py
'''

import numpy as np
import time, struct

class Trimesh():
    def __init__(self, filename=None):
        self.vertices  = []
        self.faces     = []
        self.edges     = []
        self.colors    = []
        self.normals   = []
        if filename <> None: self.load(filename)

    def cross_section(self, plane_origin=[0,0,0], plane_normal=[0,0,1], TOL=1e-7, return_planar=True):
        '''
        VERY ALPHA, not working particularly well
        Return a cross section of the trimesh based on plane origin and normal. 
        Basically a bunch of plane-line intersection queries with validation checks.
        Depends on properly ordered edge information, as done by generate_edges

        origin:        (3) array of plane origin
        normal:        (3) array for plane normal
        TOL:           float, cutoff tolerance for 'in plane'
        return_planar: bool, if True returns (m, 2) planar crossection, False returns (m, 3)

        returns: point pairs of cross section
        '''
        if len(self.faces) == 0: raise NameError("Cannot compute cross section of empty mesh.")
        if len(self.edges) == 0: self.generate_edges()
        
        d0 = (np.dot(self.vertices[[self.edges[:,0]]] - plane_origin, plane_normal))
        d1 = (np.dot(self.vertices[[self.edges[:,1]]] - plane_origin, plane_normal))

        #edge passes through plane
        hits = np.not_equal(np.sign(d0),
                            np.sign(d1))
 
        #edge endpoint(s) lie on the cross section plane:
        d0_on_plane = np.abs(d0) < TOL
        d1_on_plane = np.abs(d1) < TOL
        on_plane    = np.logical_or(d0_on_plane,
                                    d1_on_plane)

        edge_on_plane = np.logical_and(d0_on_plane,
                                       d1_on_plane)

        #an endpoint on (within TOL) the section plane is the same as a hit
        hits = np.logical_and(np.logical_or(hits, on_plane),
                              np.logical_not(edge_on_plane))

        #degenerate case is all three edges are on plane, due to 
        # triangle having zero area on plane, or is 'on edge' AKA:
        # check: 2 or more edges on plane
        sum_hits      = np.sum(hits.reshape((-1,3)), axis=1)
        sum_on        = np.sum(edge_on_plane.reshape((-1,3)), axis=1)

        not_on_edge   = np.tile(np.less(sum_on, 2), 3)
        proper_hits   = np.tile(np.equal(sum_hits, 2), 3)
        
        intersectors = np.logical_and(hits, not_on_edge)
        
        p0 = self.vertices[[self.edges[intersectors][:,0]]]
        p1 = self.vertices[[self.edges[intersectors][:,1]]]
        
        #this is a set of unmerged point pairs: 
        #[A,B,C,D,E,A], where lines are AB CD EA
        intersections = plane_line_intersection(plane_origin, plane_normal, p0, p1)
        if not return_planar: return intersections

        planar = points_to_plane(intersections, plane_origin, plane_normal)
        return planar

    def convex_hull(self, merge_radius=1e-3):
        '''
        Get a new Trimesh object representing the convex hull of the 
        current mesh. Requires scipy >.12, and doesn't produce nice normals

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

    def merge_vertices(self, tolerance=1e-7):
        '''
        Merges vertices which are identical and replaces references
        Does this by creating a KDTree.
        cKDTree requires scipy >= .12 for this query type and you 
        probably don't want to use plain python KDTree as it is crazy slow (~1000x in my tests)

        tolerance: to what precision do vertices need to be identical
        '''
        from scipy.spatial import cKDTree as KDTree

        tree    = KDTree(self.vertices)
        used    = np.zeros(len(self.vertices), dtype=np.bool)
        unique  = []
        replacement_dict = dict()

        for index, vertex in enumerate(self.vertices):
            if used[index]: continue
            neighbors = tree.query_ball_point(self.vertices[index], tolerance)
            used[[neighbors]] = True
            replacement_dict.update(np.column_stack((neighbors,
                                                     [len(unique)]*len(neighbors))))
            unique.append(index)
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
                                      self.faces[:,(2,0)])).reshape((-1,2))
        
    def generate_normals(self, fix_direction=False):
        '''
        If no normal information is loaded, we can get it from cross products
        Normal direction will be incorrect if mesh faces aren't ordered (right-hand rule)
        '''
        self.normals = np.zeros((len(self.faces),3))
        self.vertices = np.array(self.vertices)
        for index, face in enumerate(self.faces):
            v0 = (self.vertices[face[0]] - self.vertices[face[1]])
            v1 = (self.vertices[face[2]] - self.vertices[face[1]])
            self.normals[index] = np.cross(v0, v1)
        if fix_direction: self.fix_normals_direction()
            
    def fix_normals_direction(self):
        '''
        NONFUNCTIONAL
        Will eventually fix normals for a mesh. 
        '''
        visited_faces = np.zeros(len(self.faces))
        centroid = np.mean(self.vertices, axis=1)
        return None

    def transform(self, transformation_matrix):
        stacked = np.column_stack((self.vertices, np.ones(len(self.vertices))))
        self.vertices = np.dot(transformation_matrix, stacked.T)[:,0:3]

    def bounding_box(self):
        box = np.vstack((np.min(self.vertices, axis=0),
                         np.max(self.vertices, axis=0)))
        return box

    def export(self, filename):
        '''
        Export current mesh object to binary STL file. 
        '''
        export_stl(self, filename)

class STLMesh(Trimesh):
    def load(self, filename):
        if detect_binary_file(filename): self.load_binary(filename)
        else: raise NameError('We haven\'t implemented ASCII STL...')
    def read_face_binary(self, f):
        '''
        Read individual triangle from binary STL file
        '''
        self.normals[self.current_face] = np.array(struct.unpack("<3f", f.read(12)))
        for i in xrange(3):
             vertex = np.array(struct.unpack("<3f", f.read(12)))               
             self.faces[self.current_face][i]   = self.current_vertex
             self.vertices[self.current_vertex] = vertex
             self.current_vertex += 1
        b = struct.unpack("<h", f.read(2))
        self.current_face += 1
    def load_binary(self, filename):
        with open (filename, "rb") as f:
            #get the file header
            header = f.read(80)
            #use the header information about number of triangles
            tri_count = int(struct.unpack("@i", f.read(4))[0])
            self.faces    = np.zeros((tri_count, 3), dtype=np.int)  
            self.normals  = np.zeros((tri_count, 3), dtype=np.float) 
            self.colors   = np.zeros(tri_count, dtype=np.int)
            
            #length of vertices eventually depends on the topology, but 
            #in STL faces are all separate triangles with nothing merged or referenced
            self.vertices = np.zeros((tri_count*3, 3), dtype=np.float) 
            self.current_face   = 0
            self.current_vertex = 0
            while True:
                try: self.read_face_binary(f)
                except: break
                    
            self.vertices = self.vertices[:self.current_vertex]
            if self.current_face <> tri_count: 
                raise NameError('Number of faces loaded is different than specified by header!') 

def load(filename):
    '''
    Load a mesh. At the moment only binary STL is implemented.
    '''
    mesh_loaders = {'stl': STLMesh}

    extension = (str(filename).split('.')[-1]).lower()
    if extension in mesh_loaders:
        return mesh_loaders[extension](filename)
    else: raise NameError('Unable to load mesh from file of type .' + extension)

def export_stl(mesh, filename):
    '''
    Saves a mesh object as a binary STL file. 
    '''
    def write_face(file_object, vertices, normal):
        '''
        write individual triangle to binary STL file
        vertices = (3,3) array of floats
        normal (3) array of floats
        '''
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

def replace_references(data, reference_dict, return_array=False):
    '''
    Replace elements in an array as per a dictionary of replacement values

    data:           numpy array
    reference_dict: dictionary of replacements. example:
                       {2:1, 3:1, 4:5}
    return_array: if false, replaces references in place and returns nothing
    '''
    dv = data.view().reshape((-1))
    for i in xrange(len(dv)):
        if dv[i] in reference_dict:
            dv[i] = reference_dict[dv[i]]
    if return_array: return dv

def detect_binary_file(filename):
    '''
    Returns True if file has non-ascii charecters
    http://stackoverflow.com/questions/898669/how-can-i-detect-if-a-file-is-binary-non-text-in-python
    
    '''
    textchars = ''.join(map(chr, [7,8,9,10,12,13,27] + range(0x20, 0x100)))
    fbytes = open(filename).read(1024)
    return bool(fbytes.translate(None, textchars))

def plane_line_intersection(plane_ori, plane_dir, pt0, pt1):
    '''
    Calculates plane-line intersections

    plane_ori: plane origin, (3) list
    plane_dir: plane direction (3) list
    pt0: first list of line segment endpoints (n,3)
    pt1: second list of line segment endpoints (n,3)
    '''
    line_dir  = unitize(pt1 - pt0)
    plane_dir = unitize(plane_dir)
    t = np.dot(plane_dir, np.transpose(plane_ori - pt0))
    b = np.dot(plane_dir, np.transpose(line_dir))
    d = t / b
    return pt0 + np.reshape(d,(np.shape(line_dir)[0],1))*line_dir

def unitize(points):
    '''
    One liner which will unitize vectors by row
    axis arg to sum is so one vector (3,) gets vectorized correctly 
    as well as 10 vectors (10,3)

    points: numpy array/list of points to be unit vector'd
    '''
    points = np.array(points)
    return (points.T/np.sum(points ** 2, 
                            axis=(len(points.shape)-1)) ** .5 ).T
    
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

def radial_sort(points, origin=None, normal=None):
    '''
    Sorts a set of points radially (by angle) around an origin/normal

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
               
def points_to_plane(points, origin=[0,0,0], normal=[0,0,1]):
    '''
    projects a set of (n,3) points onto a plane, returning (n,2) points
    '''
    axis0 = [normal[2], normal[0], normal[1]]
    axis1 = np.cross(normal, axis0)
    pt_vec = np.array(points) - origin
    pr0 = np.dot(pt_vec, axis0)
    pr1 = np.dot(pt_vec, axis1)
    return np.column_stack((pr0, pr1))
    
def mesh_to_plane(mesh, plane_normal= [0,0,1], TOL=1e-8):
    '''
    INCOMPLETE
    Orthographic projection of a mesh to a plane
    
    input
    mesh: trimesh object
    plane_normal: plane normal (3) list
    TOL: comparison tolerance

    output:
    list of non-overlapping but possibly adjacent polygons
    '''
    planar       = points_to_plane(mesh.vertices, plane_normal)
    face_visible = np.zeros(len(mesh.faces), dtype=np.bool)
    
    for index, face in enumerate(mesh.faces):
        dot = np.dot(mesh.normals[index], plane_normal)
        '''
        dot product between face normal and plane normal:
        greater than zero: back faces
        zero: polygon viewed on edge (zero thickness)
        less than zero: front facing
        '''
        if (dot < -TOL):
            face_visible[index] = True
    return planar[[face_visible]]

def unique_rows(data, return_index=True):
    '''
    Returns unique rows of an array, using string hashes. 
    '''
    first_occur  = dict()
    unique_index = []
    for index, row in enumerate(data):
        hashable = row_to_string(row)
        if hashable in first_occur:
            continue
        first_occur[hashable] = index
        unique_index.append(index)
    if return_index: return unique_index
    else: return np.array(data)[[unique_index]]
    
def row_to_string(row, format_str="0.7f"):
    result = ""
    for i in row:
        result += format(i, format_str)
    return result

if __name__ == '__main__':
    import os
    import transformations
    m = load(os.path.join('./models', 'round.stl'))
    p = m.cross_section(plane_origin=[0,0,.1])

    tr = transformations.rotation_matrix(np.radians(34), [1,0,0])
    m.transform(tr)

    import matplotlib.pyplot as plt
    for i in xrange(0,len(p)/2,2):
        plt.plot([p[i][0], p[i+1][0]],
                 [p[i][1], p[i+1][1]])
    plt.show()
    

