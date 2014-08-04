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
        
from time import time as time_function
from string import Template
from copy import deepcopy
import networkx as nx
import os
import inspect

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

MODULE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

TOL_ZERO  = 1e-12
TOL_MERGE = 1e-9

def log_time(method):
    def timed(*args, **kwargs):
        tic    = time_function()
        result = method(*args, **kwargs)
        toc    = time_function()
        log.debug('%s executed in %.4f seconds.',
                  method.__name__,
                  toc-tic)
        return result
    return timed
    
@log_time
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

    mesh = mesh_loaders[type](file_obj)
    file_obj.close()

    log.info('loaded mesh from %s container, with %i faces and %i vertices.', 
             type, len(mesh.faces), len(mesh.vertices))

    return mesh

class Trimesh():
    def __init__(self, 
                 vertices        = None, 
                 faces           = None, 
                 face_normals    = None, 
                 vertex_normals  = None,
                 color_faces     = None,
                 color_vertices  = None):

        self.vertices        = np.array(vertices)
        self.faces           = np.array(faces)
        self.face_normals    = np.array(face_normals)
        self.vertex_normals  = np.array(vertex_normals)
        self.color_faces     = np.array(color_faces)
        self.color_vertices  = np.array(color_vertices)

        self.generate_bounds()
    
    def process(self):
        self.merge_vertices_hash()
        self.remove_duplicate_faces()
        self.remove_degenerate_faces()
        self.remove_unreferenced_vertices()
        return self

    @log_time
    def split(self):
        '''
        Returns a list of Trimesh objects, based on connectivity.
        Splits into individual components, sometimes referred to as 'bodies'
        '''
        return split_by_face_connectivity(self)
        
    @log_time
    def area(self):
        '''
        Returns the summed area of all triangles in the current mesh.
        '''
        return triangles_area(self.vertices[[self.faces]])
        
    @log_time
    def face_adjacency(self):
        '''
        Returns an (n,2) list of face indices.
        Each pair of faces in the list shares an edge, making them adjacent.
        
        This is particularly useful for finding connected subgraphs, eg:
        
        graph = nx.Graph()
        graph.add_edges_from(mesh.face_adjacency())
        groups = nx.connected_components(graph_connected.subgraph(interesting_faces))
        '''
        
        # first generate the list of edges for the current faces
        edges = faces_to_edges(self.faces, sort=True)
        # this will return the indices for duplicate edges
        # every edge appears twice in a well constructed mesh
        # so for every row in edge_idx, self.edges[edge_idx[*][0]] == self.edges[edge_idx[*][1]]
        # in this call to group rows, we discard edges which don't occur twice
        edge_idx = group_rows(edges, require_count=2, digits=0)

        if len(edge_idx) == 0:
            log.warn('No adjacent faces detected! Did you merge vertices?')
        # returns the pairs of all adjacent faces
        # so for every row in face_idx, self.faces[face_idx[*][0]] and self.faces[face_idx[*][1]]
        # will share an edge
        face_idx = np.tile(np.arange(len(self.faces)), (3,1)).T.reshape(-1)[[edge_idx]]
        return face_idx
        
    @log_time
    def is_watertight(self):
        '''
        Check if a mesh is watertight. 
        This currently only checks to see if every face has a degree of 3.
        '''
        g = nx.from_edgelist(self.face_adjacency())
        watertight = np.equal(g.degree().values(), 3).all()
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

    @log_time
    def facets(self):
        return mesh_facets(self)

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
        g = nx.from_edgelist(self.face_adjacency())
        
        # we are going to traverse the graph using BFS, so we have to start
        # a traversal for every connected component
        for connected in nx.connected_components(g):
            # we traverse every pair of faces in the graph
            # we modify self.faces and self.face_normals in place 
            for face_pair in nx.bfs_edges(g, connected[0]):
                # for each pair of faces, we convert them into edges,
                # find the edge that both faces share, and then see if the edges
                # are reversed in order as you would expect in a well constructed mesh
                pair      = self.faces[[face_pair]]
                edges     = faces_to_edges(pair, sort=False)
                overlap   = group_rows(np.sort(edges,axis=1), require_count=2)
                edge_pair = edges[[overlap[0]]]
                reversed  = edge_pair[0][0] <> edge_pair[1][0]
                if reversed: continue
                # if the edges aren't reversed, invert the order of one of the faces
                # and negate its normal vector
                self.faces[face_pair[1]] = self.faces[face_pair[1]][::-1]
                self.face_normals[face_pair[1]] *= (reversed*2) - 1

                self.vertices[[self.faces[face_pair[1]]]]
            # the normals of every connected face now all pointed in 
            # the same direction, but there is no guarantee that they aren't all
            # pointed in the wrong direction
            faces     = self.faces[[connected]]
            leftmost  = np.argmin(np.min(self.vertices[:,0][[faces]], axis=1))
            backwards = np.dot([-1.0,0,0], self.face_normals[leftmost]) < 0.0
            if backwards: self.face_normals[[connected]] *= -1.0
            
            winding_tri  = connected[0]
            winding_test = np.diff(self.vertices[[self.faces[winding_tri]]], axis=0)
            winding_dir  = np.dot(unitize(np.cross(*winding_test)), self.face_normals[winding_tri])
            if winding_dir < 0: self.faces[[connected]] = np.fliplr(self.faces[[connected]])
            
    def verify_normals(self):
        '''
        Check to make sure both face and vertex normals are defined. 
        '''
        if ((self.face_normals == None) or
            (np.shape(self.face_normals) <> np.shape(self.faces))):
            log.debug('Generating face normals for faces %s and passed face normals %s',
                      str(np.shape(self.faces)),
                      str(np.shape(self.face_normals)))
            self.generate_face_normals()

        if ((self.vertex_normals == None) or 
            (np.shape(self.vertex_normals) <> np.shape(self.vertices))): 
            self.generate_vertex_normals()

    def cross_section(self,
                      plane_normal,
                      plane_origin  = None,
                      return_planar = True):
        '''
        Returns a cross section of the current mesh and plane defined by
        plane_origin and plane_normal.
        
        If return_planar is True,  result is (n, 2, 2) 
        If return_planar is False, result is (n, 2, 3)
        '''
        if plane_origin == None: plane_origin = self.centroid
        return cross_section(self, plane_normal, plane_origin, return_planar)
       
    @log_time   
    def convex_hull(self, inflate=None):
        '''
        Get a new Trimesh object representing the convex hull of the 
        current mesh. Requires scipy >.12.
        
        Inflate is a convenience parameter which will take the convex hull of the mesh,
        and then inflate, or 'pad' the hull. 

        '''
        from scipy.spatial import ConvexHull
        faces = ConvexHull(self.vertices).simplices
        mesh  = Trimesh(vertices = self.vertices, faces = faces)
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        
        if inflate <> None:
            mesh.generate_vertex_normals()
            mesh.vertices += inflate*mesh.vertex_normals
            return mesh.convex_hull(inflate=None)
            
        return mesh

    def merge_vertices(self, angle_max=None):
        if angle_max <> None: self.merge_vertices_kdtree(angle_max)
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

        tic         = time_function()
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
        if angle_max <> None: self.generate_vertex_normals()
       
        log.debug('merge_vertices_kdtree reduced vertex count from %i to %i in %.4fs.',
                  len(used),
                  len(unique),
                  time_function()-tic)
                  
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
    def remove_unreferenced_vertices(self):
        '''
        Removes all vertices which aren't in a face
        Reindexes vertices from zero and replaces face references
        '''
        referenced = self.faces.view().reshape(-1)
        unique_ref = np.unique(referenced).astype(int)
        replacement = dict()
        replacement.update(np.column_stack((unique_ref,
                                            range(len(unique_ref)))))                            
        self.faces    = replace_references(self.faces, replacement)
        self.vertices = self.vertices[[unique_ref]]

    @log_time
    def generate_edges(self):
        '''
        Populate self.edges from face information.
        '''
        self.edges = faces_to_edges(self.faces, sort=True)
        
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

    def transform(self, matrix):
        '''
        Transform mesh vertices by matrix
        '''
        self.vertices = transform_points(self.vertices, matrix)

    def generate_bounds(self):
        '''
        Calculate bounding box and rough centroid of mesh. 
        '''
        
        shape = np.shape(self.vertices)
        if ((len(shape) <> 2) or
            (not (shape[1] in [2,3]))):
            return

        self.bounds   = np.vstack((np.min(self.vertices, axis=0),
                                   np.max(self.vertices, axis=0)))
        self.centroid = np.mean(self.vertices, axis=0)
        self.box_size = np.diff(self.bounds, axis=0)[0]
        self.scale    = np.min(self.box_size)
  
    def mass_properties(self):
        #http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf
        pass

    def show(self, smooth = False):
        '''
        Render the mesh in an opengl window. Requires pyglet.
        Smooth will re-merge vertices to fix the shading, but can be slow
        on larger meshes. 
        '''
        from mesh_render import MeshRender
        MeshRender(self, smooth=smooth)

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
    nondegenerate = np.all(np.diff(np.sort(faces, axis=1), axis=1) <> 0, axis=1)
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
    w         = points - plane_ori
    distances = np.abs(np.dot(plane_dir, w.T) / np.linalg.norm(plane_dir))
    return distances

def unitize(points, error_on_zero=False):
    '''
    Unitize vectors by row
    one vector (3,) gets vectorized correctly, 
    as well as 10 row vectors (10,3)

    points: numpy array/list of points to be unit vector'd
    error_on_zero: if zero magnitude vectors exist, throw an error
    '''
    points  = np.array(points)
    axis    = len(points.shape)-1
    norms   = np.sum(points ** 2, axis=axis) ** .5
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
               
def project_to_plane(points, normal=[0,0,1], origin=[0,0,0], return_transform=False):
    '''
    projects a set of (n,3) points onto a plane, returning (n,2) points
    '''

    if np.all(np.abs(normal) < TOL_ZERO):
        raise NameError('Normal must be nonzero!')

    log.debug('Projecting points to plane normal %s', str(normal))
  
    T        = align_vectors(normal, [0,0,1])
    T[0:3,3] = np.array(origin) 
    xy       = transform_points(points, T)[:,0:2]
    
    if return_transform: return xy, T
    return xy
    
   
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
    tic = time_function()
    mesh.generate_edges()
    intersections, valid  = plane_line_intersection(plane_origin, 
                                                    plane_normal, 
                                                    mesh.vertices[[mesh.edges.T]],
                                                    line_segments = True)
    toc = time_function()
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
            edge_normal = unitize(np.cross(ray_dir, edge_vector)[0])
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

def hash_rows(data, digits=None):
    '''
    We turn our array into integers, based on the precision 
    given by digits, and then put them in a hashable format. 
    '''
    
    if digits == None:
        digits = abs(int(np.log10(TOL_MERGE)))
        
    as_int = ((data+10**-(digits+1))*10**digits).astype(np.int64)    
    hashes = np.ascontiguousarray(as_int).view(np.dtype((np.void, 
                                               as_int.dtype.itemsize * as_int.shape[1])))
    return hashes
    
def unique_rows(data, return_inverse=False, digits=None):
    '''
    Returns indices of unique rows. It will return the 
    first occurrence of a row that is duplicated:
    [[1,2], [3,4], [1,2]] will return [0,1]
    '''
    hashes                   = hash_rows(data, digits=digits)
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
    
    require_count only returns groups of a specified length, eg:
    require_count =  2
    [[1,2], [3,4], [1,2]] will return [[0,2]]
    '''
    hashes = hash_rows(data, digits=digits)
    
    observed = dict()
    count_ok = dict()
    for index, hashable in enumerate(hashes):
        hashed = hashable.tostring()
        if hashed in observed: observed[hashed].append(index)
        else:                  observed[hashed] = [index]
        
        if not require_count: continue
        if len(observed[hashed]) == require_count: count_ok[hashed] = observed[hashed]
        elif not hashed in count_ok:               continue
        else:                                      del count_ok[hashed]
    if require_count: 
        if require_count == 1: return np.reshape(count_ok.values(), -1)
        return np.array(count_ok.values())
    return np.array(observed.values())

def stack_negative(rows):
    rows     = np.array(rows)
    width    = rows.shape[1]
    stacked  = np.column_stack((rows, rows*-1))
    negative = rows[:,0] < 0
    stacked[negative] = np.roll(stacked[negative], 3, axis=1)
    return stacked

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
            aligned_neg = tree.query_ball_point(-1*vector, TOL_END)
            aligned     = np.append(aligned, aligned_neg)                              
        aligned = aligned.astype(int)
        consumed[[aligned]] = True
        consensus_vector = vectors[aligned[-1]]
        unique_vectors.append(consensus_vector)
        aligned_index.append(aligned)
    return np.array(unique_vectors), np.array(aligned_index)
    
def load_stl(file_obj):
    if detect_binary_file(file_obj): return load_stl_binary(file_obj)
    else:                            return load_stl_ascii(file_obj)
        
def load_stl_binary(file_obj):
    '''
    Load a binary STL file into a trimesh object. 
    Uses a single main struct.unpack call, and is significantly faster
    than looping methods or ASCII STL. 
    '''
    #get the file_obj header
    header = file_obj.read(80)
    
    #use the header information about number of triangles
    tri_count    = int(struct.unpack("@i", file_obj.read(4))[0])
    faces        = np.arange(tri_count*3).reshape((-1,3))
    
    # this blob extracts 12 float values, with 2 pad bytes per face
    # the first three floats are the face normal
    # the next 9 are the three vertices 
    blob = np.array(struct.unpack("<" + "12fxx"*tri_count, 
                                  file_obj.read(50*tri_count))).reshape((-1,4,3))

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

def load_assimp(filename):
    def LPMesh_to_Trimesh(lp):
        return Trimesh(vertices       = lp.vertices,
                       vertex_normals = lp.normals,
                       faces          = lp.faces)
    import pyassimp
    scene  = pyassimp.load(filename)
    meshes = map(LPMesh_to_Trimesh, scene.meshes)
    pyassimp.release(scene)
    if len(meshes) == 1: return meshes[0]
    return meshes

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

def mesh_facets(mesh):
    face_idx       = mesh.face_adjacency()
    normal_pairs   = mesh.face_normals[[face_idx]]
    pair_axis      = np.cross(normal_pairs[:,0,:], normal_pairs[:,1,:]) 
    pair_norms     = np.sum(pair_axis ** 2, axis=(1)) ** .5
    parallel       = pair_norms < TOL_ZERO
    graph_parallel = nx.from_edgelist(face_idx[parallel])
    facets         = nx.connected_components(graph_parallel)
    facets_area    = [triangles_area(mesh.vertices[[mesh.faces[facet]]]) for facet in facets]
    return facets, facets_area

def split_by_face_connectivity(mesh, check_watertight=True):
    def mesh_from_components(connected_faces):
        if check_watertight:
            subgraph   = nx.subgraph(face_adjacency, connected_faces)
            watertight = np.equal(subgraph.degree().values(), 3).all()
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
    map(mesh_from_components, nx.connected_components(face_adjacency))
    log.info('split mesh into %i components.',
             len(new_meshes))
    return list(new_meshes)
    
def transform_points(points, matrix):
    stacked     = np.column_stack((points, np.ones(len(points))))
    transformed = np.dot(matrix, stacked.T).T[:,0:3]
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
    norm         = np.clip(np.linalg.norm(cross), -1, 1)
    if norm < TOL_ZERO: return np.eye(4)
    angle        = np.arcsin(norm) 
    T            = tr.rotation_matrix(angle, cross)
    return T

def triangles_cross(triangles):
    '''
    Returns the cross product of two edges from input triangles 

    triangles: (n,3,3)
    returns: 
    '''

    vectors = np.diff(triangles, axis=1)
    crosses = np.cross(vectors[:,0], vectors[:,1])
    return crosses
    
def triangles_area(triangles):
    '''
    Calculates the sum area of input triangles 

    triangles: vertices of triangles (n,3,3)
    returns:   area (float)
    '''
    crosses = triangles_cross(triangles)
    area    = np.sum(np.sum(crosses**2, axis=1)**.5)*.5
    return area
    
def triangles_normal(triangles):
    '''
    Calculates the normals of input triangles 
    '''
    crosses = triangles_cross(triangles)
    return unitize(crosses)

if __name__ == '__main__':
    formatter = logging.Formatter("[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s", "%Y-%m-%d %H:%M:%S")
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    handler_stream.setLevel(logging.DEBUG)
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    log.addHandler(handler_stream)
    np.set_printoptions(precision=4, suppress=True)
    
    m = load_mesh('models/octagonal_pocket.stl')
    m.process()
 
