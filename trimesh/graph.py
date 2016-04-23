import numpy as np
import networkx as nx

from collections import deque

from .constants import log, tol, MeshError
from .grouping  import group, group_rows, boolean_rows
from .geometry  import faces_to_edges
from .points    import unitize
from .util      import diagonal_dot, is_sequence

try: from scipy.spatial import cKDTree as KDTree
except ImportError: log.warning('Scipy unavailable')

try: 
    from graph_tool import Graph as GTGraph
    from graph_tool.topology import label_components
    _has_gt = True
except: 
    _has_gt = False
    log.warning('No graph-tool! Some operations will be much slower!')

def face_adjacency(faces, return_edges=False):
    '''
    Returns an (n,2) list of face indices.
    Each pair of faces in the list shares an edge, making them adjacent.


    Arguments
    ----------
    faces: (n, d) int, set of faces referencing vertices by index
    return_edges: bool, return the edges shared by adjacent faces

    Returns
    ---------
    adjacency: (m,2) int, indexes of faces that are adjacent
    
    if return_edges: 
         edges: (m,2) int, indexes of vertices which make up the 
                 edges shared by the adjacent faces

    Example
    ----------
    This is useful for lots of things, such as finding connected components:

    graph = nx.Graph()
    graph.add_edges_from(mesh.face_adjacency)
    groups = nx.connected_components(graph_connected)
    '''

    # first generate the list of edges for the current faces
    # also return the index for which face the edge is from
    edges, edge_face_index = faces_to_edges(faces, return_index = True)
    edges.sort(axis=1)
    # this will return the indices for duplicate edges
    # every edge appears twice in a well constructed mesh
    # so for every row in edge_idx, edges[edge_idx[*][0]] == edges[edge_idx[*][1]]
    # in this call to group rows, we discard edges which don't occur twice
    edge_groups = group_rows(edges, require_count=2)

    if len(edge_groups) == 0:
        log.error('No adjacent faces detected! Did you merge vertices?')

    # the pairs of all adjacent faces
    # so for every row in face_idx, self.faces[face_idx[*][0]] and 
    # self.faces[face_idx[*][1]] will share an edge
    face_adjacency = edge_face_index[edge_groups]
    if return_edges:
        face_adjacency_edges = edges[edge_groups[:,0]]
        return face_adjacency, face_adjacency_edges
    return face_adjacency

def adjacency_angle(mesh, angle, direction=np.less, return_edges=False):
    '''
    Return the adjacent faces of a mesh only if the faces
    are at less than a specified angle.

    Arguments
    ----------
    mesh:         Trimesh object
    angle:        float, angle in radians by default faces at angles LARGER than 
                   this will be considered NOT adjacenct
    direction:    function, used to test face angle against angle kwarg
                   by default set to np.less
    return_edges: bool, return edges affiliated with adjacency or not

    Returns
    ----------
    adjacency: (n,2) int list of face indices in mesh
    if return_edges:
        edges: (n,2) int list of vertex indices in mesh (edges)
    '''

    # use the cached adjacency if possible (n,2)
    adjacency = mesh.face_adjacency
    # normal vectors for adjacent faces (n, 2, 3)
    normals = mesh.face_normals[adjacency]
    # dot products of normals (n)
    dots = diagonal_dot(normals[:,0], normals[:,1])
    # clip for floating point error
    dots = np.clip(dots, -1.0, 1.0)
    adj_ok = direction(np.abs(np.arccos(dots)), angle)
    # result is (m,2)
    new_adjacency = adjacency[adj_ok]
    if return_edges:
        edges = mesh.face_adjacency_edges[adj_ok]
        return new_adjacency, edges
    return new_adjacency

def shared_edges(faces_a, faces_b):
    '''
    Given two sets of faces, find the edges which are in both sets.

    Arguments
    ---------
    faces_a: (n,3) int, set of faces
    faces_b: (m,3) int, set of faces

    Returns
    ---------
    shared: (p, 2) int, set of edges
    '''
    e_a = np.sort(faces_to_edges(faces_a), axis=1)
    e_b = np.sort(faces_to_edges(faces_b), axis=1)
    shared = boolean_rows(e_a, e_b, operation=set.intersection)
    return shared

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
 
def facets(mesh):
    '''
    Find the list of parallel adjacent faces.
    
    Arguments
    ---------
    mesh:  Trimesh
    
    Returns
    ---------
    facets: list of groups of face indexes (in mesh.faces) of parallel 
            adjacent faces. 
    '''
    def facets_nx():
        graph_parallel = nx.from_edgelist(face_idx[parallel])
        facets_idx = np.array([list(i) for i in nx.connected_components(graph_parallel)])
        return facets_idx
        
    def facets_gt():
        graph_parallel = GTGraph()
        graph_parallel.add_edge_list(face_idx[parallel])
        connected  = label_components(graph_parallel, directed=False)[0].a
        facets_idx = group(connected, min_len=2)
        return facets_idx

    # (n,2) list of adjacent face indices
    face_idx    = mesh.face_adjacency

    # test adjacent faces for angle
    normal_pairs = mesh.face_normals[[face_idx]]
    normal_dot   = (np.sum(normal_pairs[:,0,:] * normal_pairs[:,1,:], axis=1) - 1)**2

    # if normals are actually equal, they are parallel with a high degree of confidence
    parallel     = normal_dot < tol.zero
    non_parallel = np.logical_not(parallel)

    # saying that two faces are *not* parallel is susceptible to error
    # so we add a radius check which computes the distance between face
    # centroids and divides it by the dot product of the normals
    # this means that small angles between big faces will have a large
    # radius which we can filter out easily.
    # if you don't do this, floating point error on tiny faces can push
    # the normals past a pure angle threshold even though the actual 
    # deviation across the face is extremely small. 
    center      = mesh.triangles.mean(axis=1)
    center_sq = np.sum(np.diff(center[face_idx], 
                               axis = 1).reshape((-1,3)) ** 2, axis=1)
    radius_sq = center_sq[non_parallel] / normal_dot[non_parallel]
    parallel[non_parallel] = radius_sq > tol.facet_rsq

    # graph-tool is ~6x faster than networkx but is more difficult to install
    if _has_gt: return facets_gt()
    else:       return facets_nx()

def split(mesh, only_watertight=True, adjacency=None):
    '''
    Given a mesh, will split it up into a list of meshes based on face connectivity
    If only_watertight is true, it will only return meshes where each face has
    exactly 3 adjacent faces.

    Arguments
    ----------
    mesh: Trimesh 
    only_watertight: if True, only return watertight components
    adjacency: (n,2) list of face adjacency to override using the plain
               adjacency calculated automatically. 

    Returns
    ----------
    meshes: list of Trimesh objects
    '''

    def split_nx():
        adjacency_graph = nx.from_edgelist(adjacency)
        components = nx.connected_components(adjacency_graph)
        result = mesh.submesh(components, only_watertight=only_watertight)
        return result

    def split_gt():
        g = GTGraph()
        g.add_edge_list(adjacency)
        component_labels = label_components(g, directed=False)[0].a
        components = group(component_labels)
        result = mesh.submesh(components, only_watertight=only_watertight)
        return result

    if adjacency is None:
        adjacency = mesh.face_adjacency
    
    if _has_gt: 
        return split_gt()
    else:       
        return split_nx()

def smoothed(mesh, angle):
    '''
    Return a non- watertight version of the mesh which will
    render nicely with smooth shading. 

    Arguments
    ---------
    mesh:  Trimesh object
    angle: float, angle in radians, adjacent faces which have normals
           below this angle will be smoothed.

    Returns
    ---------
    smooth: Trimesh object
    '''
    adjacency = adjacency_angle(mesh, angle)
    graph = nx.from_edgelist(adjacency)
    graph.add_nodes_from(np.arange(len(mesh.faces)))
    smooth = mesh.submesh(nx.connected_components(graph),
                          only_watertight = False,
                          append = True)
    return smooth

def is_watertight(edges, return_winding=False):
    '''
    Arguments
    ---------
    edges: (n,2) int, set of vertex indices
    
    Returns
    ---------
    watertight: boolean, whether every edge is contained by two faces
    '''
    edges_sorted = np.sort(edges, axis=1)
    groups = group_rows(edges_sorted, require_count=2)
    watertight = (len(groups) * 2) == len(edges)
    if return_winding:
        opposing = edges[groups].reshape((-1,4))[:,1:3].T
        reversed = np.equal(*opposing).all()
        return watertight, reversed
    return watertight
