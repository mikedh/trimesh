import networkx as nx
import numpy as np
from collections import deque
from copy import deepcopy

from .constants import *
from .grouping import group, group_rows, replace_references
from .geometry import faces_to_edges, unitize

try: 
    from graph_tool import Graph as GTGraph
    from graph_tool.topology import label_components
    _has_gt = True
except: 
    _has_gt = False
    log.warn('No graph-tool! Some operations will be much slower!')

def face_adjacency(faces):
    '''
    Returns an (n,2) list of face indices.
    Each pair of faces in the list shares an edge, making them adjacent.

    This is useful for lots of things, for example finding connected subgraphs:

    graph = nx.Graph()
    graph.add_edges_from(mesh.face_adjacency())
    groups = nx.connected_components(graph_connected.subgraph(interesting_faces))
    '''

    # first generate the list of edges for the current faces
    # also return the index for which face the edge is from
    edges, edge_face_index = faces_to_edges(faces, sort=True, return_index=True)
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
    adjacency = edge_face_index[edge_groups]
    return adjacency

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
    if _has_gt: return facets_gt(mesh)
    else:       return facets_nx(mesh)

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

def facets_group(mesh):
    '''
    Find facets by grouping normals then getting the adjacency subgraph.
    The other two methods for finding facets rely on looking at the angle between
    adjacent faces, and then if they are below TOL_ZERO, adding them to a graph
    of parallel faces. This method should be somewhat more robust.
    '''
    adjacency = nx.from_edgelist(mesh.face_adjacency())
    facets    = deque()
    for row_group in group_rows(mesh.face_normals):
        if len(row_group) < 2: continue
        facets.extend([i for i in nx.connected_components(adjacency.subgraph(row_group)) if len(i) > 1])
    return np.array(facets)

def facets_nx(mesh):
    '''
    Returns lists of facets of a mesh. 
    Facets are defined as groups of faces which are both adjacent and parallel
    
    facets returned reference indices in mesh.faces
    If return_area is True, both the list of facets and their area are returned. 
    '''
    face_idx       = mesh.face_adjacency()
    normal_pairs   = mesh.face_normals[[face_idx]]
    parallel       = np.abs(np.sum(normal_pairs[:,0,:] * normal_pairs[:,1,:], axis=1) - 1) < TOL_PLANAR
    graph_parallel = nx.from_edgelist(face_idx[parallel])
    facets         = list(nx.connected_components(graph_parallel))
    return facets
    
def facets_gt(mesh):
    '''
    Returns lists of facets of a mesh. 
    Facets are defined as groups of faces which are both adjacent and parallel
    
    facets returned reference indices in mesh.faces
    If return_area is True, both the list of facets and their area are returned. 
    '''
    face_idx       = mesh.face_adjacency()
    normal_pairs   = mesh.face_normals[[face_idx]]
    parallel       = np.abs(np.sum(normal_pairs[:,0,:] * normal_pairs[:,1,:], axis=1) - 1) < TOL_PLANAR
    graph_parallel = GTGraph()
    graph_parallel.add_edge_list(face_idx[parallel])

    connected  = label_components(graph_parallel, directed=False)[0].a
    facets_idx = group(connected, min_length=2)
    return facets_idx

def split_nx(mesh, check_watertight=True, only_count=False):
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
        new_meshes.append(mesh.__class__(vertices     = mesh.vertices[[unique]],
                                         faces        = faces,
                                         face_normals = mesh.face_normals[[connected_faces]]))
    face_adjacency = nx.from_edgelist(mesh.face_adjacency())
    new_meshes     = deque()
    components     = list(nx.connected_components(face_adjacency))
    if only_count: return len(components)

    for component in components: mesh_from_components(component)
    log.info('split mesh into %i components.',
             len(new_meshes))
    return list(new_meshes)

def split_gt(mesh, check_watertight=True, only_count=False):
    g = GTGraph()
    g.add_edge_list(mesh.face_adjacency())    
    component_labels = label_components(g, directed=False)[0].a
    if check_watertight: 
        degree = g.degree_property_map('total').a
    meshes     = deque()
    components = group(component_labels)
    if only_count: return len(components)

    for i, current in enumerate(components):
        fill_holes = False
        if check_watertight:
            degree_3 = degree[current] == 3
            degree_2 = degree[current] == 2
            if not degree_3.all():
                if np.logical_or(degree_3, degree_2).all():
                    fill_holes = True
                else: 
                    continue

        # these faces have the original vertex indices
        faces_original = mesh.faces[current]
        face_normals   = mesh.face_normals[current]
        # we find the unique vertex indices, so we can reindex from zero
        unique_vert    = np.unique(faces_original)
        vertices       = mesh.vertices[unique_vert]
        replacement    = np.zeros(unique_vert.max()+1, dtype=np.int)
        replacement[unique_vert] = np.arange(len(unique_vert))
        faces                    = replacement[faces_original]
        new_mesh = mesh.__class__(faces        = faces, 
                                  face_normals = face_normals, 
                                  vertices     = vertices)
        new_meta = deepcopy(mesh.metadata)
        if 'name' in new_meta:
            new_meta['name'] = new_meta['name'] + '_' + str(i)
        new_mesh.metadata.update(new_meta)
        if fill_holes: 
            try:              new_mesh.fill_holes(raise_watertight=True)
            except MeshError: continue
        meshes.append(new_mesh)
    return list(meshes)

def split(mesh, check_watertight=True, only_count=False):
    if _has_gt: return split_gt(mesh, check_watertight, only_count)
    else:       return split_nx(mesh, check_watertight, only_count)

def is_watertight_gt(mesh):
    g = GTGraph()
    g.add_edge_list(mesh.face_adjacency())    
    degree     = g.degree_property_map('total').a
    watertight = np.equal(degree, 3).all()
    return watertight

def is_watertight_nx(mesh):
    adjacency  = nx.from_edgelist(mesh.face_adjacency())
    watertight = np.equal(list(adjacency.degree().values()), 3).all()
    return watertight
    
def is_watertight(mesh):
    if _has_gt: return is_watertight_gt(mesh)
    else:       return is_watertight_nx(mesh)
    
def fix_normals(mesh):
    '''
    Find and fix problems with mesh.face_normals and mesh.faces winding direction.
    
    For face normals ensure that vectors are consistently pointed outwards,
    and that mesh.faces is wound in the correct direction for all connected components.
    '''
    mesh.generate_face_normals()
    # we create the face adjacency graph: 
    # every node in g is an index of mesh.faces
    # every edge in g represents two faces which are connected
    graph = nx.from_edgelist(mesh.face_adjacency())
    
    # we are going to traverse the graph using BFS, so we have to start
    # a traversal for every connected component
    for connected in nx.connected_components(graph):
        # we traverse every pair of faces in the graph
        # we modify mesh.faces and mesh.face_normals in place 
        for face_pair in nx.bfs_edges(graph, connected[0]):
            # for each pair of faces, we convert them into edges,
            # find the edge that both faces share, and then see if the edges
            # are reversed in order as you would expect in a well constructed mesh
            pair      = mesh.faces[[face_pair]]
            edges     = faces_to_edges(pair, sort=False)
            overlap   = group_rows(np.sort(edges,axis=1), require_count=2)
            edge_pair = edges[[overlap[0]]]
            reversed  = edge_pair[0][0] != edge_pair[1][0]
            if reversed: continue
            # if the edges aren't reversed, invert the order of one of the faces
            # and negate its normal vector
            mesh.faces[face_pair[1]] = mesh.faces[face_pair[1]][::-1]
            mesh.face_normals[face_pair[1]] *= (reversed*2) - 1
            
        # the normals of every connected face now all pointed in 
        # the same direction, but there is no guarantee that they aren't all
        # pointed in the wrong direction
        faces           = mesh.faces[[connected]]
        faces_x         = np.min(mesh.vertices[:,0][[faces]], axis=1)
        left_order      = np.argsort(faces_x)
        left_values     = faces_x[left_order]
        left_candidates = np.abs(left_values - left_values[0]) < TOL_ZERO
        backwards       = None
        
        # note that we have to find a face which ISN'T perpendicular to the x axis 
        # thus we go through all the candidate faces that are at the extreme left
        # until we find one that has a nonzero dot product with the x axis
        for leftmost in left_order[left_candidates]:                
            face_dot = np.dot([-1.0,0,0], mesh.face_normals[leftmost]) 
            if abs(face_dot) > TOL_ZERO: 
                backwards = face_dot < 0.0
                break
        if backwards: mesh.face_normals[[connected]] *= -1.0
        
        winding_tri  = connected[0]
        winding_test = np.diff(mesh.vertices[[mesh.faces[winding_tri]]], axis=0)
        winding_dir  = np.dot(unitize(np.cross(*winding_test)), mesh.face_normals[winding_tri])
        if winding_dir < 0: mesh.faces[[connected]] = np.fliplr(mesh.faces[[connected]])
            
    
    
