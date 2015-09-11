import networkx as nx
import numpy as np

from collections import deque
from copy import deepcopy

from .constants import *
from .grouping  import group, group_rows, replace_references
from .geometry  import faces_to_edges
from .points    import unitize
from .util      import diagonal_dot, is_sequence

from scipy.spatial import cKDTree as KDTree

try: 
    from graph_tool import Graph as GTGraph
    from graph_tool.topology import label_components
    _has_gt = True
except: 
    _has_gt = False
    log.warning('No graph-tool! Some operations will be much slower!')

def face_adjacency(faces):
    '''
    Returns an (n,2) list of face indices.
    Each pair of faces in the list shares an edge, making them adjacent.

    This is useful for lots of things, for example finding connected subgraphs:

    graph = nx.Graph()
    graph.add_edges_from(mesh.face_adjacency)
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
    of parallel faces. This method is 'fuzzier'
    '''
    adjacency = nx.from_edgelist(mesh.face_adjacency)
    facets    = deque()
    for row_group in group_rows(mesh.face_normals):
        if len(row_group) < 2: continue
        facets.extend([list(i) for i in nx.connected_components(adjacency.subgraph(row_group)) if len(i) > 1])
    return np.array(facets)

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
        facets_idx     = [list(i) for i in nx.connected_components(graph_parallel)]
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
    parallel     = normal_dot < TOL_ZERO
    non_parallel = np.logical_not(parallel)

    # saying that two faces *arent* parallel is susceptible to error
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
    parallel[non_parallel] = radius_sq > TOL_FACET_RSQ

    # graph-tool is ~6x faster than networkx but is more difficult to install
    if _has_gt: return facets_gt()
    else:       return facets_nx()

def split(mesh, check_watertight=True, only_count=False):
    '''
    Given a mesh, will split it up into a list of meshes based on face connectivity
    If check_watertight is true, it will only return meshes where each face has
    exactly 3 adjacent faces.

    Arguments
    ----------
    mesh: Trimesh 
    check_watertight: if True, only return watertight components
    only_count:       if True, return number of components rather
                      than the components

    Returns
    ----------
    if only_count: 
        count: int, number of components
    else:
        meshes: list of Trimesh objects
    '''

    def split_nx():
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
        face_adjacency = nx.from_edgelist(mesh.face_adjacency)
        new_meshes     = deque()
        components     = [list(i) for i in nx.connected_components(face_adjacency)]
        if only_count: return len(components)

        for component in components: mesh_from_components(component)
        log.info('split mesh into %i components.',
                 len(new_meshes))
        return list(new_meshes)

    def split_gt():
        g = GTGraph()
        g.add_edge_list(mesh.face_adjacency)    
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

    if _has_gt: return split_gt()
    else:       return split_nx()
    
def is_watertight(mesh):
    def is_watertight_gt():
        g = GTGraph()
        g.add_edge_list(mesh.face_adjacency)    
        degree     = g.degree_property_map('total').a
        watertight = np.equal(degree, 3).all()
        return watertight
    def is_watertight_nx():
        adjacency  = nx.from_edgelist(mesh.face_adjacency)
        watertight = np.equal(list(adjacency.degree().values()), 3).all()
        return watertight

    if _has_gt: return is_watertight_gt()
    else:       return is_watertight_nx()

def fix_winding(mesh):
    '''
    Traverse and change mesh faces in-place to make sure winding is coherent, 
    or that edges on adjacent faces are in opposite directions
    '''
    # we create the face adjacency graph: 
    # every node in g is an index of mesh.faces
    # every edge in g represents two faces which are connected
    graph_all = nx.from_edgelist(mesh.face_adjacency)
    
    # we are going to traverse the graph using BFS, so we have to start
    # a traversal for every connected component
    for graph in nx.connected_component_subgraphs(graph_all):
        start = graph.nodes()[0]
        # we traverse every pair of faces in the graph
        # we modify mesh.faces and mesh.face_normals in place 
        for face_pair in nx.bfs_edges(graph, start):
            # for each pair of faces, we convert them into edges,
            # find the edge that both faces share, and then see if the edges
            # are reversed in order as you would expect in a well constructed mesh
            pair    = mesh.faces[[face_pair]]            
            edges   = faces_to_edges(pair, sort=False)
            overlap = group_rows(np.sort(edges,axis=1), require_count=2)
            if len(overlap) == 0:
                # only happens on non-watertight meshes
                continue
            edge_pair = edges[[overlap[0]]]
            if edge_pair[0][0] == edge_pair[1][0]:
                # if the edges aren't reversed, invert the order of one of the faces
                mesh.faces[face_pair[1]] = mesh.faces[face_pair[1]][::-1]

def fix_direction(mesh):
    '''
    Check to see if a mesh has normals pointed outside the solid using ray tests.

    If the mesh is not watertight, this is meaningless.
    '''
    # force face normals to be regenerated according to the right-hand rule
    mesh.face_normals = None
    # which direction should our test rays go
    direction = -mesh.face_normals[0]
    # origin of test ray
    origin    = mesh.vertices[mesh.faces[0]].mean(axis=0)
    origin   -= direction * mesh.box_size.max() * 2
    rays = [[origin, direction]]
    location, hit_id = mesh.ray.intersects_location(rays, return_id=True)
    # the distance along the ray vector the hit happened at
    projection = np.dot(location - origin, direction)
    # the face index of the farthest face along the vector
    face_outer = hit_id[0][projection.argmax()]
    # the normal of this presumed outer face
    normal_outer = mesh.face_normals[face_outer]
    # dot product with our direction to see if it is pointing inside or 
    # outside
    normal_sign  = np.sign(np.dot(normal_outer, direction))

    if normal_sign < 0:
        # reverse the face normals
        mesh.face_normals *= normal_sign
        # since normals were regenerated, this means winding is backwards
        # if winding is incoherent this won't fix anything
        mesh.faces = np.fliplr(mesh.faces)

def fix_normals(mesh):
    '''
    Fix the winding and direction of a mesh face and face normals in-place

    Really only meaningful on watertight meshes, but will orient all 
    faces and winding in a uniform way for non-watertight face patches as well.
    '''
    fix_winding(mesh)
    fix_direction(mesh)

def broken_faces(mesh, color=None):
    '''
    Return the index of faces in the mesh which break the watertight status
    of the mesh. If color is set, change the color of the broken faces. 
    '''
    adjacency = nx.from_edgelist(mesh.face_adjacency)
    broken    = [k for k, v in adjacency.degree().iteritems() if v != 3]
    broken    = np.array(broken)
    if not (color is None):
        if (not is_sequence(color)):
            color = [255,0,0]
        mesh.visual.face_colors[broken] = color
    return broken
