import numpy as np
import networkx as nx

import collections

from . import util
from . import grouping

from .constants import log, tol
from .geometry import faces_to_edges

try:
    from graph_tool import Graph as GTGraph
    from graph_tool.topology import label_components
    _has_gt = True
except ImportError:
    _has_gt = False
    log.warning('graph-tool unavailable, some operations will be much slower')

try:
    from scipy import sparse
except ImportError:
    log.warning('no scipy')


def face_adjacency(faces=None, mesh=None, return_edges=False):
    '''
    Returns an (n,2) list of face indices.
    Each pair of faces in the list shares an edge, making them adjacent.


    Parameters
    ----------
    faces:        (n, d) int, set of faces referencing vertices by index
    mesh:         Trimesh object, optional if passed will used cached edges
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

    if mesh is None:
        # first generate the list of edges for the current faces
        # also return the index for which face the edge is from
        edges, edges_face = faces_to_edges(faces, return_index=True)
        edges.sort(axis=1)
    else:
        # if passed a mesh, used the cached values for edges sorted
        edges = mesh.edges_sorted
        edges_face = mesh.edges_face

    # this will return the indices for duplicate edges
    # every edge appears twice in a well constructed mesh
    # so for every row in edge_idx, edges[edge_idx[*][0]] == edges[edge_idx[*][1]]
    # in this call to group rows, we discard edges which don't occur twice
    edge_groups = grouping.group_rows(edges, require_count=2)

    if len(edge_groups) == 0:
        log.error('No adjacent faces detected! Did you merge vertices?')

    # the pairs of all adjacent faces
    # so for every row in face_idx, self.faces[face_idx[*][0]] and
    # self.faces[face_idx[*][1]] will share an edge
    face_adjacency = edges_face[edge_groups]
    if return_edges:
        face_adjacency_edges = edges[edge_groups[:, 0]]
        return face_adjacency, face_adjacency_edges
    return face_adjacency


def shared_edges(faces_a, faces_b):
    '''
    Given two sets of faces, find the edges which are in both sets.

    Parameters
    ---------
    faces_a: (n,3) int, set of faces
    faces_b: (m,3) int, set of faces

    Returns
    ---------
    shared: (p, 2) int, set of edges
    '''
    e_a = np.sort(faces_to_edges(faces_a), axis=1)
    e_b = np.sort(faces_to_edges(faces_b), axis=1)
    shared = grouping.boolean_rows(e_a, e_b, operation=set.intersection)
    return shared


def connected_edges(G, nodes):
    '''
    Given graph G and list of nodes, return the list of edges that
    are connected to nodes
    '''
    nodes_in_G = collections.deque()
    for node in nodes:
        if not G.has_node(node):
            continue
        nodes_in_G.extend(nx.node_connected_component(G, node))
    edges = G.subgraph(nodes_in_G).edges()
    return edges


def facets(mesh, engine=None):
    '''
    Find the list of parallel adjacent faces.

    Parameters
    ---------
    mesh:  Trimesh
    engine: str, which graph engine to use ('scipy', 'networkx', 'graphtool')

    Returns
    ---------
    facets: list of groups of face indexes (mesh.faces) of parallel adjacent faces.
    '''

    # (n,2) list of adjacent face indices
    face_idx = mesh.face_adjacency

    # test adjacent faces for angle
    normal_pairs = mesh.face_normals[[face_idx]]
    normal_dot = (np.sum(normal_pairs[:, 0, :]
                         * normal_pairs[:, 1, :], axis=1) - 1)**2

    # if normals are actually equal, they are parallel with a high degree of
    # confidence
    parallel = normal_dot < tol.zero
    non_parallel = np.logical_not(parallel)

    # saying that two faces are *not* parallel is susceptible to error
    # so we add a radius check which computes the distance between face
    # centroids and divides it by the dot product of the normals
    # this means that small angles between big faces will have a large
    # radius which we can filter out easily.
    # if you don't do this, floating point error on tiny faces can push
    # the normals past a pure angle threshold even though the actual
    # deviation across the face is extremely small.
    center_sq = np.sum(np.diff(mesh.triangles_center[face_idx],
                               axis=1).reshape((-1, 3)) ** 2, axis=1)
    radius_sq = center_sq[non_parallel] / normal_dot[non_parallel]
    parallel[non_parallel] = radius_sq > tol.facet_rsq

    components = connected_components(face_idx[parallel],
                                      node_count=len(mesh.faces),
                                      min_len=2,
                                      engine=engine)
    return components


def split(mesh,
          only_watertight=True,
          adjacency=None,
          engine=None):
    '''
    Split a mesh into multiple meshes from face connectivity.

    If only_watertight is true, it will only return watertight meshes
    and will attempt single triangle/quad repairs.

    Parameters
    ----------
    mesh: Trimesh
    only_watertight: if True, only return watertight components
    adjacency: (n,2) list of face adjacency to override using the plain
               adjacency calculated automatically.
    engine: str, which engine to use. ('networkx', 'scipy', or 'graphtool')

    Returns
    ----------
    meshes: list of Trimesh objects
    '''

    if adjacency is None:
        adjacency = mesh.face_adjacency

    # if only watertight the shortest thing we can split has 3 triangles
    if only_watertight:
        min_len = 3
    else:
        min_len = 1

    components = connected_components(edges=adjacency,
                                      node_count=len(mesh.faces),
                                      min_len=min_len,
                                      engine=engine)
    meshes = mesh.submesh(components,
                          only_watertight=only_watertight)
    return meshes


def connected_components(edges,
                         node_count,
                         min_len=1,
                         engine=None):
    '''
    Find groups of connected nodes from an edge list.

    Parameters
    -----------
    edges:      (n,2) int, edges between nodes
    node_count: int, the largest node in the graph
    min_len:    int, minimum length of a component group to return
    engine:     str, which graph engine to use.
                ('networkx', 'scipy', or 'graphtool')
                If None, will automatically choose fastest available.

    Returns
    -----------
    components: (n,) sequence of lists, nodes which are connected
    '''
    def components_networkx():
        graph = nx.from_edgelist(edges)
        # make sure every face has a node, so single triangles
        # aren't discarded (as they aren't adjacent to anything)
        if min_len <= 1:
            graph.add_nodes_from(np.arange(node_count))
        iterable = nx.connected_components(graph)
        # newer versions of networkx return sets rather than lists
        components = np.array([list(i) for i in iterable if len(i) >= min_len])
        return components

    def components_graphtool():
        g = GTGraph()
        # make sure all the nodes are in the graph
        if min_len <= 1:
            g.add_vertex(node_count)
        g.add_edge_list(edges)
        component_labels = label_components(g, directed=False)[0].a
        components = grouping.group(component_labels, min_len=min_len)
        return components

    def components_csgraph():
        component_labels = connected_component_labels(edges,
                                                      node_count=node_count)
        components = grouping.group(component_labels, min_len=min_len)
        return components

    # check input edges
    edges = np.asanyarray(edges, dtype=np.int)
    if not (len(edges) == 0 or
            util.is_shape(edges, (-1, 2))):
        raise ValueError('edges must be (n,2)!')

    # graphtool is usually faster then scipy by ~10%, however on very
    # large or very small graphs graphtool outperforms scipy substantially
    # networkx is pure python and is usually 5-10x slower
    engines = collections.OrderedDict((('graphtool', components_graphtool),
                                       ('scipy',     components_csgraph),
                                       ('networkx',  components_networkx)))

    # if a graph engine has explictly been requested use it
    if engine in engines:
        return engines[engine]()

    # otherwise, go through our ordered list of graph engines
    # until we get to one that has actually been installed
    for function in engines.values():
        try:
            return function()
        # will be raised if the library didn't import correctly above
        except NameError:
            continue
    raise ImportError('No connected component engines available!')


def connected_component_labels(edges, node_count):
    '''
    Label graph nodes from an edge list, using scipy.sparse.csgraph

    Parameters
    ----------
    edges: (n, 2) int, edges of a graph
    node_count: int, the largest node in the graph. 

    Returns
    ---------
    labels: (node_count,) int, component labels for each node
    '''
    edges = np.asanyarray(edges, dtype=np.int)
    if not (len(edges) == 0 or
            util.is_shape(edges, (-1, 2))):
        raise ValueError('edges must be (n,2)!')

    matrix = sparse.coo_matrix((np.ones(len(edges), dtype=np.bool),
                                (edges[:, 0], edges[:, 1])),
                               dtype=np.bool,
                               shape=(node_count, node_count))
    body_count, labels = sparse.csgraph.connected_components(matrix,
                                                             directed=False)
    return labels


def smoothed(mesh, angle):
    '''
    Return a non- watertight version of the mesh which will
    render nicely with smooth shading.

    Parameters
    ---------
    mesh:  Trimesh object
    angle: float, angle in radians, adjacent faces which have normals
           below this angle will be smoothed.

    Returns
    ---------
    smooth: Trimesh object
    '''
    if len(mesh.face_adjacency) == 0:
        return mesh
    angle_ok = mesh.face_adjacency_angles <= angle
    adjacency = mesh.face_adjacency[angle_ok]
    components = connected_components(adjacency,
                                      min_len=1,
                                      node_count=len(mesh.faces))
    smooth = mesh.submesh(components,
                          only_watertight=False,
                          append=True)
    return smooth


def is_watertight(edges, edges_sorted=None):
    '''
    Parameters
    ---------
    edges: (n,2) int, set of vertex indices

    Returns
    ---------
    watertight: boolean, whether every edge is contained by two faces
    '''
    if edges_sorted is None:
        edges_sorted = np.sort(edges, axis=1)
    groups = grouping.group_rows(edges_sorted, require_count=2)
    watertight = (len(groups) * 2) == len(edges)

    opposing = edges[groups].reshape((-1, 4))[:, 1:3].T
    reversed = np.equal(*opposing).all()
    return watertight, reversed


def graph_to_svg(graph):
    '''
    Turn a networkx graph into an SVG string, using graphviz dot.

    Arguments
    ----------
    graph: networkx graph

    Returns
    ---------
    svg: string, pictoral layout in SVG format
    '''

    import tempfile
    import subprocess
    with tempfile.NamedTemporaryFile() as dot_file:
        nx.drawing.nx_agraph.write_dot(graph, dot_file.name)
        svg = subprocess.check_output(['dot', dot_file.name, '-Tsvg'])
    return svg
