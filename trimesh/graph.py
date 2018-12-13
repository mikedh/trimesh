"""
graph.py
-------------

Deal with graph operations. Primarily deal with graphs in (n,2)
edge list form, and abstract the backend graph library being used.

Currently uses networkx, scipy.sparse.csgraph, or graph_tool
backends.
"""

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

try:
    from scipy.sparse import csgraph, coo_matrix
except ImportError:
    log.warning('no scipy')


def face_adjacency(faces=None,
                   mesh=None,
                   return_edges=False):
    """
    Returns an (n,2) list of face indices.
    Each pair of faces in the list shares an edge, making them adjacent.


    Parameters
    ----------
    faces : (n, 3) int, or None
        List of vertex indices representing triangles
    mesh : Trimesh object
        If passed will used cached edges instead of faces
    return_edges : bool
        Return the edges shared by adjacent faces

    Returns
    ---------
    adjacency : (m,2) int
        Indexes of faces that are adjacent
    edges: (m,2) int
        Only returned if return_edges is True
        Indexes of vertices which make up the
        edges shared by the adjacent faces

    Examples
    ----------
    This is useful for lots of things such as finding
    face- connected components:
    >>> graph = nx.Graph()
    >>> graph.add_edges_from(mesh.face_adjacency)
    >>> groups = nx.connected_components(graph_connected)
    """

    if mesh is None:
        # first generate the list of edges for the current faces
        # also return the index for which face the edge is from
        edges, edges_face = faces_to_edges(faces,
                                           return_index=True)
        # make sure edge rows are sorted
        edges.sort(axis=1)
    else:
        # if passed a mesh, used the cached values
        edges = mesh.edges_sorted
        edges_face = mesh.edges_face

    # this will return the indices for duplicate edges
    # every edge appears twice in a well constructed mesh
    # so for every row in edge_idx:
    # edges[edge_idx[*][0]] == edges[edge_idx[*][1]]
    # in this call to group rows we discard edges which
    # don't occur twice
    edge_groups = grouping.group_rows(edges, require_count=2)

    if len(edge_groups) == 0:
        log.error('No adjacent faces detected! Did you merge vertices?')

    # the pairs of all adjacent faces
    # so for every row in face_idx, self.faces[face_idx[*][0]] and
    # self.faces[face_idx[*][1]] will share an edge
    face_adjacency = edges_face[edge_groups]

    # sort pairs so we can search for indexes with ordered pairs
    face_adjacency.sort(axis=1)

    if return_edges:
        face_adjacency_edges = edges[edge_groups[:, 0]]
        return face_adjacency, face_adjacency_edges
    return face_adjacency


def face_adjacency_unshared(mesh):
    """
    Return the vertex index of the two vertices not in the shared
    edge between two adjacent faces

    Parameters
    ----------
    mesh : Trimesh object

    Returns
    -----------
    vid_unshared : (len(mesh.face_adjacency), 2) int
        Indexes of mesh.vertices
    """

    # the non- shared vertex index is the same shape as face_adjacnecy
    # just holding vertex indices rather than face indices
    vid_unshared = np.zeros_like(mesh.face_adjacency,
                                 dtype=np.int64)
    # loop through both columns of face adjacency
    for i, adjacency in enumerate(mesh.face_adjacency.T):
        # faces from the current column of face adjacency
        faces = mesh.faces[adjacency]
        shared = np.logical_or(
            faces == mesh.face_adjacency_edges[:, 0].reshape((-1, 1)),
            faces == mesh.face_adjacency_edges[:, 1].reshape((-1, 1)))
        vid_unshared[:, i] = faces[np.logical_not(shared)]
    return vid_unshared


def face_adjacency_radius(mesh):
    """
    Compute an approximate radius between adjacent faces.

    Parameters
    --------------
    mesh : trimesh.Trimesh

    Returns
    -------------
    radii : (len(self.face_adjacency),) float
        Approximate radius between faces
        Parallel faces will have a value of np.inf
    span :  (len(self.face_adjacency),) float
        Perpendicular projection distance of two
        unshared vertices onto the shared edge
    """

    # solve for the radius of the adjacent faces
    #         distance
    # R = ------------------
    #     2 * sin(theta / 2)
    nonzero = mesh.face_adjacency_angles > np.radians(.01)
    denominator = np.abs(
        2.0 * np.sin(mesh.face_adjacency_angles[nonzero] / 1.0))

    # consider the distance between the non- shared vertices of the
    # face adjacency pair as the key distance
    point_pairs = mesh.vertices[mesh.face_adjacency_unshared]
    vectors = np.diff(point_pairs,
                      axis=1).reshape((-1, 3))

    # the vertex indices of the shared edge for the adjacency pairx
    edges = mesh.face_adjacency_edges
    # unit vector along shared the edge
    edges_vec = util.unitize(np.diff(mesh.vertices[edges],
                                     axis=1).reshape((-1, 3)))

    # the vector of the perpendicular projection to the shared edge
    perp = np.subtract(
        vectors, (util.diagonal_dot(
            vectors, edges_vec).reshape(
            (-1, 1)) * edges_vec))
    # the length of the perpendicular projection
    span = np.linalg.norm(perp, axis=1)

    # complete the values for non- infinite radii
    radii = np.ones(len(mesh.face_adjacency)) * np.inf
    radii[nonzero] = span[nonzero] / denominator

    return radii, span


def vertex_adjacency_graph(mesh):
    """
    Returns a networkx graph representing the vertices and
    their connections in the mesh.

    Parameters
    ----------
    mesh : Trimesh object

    Returns
    ---------
    graph : networkx.Graph
        Graph representing vertices and edges between
        them where vertices are nodes and edges are edges

    Examples
    ----------
    This is useful for getting nearby vertices for a given vertex,
    potentially for some simple smoothing techniques.
    >>> graph = mesh.vertex_adjacency_graph
    >>> graph.neighbors(0)
    > [1,3,4]
    """
    g = nx.Graph()
    g.add_edges_from(mesh.edges_unique)
    return g


def shared_edges(faces_a, faces_b):
    """
    Given two sets of faces, find the edges which are in both sets.

    Parameters
    ---------
    faces_a: (n,3) int, set of faces
    faces_b: (m,3) int, set of faces

    Returns
    ---------
    shared: (p, 2) int, set of edges
    """
    e_a = np.sort(faces_to_edges(faces_a), axis=1)
    e_b = np.sort(faces_to_edges(faces_b), axis=1)
    shared = grouping.boolean_rows(e_a, e_b, operation=np.intersect1d)
    return shared


def connected_edges(G, nodes):
    """
    Given graph G and list of nodes, return the list of edges that
    are connected to nodes

    """
    nodes_in_G = collections.deque()
    for node in nodes:
        if not G.has_node(node):
            continue
        nodes_in_G.extend(nx.node_connected_component(G, node))
    edges = G.subgraph(nodes_in_G).edges()
    return edges


def facets(mesh, engine=None):
    """
    Find the list of parallel adjacent faces.

    Parameters
    ---------
    mesh :  trimesh.Trimesh
    engine : str
       Which graph engine to use:
       ('scipy', 'networkx', 'graphtool')

    Returns
    ---------
    facets : sequence of (n,) int
        Groups of face indexes of
        parallel adjacent faces.
    """
    # what is the radius of a circle that passes through the perpendicular
    # projection of the vector between the two non- shared vertices
    # onto the shared edge, with the face normal from the two adjacent faces
    radii = mesh.face_adjacency_radius
    # what is the span perpendicular to the shared edge
    span = mesh.face_adjacency_span
    # a very arbitrary formula for declaring two adjacent faces
    # parallel in a way that is hopefully (and anecdotally) robust
    # to numeric error
    # a common failure mode is two faces that are very narrow with a slight
    # angle between them, so here we divide by the perpendicular span
    # to penalize very narrow faces, and then square it just for fun
    parallel = np.ones(len(radii), dtype=np.bool)
    # if span is zero we know faces are small/parallel
    nonzero = np.abs(span) > tol.zero
    # faces with a radii/span ratio larger than a threshold pass
    parallel[nonzero] = (radii[nonzero] /
                         span[nonzero]) ** 2 > tol.facet_threshold

    # run connected components on the parallel faces to group them
    components = connected_components(mesh.face_adjacency[parallel],
                                      nodes=np.arange(len(mesh.faces)),
                                      min_len=2,
                                      engine=engine)
    return components


def split(mesh,
          only_watertight=True,
          adjacency=None,
          engine=None):
    """
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
    """

    if adjacency is None:
        adjacency = mesh.face_adjacency

    # if only watertight the shortest thing we can split has 3 triangles
    if only_watertight:
        min_len = 3
    else:
        min_len = 1

    components = connected_components(edges=adjacency,
                                      nodes=np.arange(len(mesh.faces)),
                                      min_len=min_len,
                                      engine=engine)
    meshes = mesh.submesh(components,
                          only_watertight=only_watertight)
    return meshes


def connected_components(edges,
                         min_len=1,
                         nodes=None,
                         engine=None):
    """
    Find groups of connected nodes from an edge list.

    Parameters
    -----------
    edges:      (n,2) int, edges between nodes
    nodes:      (m, ) int, list of nodes that exist
    min_len:    int, minimum length of a component group to return
    engine:     str, which graph engine to use.
                ('networkx', 'scipy', or 'graphtool')
                If None, will automatically choose fastest available.

    Returns
    -----------
    components: (n,) sequence of lists, nodes which are connected
    """
    def components_networkx():
        """
        Find connected components using networkx
        """
        graph = nx.from_edgelist(edges)
        # make sure every face has a node, so single triangles
        # aren't discarded (as they aren't adjacent to anything)
        if min_len <= 1:
            graph.add_nodes_from(nodes)
        iterable = nx.connected_components(graph)
        # newer versions of networkx return sets rather than lists
        components = np.array([np.array(list(i), dtype=np.int64)
                               for i in iterable if len(i) >= min_len])
        return components

    def components_graphtool():
        """
        Find connected components using graphtool
        """
        g = GTGraph()
        # make sure all the nodes are in the graph
        g.add_vertex(node_count)
        # add the edge list
        g.add_edge_list(edges)

        labels = np.array(label_components(g, directed=False)[0].a,
                          dtype=np.int64)[:node_count]

        # we have to remove results that contain nodes outside
        # of the specified node set and reindex
        contained = np.zeros(node_count, dtype=np.bool)
        contained[nodes] = True
        index = np.arange(node_count, dtype=np.int64)[contained]

        components = grouping.group(labels[contained], min_len=min_len)
        components = np.array([index[c] for c in components])

        return components

    def components_csgraph():
        """
        Find connected components using scipy.sparse.csgraph
        """
        # label each node
        labels = connected_component_labels(edges,
                                            node_count=node_count)

        # we have to remove results that contain nodes outside
        # of the specified node set and reindex
        contained = np.zeros(node_count, dtype=np.bool)
        contained[nodes] = True
        index = np.arange(node_count, dtype=np.int64)[contained]

        components = grouping.group(labels[contained], min_len=min_len)
        components = np.array([index[c] for c in components])

        return components

    # check input edges
    edges = np.asanyarray(edges, dtype=np.int64)
    # if no nodes were specified just use unique
    if nodes is None:
        nodes = np.unique(edges)

    # exit early if we have no nodes
    if len(nodes) == 0:
        return np.array([])
    elif len(edges) == 0:
        if min_len <= 1:
            return np.reshape(nodes, (-1, 1))
        else:
            return np.array([])

    if not util.is_shape(edges, (-1, 2)):
        raise ValueError('edges must be (n,2)!')

    # find the maximum index referenced in either nodes or edges
    counts = [0]
    if len(edges) > 0:
        counts.append(edges.max())
    if len(nodes) > 0:
        counts.append(nodes.max())
    node_count = np.max(counts) + 1

    # remove edges that don't have both nodes in the node set
    mask = np.zeros(node_count, dtype=np.bool)
    mask[nodes] = True
    edges_ok = mask[edges].all(axis=1)
    edges = edges[edges_ok]

    # graphtool is usually faster then scipy by ~10%, however on very
    # large or very small graphs graphtool outperforms scipy substantially
    # networkx is pure python and is usually 5-10x slower
    engines = collections.OrderedDict((('graphtool', components_graphtool),
                                       ('scipy', components_csgraph),
                                       ('networkx', components_networkx)))

    # if a graph engine has explicitly been requested use it
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


def connected_component_labels(edges, node_count=None):
    """
    Label graph nodes from an edge list, using scipy.sparse.csgraph

    Parameters
    ----------
    edges : (n, 2) int
       Edges of a graph
    node_count : int, or None
        The largest node in the graph.

    Returns
    ---------
    labels : (node_count,) int
        Component labels for each node
    """
    matrix = edges_to_coo(edges, node_count)
    body_count, labels = csgraph.connected_components(
        matrix, directed=False)

    assert len(labels) == node_count

    return labels


def split_traversal(traversal,
                    edges,
                    edges_hash=None):
    """
    Given a traversal as a list of nodes, split the traversal
    if a sequential index pair is not in the given edges.

    Parameters
    --------------
    edges : (n, 2) int
       Graph edge indexes
    traversal : (m,) int
       Traversal through edges
    edge_hash : (n,)
       Edges sorted on axis=1 and
       passed to grouping.hashable_rows

    Returns
    ---------------
    split : sequence of (p,) int
    """
    traversal = np.asanyarray(traversal,
                              dtype=np.int64)

    # hash edge rows for contains checks
    if edges_hash is None:
        edges_hash = grouping.hashable_rows(
            np.sort(edges, axis=1))

    # turn the (n,) traversal into (n-1,2) edges
    trav_edge = np.column_stack((traversal[:-1],
                                 traversal[1:]))
    # hash each edge so we can compare to edge set
    trav_hash = grouping.hashable_rows(
        np.sort(trav_edge, axis=1))
    # check if each edge is contained in edge set
    contained = np.in1d(trav_hash, edges_hash)

    # exit early if every edge of traversal exists
    if contained.all():
        # just reshape one traversal
        split = [traversal]
    else:
        # find contiguous groups of contained edges
        blocks = grouping.blocks(contained,
                                 min_len=1,
                                 only_nonzero=True)

        # turn edges back in to sequence of traversals
        split = [np.append(trav_edge[b][:, 0],
                           trav_edge[b[-1]][1])
                 for b in blocks]

    # close traversals if necessary
    for i, t in enumerate(split):
        # make sure elements of sequence are numpy arrays
        split[i] = np.asanyarray(split[i], dtype=np.int64)
        # don't close if its a single edge
        if len(t) <= 2:
            continue
        # make sure it's not already closed
        edge = np.sort([t[0], t[-1]])
        if edge.ptp() == 0:
            continue
        close = grouping.hashable_rows(edge.reshape((1, 2)))[0]
        # if we need the edge add it
        if close in edges_hash:
            split[i] = np.append(t, t[0]).astype(np.int64)
    result = np.array(split)

    return result


def fill_traversals(traversals, edges, edges_hash=None):
    """
    Convert a traversal of a list of edges into a sequence of
    traversals where every pair of consecutive node indexes
    is an edge in a passed edge list

    Parameters
    -------------
    traversals : sequence of (m,) int
       Node indexes of traversals of a graph
    edges : (n, 2) int
       Pairs of connected node indexes
    edges_hash : None, or (n,) int
       Edges sorted along axis 1 then hashed
       using grouping.hashable_rows

    Returns
    --------------
    splits : sequence of (p,) int
       Node indexes of connected traversals
    """
    # make sure edges are correct type
    edges = np.asanyarray(edges, dtype=np.int64)
    # make sure edges are sorted
    edges.sort(axis=1)

    # if there are no traversals just return edges
    if len(traversals) == 0:
        return edges.copy()

    # hash edges for contains checks
    if edges_hash is None:
        edges_hash = grouping.hashable_rows(edges)

    splits = []
    for nodes in traversals:
        # split traversals to remove edges
        # that don't actually exist
        splits.extend(split_traversal(
            traversal=nodes,
            edges=edges,
            edges_hash=edges_hash))
    # turn the split traversals back into (n,2) edges
    included = util.vstack_empty([np.column_stack((i[:-1], i[1:]))
                                  for i in splits])
    if len(included) > 0:
        # sort included edges in place
        included.sort(axis=1)
        # make sure any edges not included in split traversals
        # are just added as a length 2 traversal
        splits.extend(grouping.boolean_rows(
            edges,
            included,
            operation=np.setdiff1d))
    else:
        # no edges were included, so our filled traversal
        # is just the original edges copied over
        splits = edges.copy()

    return splits


def traversals(edges, mode='bfs'):
    """
    Given an edge list, generate a sequence of ordered
    depth first search traversals, using scipy.csgraph routines.

    Parameters
    ------------
    edges : (n,2) int, undirected edges of a graph
    mode :  str, 'bfs', or 'dfs'

    Returns
    -----------
    traversals: (m,) sequence of (p,) int,
                ordered DFS or BFS traversals of the graph.
    """
    edges = np.asanyarray(edges, dtype=np.int64)
    if len(edges) == 0:
        return []
    elif not util.is_shape(edges, (-1, 2)):
        raise ValueError('edges are not (n,2)!')

    # pick the traversal method
    mode = str(mode).lower().strip()
    if mode == 'bfs':
        func = csgraph.breadth_first_order
    elif mode == 'dfs':
        func = csgraph.depth_first_order
    else:
        raise ValueError('traversal mode must be either dfs or bfs')

    # make sure edges are sorted so we can query
    # an ordered pair later
    edges.sort(axis=1)
    # set of nodes to make sure we get every node
    nodes = set(edges.reshape(-1))
    # coo_matrix for csgraph routines
    graph = edges_to_coo(edges)

    # we're going to make a sequence of traversals
    traversals = []

    while len(nodes) > 0:
        # starting at any node
        start = nodes.pop()
        # get an (n,) ordered traversal
        ordered = func(graph,
                       i_start=start,
                       return_predecessors=False,
                       directed=False).astype(np.int64)

        traversals.append(ordered)
        # remove the nodes we've consumed
        nodes.difference_update(ordered)

    return traversals


def edges_to_coo(edges, count=None, data=None):
    """
    Given an edge list, return a boolean scipy.sparse.coo_matrix
    representing the edges in matrix form.

    Parameters
    ------------
    edges : (n,2) int
      Edges of a graph
    count : int
      The total number of nodes in the graph
      if None: count = edges.max() + 1
    data : (n,) any
      Assign data to each edge, if None will
      be bool True for each specified edge

    Returns
    ------------
    matrix: (count, count) scipy.sparse.coo_matrix
      Sparse COO
    """
    edges = np.asanyarray(edges, dtype=np.int64)
    if not (len(edges) == 0 or
            util.is_shape(edges, (-1, 2))):
        raise ValueError('edges must be (n,2)!')

    # if count isn't specified just set it to largest
    # value referenced in edges
    if count is None:
        count = edges.max() + 1
    count = int(count)

    # if no data is specified set every specified edge
    # to True
    if data is None:
        data = np.ones(len(edges), dtype=np.bool)

    matrix = coo_matrix((data, edges.T),
                        dtype=data.dtype,
                        shape=(count, count))
    return matrix


def smoothed(mesh, angle):
    """
    Return a non- watertight version of the mesh which will
    render nicely with smooth shading by disconnecting faces
    at sharp angles to each other.

    Parameters
    ---------
    mesh :  trimesh.Trimesh
      Source geometry
    angle : float
      Angle in radians, adjacent faces which have normals
      below this angle will be smoothed

    Returns
    ---------
    smooth : trimesh.Trimesh
      Geometry with disconnected face patches
    """
    # if the mesh has no adjacent faces return a copy
    if len(mesh.face_adjacency) == 0:
        return mesh.copy()

    # face pairs below angle threshold
    angle_ok = mesh.face_adjacency_angles <= angle
    # subset of face adjacency
    adjacency = mesh.face_adjacency[angle_ok]
    # list of connected groups of faces
    components = connected_components(adjacency,
                                      min_len=1,
                                      nodes=np.arange(len(mesh.faces)))
    # get a submesh as a single appended Trimesh
    smooth = mesh.submesh(components,
                          only_watertight=False,
                          append=True)
    return smooth


def is_watertight(edges, edges_sorted=None):
    """
    Parameters
    ---------
    edges : (n, 2) int
      List of vertex indices
    edges_sorted : (n, 2) int
      Pass vertex indices sorted on axis 1 as a speedup

    Returns
    ---------
    watertight : boolean
      Whether every edge is shared by an even
      number of faces
    winding : boolean
      Whether every shared edge is reversed
    """
    # passing edges_sorted is a speedup only
    if edges_sorted is None:
        edges_sorted = np.sort(edges, axis=1)

    # group sorted edges
    groups = grouping.group_rows(edges_sorted, require_count=2)
    watertight = bool((len(groups) * 2) == len(edges))

    # are opposing edges reversed
    opposing = edges[groups].reshape((-1, 4))[:, 1:3].T
    # wrap the weird numpy bool
    winding = bool(np.equal(*opposing).all())

    return watertight, winding


def graph_to_svg(graph):
    """
    Turn a networkx graph into an SVG string, using graphviz dot.

    Parameters
    ----------
    graph: networkx graph

    Returns
    ---------
    svg: string, pictoral layout in SVG format
    """

    import tempfile
    import subprocess
    with tempfile.NamedTemporaryFile() as dot_file:
        nx.drawing.nx_agraph.write_dot(graph, dot_file.name)
        svg = subprocess.check_output(['dot', dot_file.name, '-Tsvg'])
    return svg


def multigraph_paths(G, source, cutoff=None):
    """
    For a networkx MultiDiGraph, find all paths from a source node
    to leaf nodes. This function returns edge instance numbers
    in addition to nodes, unlike networkx.all_simple_paths.

    Parameters
    ---------------
    G: networkx.MultiDiGraph
    source: str, node to start off
    cutoff: int, number of nodes to visit
                 if None, will

    Returns
    ----------
    traversals: (n,) list of [(node, edge instance index), ] paths
    """
    if cutoff is None:
        cutoff = (len(G.edges()) * len(G.nodes())) + 1

    # the path starts at the node specified
    current = [(source, 0)]
    # traversals we need to go back and do
    queue = []
    # completed paths
    traversals = []

    for i in range(cutoff):
        # paths are stored as (node, instance) so
        # get the node of the last place visited
        current_node = current[-1][0]
        # get all the children of the current node
        child = G[current_node]

        if len(child) == 0:
            # we have no children, so we are at the end of this path
            # save the path as a completed traversal
            traversals.append(current)
            # if there is nothing on the queue, we are done
            if len(queue) == 0:
                break
            # otherwise continue traversing with the next path
            # on the queue
            current = queue.pop()
        else:
            # oh no, we have multiple edges from current -> child
            start = True
            # iterate through child nodes and edge instances
            for node in child.keys():
                for instance in child[node].keys():
                    if start:
                        # if this is the first edge, keep it on the
                        # current traversal and save the others for later
                        current.append((node, instance))
                        start = False
                    else:
                        # this child has multiple instances
                        # so we will need to traverse them multiple times
                        # we appended a node to current, so only take the
                        # first n-1 visits
                        queue.append(current[:-1] + [(node, instance)])
    return traversals


def multigraph_collect(G, traversal, attrib=None):
    """
    Given a MultiDiGraph traversal, collect attributes along it.

    Parameters
    -------------
    G:          networkx.MultiDiGraph
    traversal:  (n) list of (node, instance) tuples
    attrib:     dict key, name to collect. If None, will return all

    Returns
    -------------
    collected: (len(traversal) - 1) list of attributes
    """

    collected = []
    for u, v in util.pairwise(traversal):
        attribs = G[u[0]][v[0]][v[1]]
        if attrib is None:
            collected.append(attribs)
        else:
            collected.append(attribs[attrib])
    return collected
