"""
graph.py
-------------

Deal with graph operations. Primarily deal with graphs in (n, 2)
edge list form, and abstract the backend graph library being used.

Currently uses networkx or scipy.sparse.csgraph backend.
"""

import collections

import numpy as np

from . import exceptions, grouping, triangles, util
from .constants import log, tol
from .geometry import faces_to_edges
from .typed import (
    ArrayLike,
    GraphEngineType,
    Integer,
    NDArray,
    Number,
    Sequence,
    int64,
)

try:
    from scipy.sparse import coo_matrix, csgraph
    from scipy.spatial import cKDTree
except BaseException as E:
    # re-raise exception when used
    cKDTree = exceptions.ExceptionWrapper(E)
    csgraph = exceptions.ExceptionWrapper(E)
    coo_matrix = exceptions.ExceptionWrapper(E)

try:
    import networkx as nx
except BaseException as E:
    # create a dummy module which will raise the ImportError
    # or other exception only when someone tries to use networkx
    nx = exceptions.ExceptionWrapper(E)


def face_adjacency(faces=None, mesh=None, return_edges=False):
    """
    Returns an (n, 2) list of face indices.
    Each pair of faces in the list shares an edge, making them adjacent.


    Parameters
    -----------
    faces : (n, 3) int, or None
        Vertex indices representing triangles
    mesh : Trimesh object
        If passed will used cached edges
        instead of generating from faces
    return_edges : bool
        Return the edges shared by adjacent faces

    Returns
    ----------
    adjacency : (m, 2) int
        Indexes of faces that are adjacent
    edges: (m, 2) int
        Only returned if return_edges is True
        Indexes of vertices which make up the
        edges shared by the adjacent faces

    Examples
    ----------
    This is useful for lots of things such as finding
    face- connected components:
    ```python
    >>> graph = nx.Graph()
    >>> graph.add_edges_from(mesh.face_adjacency)
    >>> groups = nx.connected_components(graph_connected)
    ```
    """

    if mesh is None:
        # first generate the list of edges for the current faces
        # also return the index for which face the edge is from
        edges, edges_face = faces_to_edges(faces, return_index=True)
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
        log.debug("No adjacent faces detected! Did you merge vertices?")

    # the pairs of all adjacent faces
    # so for every row in face_idx, self.faces[face_idx[*][0]] and
    # self.faces[face_idx[*][1]] will share an edge
    adjacency = edges_face[edge_groups]

    # degenerate faces may appear in adjacency as the same value
    nondegenerate = adjacency[:, 0] != adjacency[:, 1]
    adjacency = adjacency[nondegenerate]

    # sort pairs in-place so we can search for indexes with ordered pairs
    adjacency.sort(axis=1)

    if return_edges:
        adjacency_edges = edges[edge_groups[:, 0][nondegenerate]]
        assert len(adjacency_edges) == len(adjacency)
        return adjacency, adjacency_edges

    return adjacency


def face_neighborhood(mesh) -> NDArray[int64]:
    """
    Find faces that share a vertex i.e. 'neighbors' faces.
    Relies on the fact that an adjacency matrix at a power p
    contains the number of paths of length p connecting two nodes.
    Here we take the bipartite graph from mesh.faces_sparse to the power 2.
    The non-zeros are the faces connected by one vertex.

    Returns
    ----------
    neighborhood : (n, 2) int
        Pairs of faces which share a vertex
    """

    VT = mesh.faces_sparse
    TT = VT.T * VT
    TT.setdiag(0)
    TT.eliminate_zeros()
    TT = TT.tocoo()
    neighborhood = np.concatenate(
        (TT.row[:, None], TT.col[:, None]), axis=-1, dtype=np.int64
    )
    return neighborhood


def face_adjacency_unshared(mesh):
    """
    Return the vertex index of the two vertices not in the shared
    edge between two adjacent faces

    Parameters
    ----------
    mesh : Trimesh object
      Input mesh

    Returns
    -----------
    vid_unshared : (len(mesh.face_adjacency), 2) int
      Indexes of mesh.vertices
      for degenerate faces without exactly
      one unshared vertex per face it will be -1
    """

    # the non- shared vertex index is the same shape
    # as face_adjacency holding vertex indices vs face indices
    vid_unshared = np.zeros_like(mesh.face_adjacency, dtype=np.int64) - 1
    # get the shared edges between adjacent faces
    edges = mesh.face_adjacency_edges

    # loop through the two columns of face adjacency
    for i, fid in enumerate(mesh.face_adjacency.T):
        # faces from the current column of face adjacency
        faces = mesh.faces[fid]
        # should have one True per row of (3,)
        # index of vertex not included in shared edge
        unshared = np.logical_not(
            np.logical_or(
                faces == edges[:, 0].reshape((-1, 1)),
                faces == edges[:, 1].reshape((-1, 1)),
            )
        )
        # each row should have exactly one uncontained verted
        row_ok = unshared.sum(axis=1) == 1
        # any degenerate row should be ignored
        unshared[~row_ok, :] = False
        # set the
        vid_unshared[row_ok, i] = faces[unshared]

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
    #       distance
    # R = ---------------
    #     2 * sin(theta)
    nonzero = mesh.face_adjacency_angles > np.radians(0.01)
    denominator = np.abs(2.0 * np.sin(mesh.face_adjacency_angles[nonzero]))

    # consider the distance between the non- shared vertices of the
    # face adjacency pair as the key distance
    point_pairs = mesh.vertices[mesh.face_adjacency_unshared]
    vectors = np.diff(point_pairs, axis=1).reshape((-1, 3))

    # the vertex indices of the shared edge for the adjacency pairx
    edges = mesh.face_adjacency_edges
    # unit vector along shared the edge
    edges_vec = util.unitize(np.diff(mesh.vertices[edges], axis=1).reshape((-1, 3)))

    # the vector of the perpendicular projection to the shared edge
    perp = np.subtract(
        vectors, (util.diagonal_dot(vectors, edges_vec).reshape((-1, 1)) * edges_vec)
    )
    # the length of the perpendicular projection
    span = util.row_norm(perp)

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
    > [1, 3, 4]
    """
    g = nx.Graph()
    g.add_edges_from(mesh.edges_unique)
    return g


def shared_edges(faces_a, faces_b):
    """
    Given two sets of faces, find the edges which are in both sets.

    Parameters
    ---------
    faces_a : (n, 3) int
      Array of faces
    faces_b : (m, 3) int
      Array of faces

    Returns
    ---------
    shared : (p, 2) int
      Edges shared between faces
    """
    e_a = np.sort(faces_to_edges(faces_a), axis=1)
    e_b = np.sort(faces_to_edges(faces_b), axis=1)
    shared = grouping.boolean_rows(e_a, e_b, operation=np.intersect1d)
    return shared


def facets(mesh, engine: GraphEngineType = None, facet_threshold: Number | None = None):
    """
    Find the list of parallel adjacent faces.

    Parameters
    -----------
    mesh : trimesh.Trimesh
    engine
      Which graph engine to use
    facet_threshold : float
      Threshold for two facets to be considered coplanar

    Returns
    ---------
    facets : sequence of (n,) int
        Groups of face indexes of
        parallel adjacent faces.
    """
    if facet_threshold is None:
        facet_threshold = tol.facet_threshold
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
    parallel = np.ones(len(radii), dtype=bool)
    # if span is zero we know faces are small/parallel
    nonzero = np.abs(span) > tol.zero
    # faces with a radii/span ratio larger than a threshold pass
    parallel[nonzero] = (radii[nonzero] / span[nonzero]) ** 2 > facet_threshold

    # run connected components on the parallel faces to group them
    components = connected_components(
        mesh.face_adjacency[parallel],
        nodes=np.arange(len(mesh.faces)),
        min_len=2,
        engine=engine,
    )

    return components


def split(
    mesh,
    only_watertight: bool = True,
    repair: bool = True,
    adjacency: ArrayLike | None = None,
    engine: GraphEngineType = None,
    solids: bool = False,
    orient: bool = True,
    **kwargs,
) -> list:
    """
    Split a mesh into multiple meshes from face
    connectivity.

    If only_watertight is true it will only return
    watertight meshes and will attempt to repair
    single triangle or quad holes.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      The source multibody mesh to split
    only_watertight
      Only return watertight components and discard
      any connected component that isn't fully watertight.
    repair
      If set try to fill small holes in a mesh, before the
      discard step in `only_watertight.
    adjacency : (n, 2) int
      If passed will be used instead of `mesh.face_adjacency`
    engine
      Which graph engine to use for the connected components.
    solids
      If True, group watertight shells by geometric containment
      so cavity shells stay with their enclosing solid. Containment is
      evaluated with ray tests and assumes shells are disjoint: results
      are undefined for shells that touch or intersect one another.
    orient
      If True and `solids` is True, orient grouped solid shells
      so exteriors have positive volume and cavities have negative volume.
      Orientation flips whole shells based on their signed volume; shells
      with internally inconsistent winding are left untouched.

    Returns
    ----------
    meshes : (m,) trimesh.Trimesh
      Results of splitting based on parameters.
    """
    if adjacency is None:
        adjacency = mesh.face_adjacency

    if only_watertight:
        # the smallest watertight mesh has 4 faces
        min_len = 4
    else:
        # allow any size of connected component
        min_len = 1

    components = connected_components(
        edges=adjacency, nodes=np.arange(len(mesh.faces)), min_len=min_len, engine=engine
    )

    if solids and len(components) > 0:
        components, flips = _split_solids(mesh=mesh, components=components, orient=orient)
        # `min_faces` is applied per-group inside `submesh` which would drop
        # groups and break the positional mapping used to apply `flips`.
        # Apply it here and remap `flips` so the two stay aligned, and don't
        # pass it through to `submesh` a second time.
        min_faces = kwargs.pop("min_faces", None)
        if min_faces is not None:
            keep = [i for i, group in enumerate(components) if len(group) >= min_faces]
            flips = {new: flips[old] for new, old in enumerate(keep) if old in flips}
            components = [components[i] for i in keep]
        result = mesh.submesh(
            components, only_watertight=only_watertight, repair=repair, **kwargs
        )
        if orient and len(flips) > 0:
            _flip_result_groups(result=result, groups=components, flips=flips)
        return result

    return mesh.submesh(
        components, only_watertight=only_watertight, repair=repair, **kwargs
    )


def _component_volume(mesh, component) -> float:
    """
    Signed volume for a connected face component.
    """
    return triangles.mass_properties(
        mesh.triangles[component],
        crosses=mesh.triangles_cross[component],
        skip_inertia=True,
    ).volume


# number of ray-casting rounds `_contains_matrix` will attempt before giving
# up on a disagreeing pair: one fixed direction followed by random-direction
# retries. `ray.ray_util.contains_points` uses the same fixed-then-random
# approach but only retries once; batching many shells into a single query
# makes extra retries cheap here, so we allow a few more to further reduce the
# odds of leaving a pair unresolved.
_CONTAINS_ROUNDS = 4


def _contains_matrix(mesh, labels, points, candidate, direction=None):
    """
    Determine which shells contain which other shells' representative points.

    A single bidirectional ray query is run against the parent mesh's cached
    BVH and the hits are bucketed per shell using `labels`, replicating the
    parity test in `ray.contains_points` without building one intersector
    per shell.

    Parameters
    ----------
    mesh : Trimesh
      Source mesh whose `.ray` intersector and faces are queried.
    labels : (len(mesh.faces),) int
      Shell index for each face, or -1 for faces that belong to no shell.
    points : (n, 3) float
      One representative point per shell.
    candidate : (n, n) bool
      `candidate[i, j]` is True when shell i may contain shell j (AABB test).
    direction : (3,) float or None
      Initial ray direction; if None the fixed direction used by
      `ray.contains_points` is used. Later retries always use a random
      direction. Exposed mainly so tests can force the retry path.

    Returns
    ----------
    contains : (n, n) bool
      `contains[i, j]` is True when shell i contains shell j's point.
    """
    n = len(points)
    contains = np.zeros((n, n), dtype=bool)

    # only cast from origins that some shell's AABB could contain: for a fully
    # disjoint multibody mesh this is empty and we never build the ray BVH
    needed = candidate.any(axis=0)
    if not needed.any():
        return contains

    # a pair is settled once we trust its result; non-candidate pairs are
    # settled as "not contained" from the start so they are never revisited
    resolved = ~candidate

    # start with the fixed, debuggable direction used by ray.contains_points
    if direction is None:
        direction = np.array([0.4395064455, 0.617598629942, 0.652231566745])
    else:
        direction = np.array(direction, dtype=np.float64)

    for _ in range(_CONTAINS_ROUNDS):
        # origins that still have an unresolved candidate pair
        todo = np.nonzero(needed & ~resolved.all(axis=0))[0]
        if len(todo) == 0:
            break

        # cast forwards and backwards from every origin in a single query
        m = len(todo)
        origins = points[todo]
        both_origins = np.vstack((origins, origins))
        both_dirs = np.vstack((np.tile(direction, (m, 1)), np.tile(-direction, (m, 1))))
        _loc, index_ray, index_tri = mesh.ray.intersects_location(
            both_origins, both_dirs, multiple_hits=True
        )

        # per (origin position, shell) hit counts in each direction
        fwd = np.zeros((m, n), dtype=np.int64)
        bwd = np.zeros((m, n), dtype=np.int64)
        if len(index_ray) > 0:
            # ray index < m is the forward cast, >= m is the backward cast,
            # both belonging to origin `todo[position]`
            position = index_ray % m
            is_backward = index_ray >= m
            hit_shell = labels[index_tri]
            # drop faces belonging to no shell and self-hits on the origin
            keep = (hit_shell != -1) & (hit_shell != todo[position])
            pk, sk, bk = position[keep], hit_shell[keep], is_backward[keep]
            np.add.at(fwd, (pk[~bk], sk[~bk]), 1)
            np.add.at(bwd, (pk[bk], sk[bk]), 1)

        parity_fwd = (fwd % 2) == 1
        agree = parity_fwd == ((bwd % 2) == 1)
        # a ray that misses a shell in either direction proves the origin is
        # outside it: any ray from inside a closed shell must hit it an odd
        # (hence nonzero) number of times
        free = (fwd == 0) | (bwd == 0)

        # (m, n) view of which pairs for these origins are still open
        open_pair = ~resolved[:, todo].T
        settle_agree = agree & open_pair
        settle_free = free & ~agree & open_pair

        sub = contains[:, todo].T
        sub[settle_agree] = parity_fwd[settle_agree]
        sub[settle_free] = False
        contains[:, todo] = sub.T

        done = resolved[:, todo].T
        done[settle_agree | settle_free] = True
        resolved[:, todo] = done.T

        # remaining disagreements get a fresh random direction next round
        direction = util.unitize(np.random.random(3) - 0.5)

    # any still-unresolved candidate pairs default to "not contained" (outside)
    return contains


def _shell_bounds(mesh, closed_components):
    """
    Geometric inputs for the containment test over closed shells.

    Taken straight from the parent mesh's cached arrays, with no submesh
    construction, so `_split_solids` and its tests share one definition.

    Parameters
    ----------
    mesh : Trimesh
      Source mesh providing cached triangle and vertex arrays.
    closed_components : (n,) sequence of (m,) int
      Face indices for each closed (watertight) shell.

    Returns
    ----------
    mins : (n, 3) float
      Lower AABB corner of each shell.
    maxs : (n, 3) float
      Upper AABB corner of each shell.
    points : (n, 3) float
      One representative vertex per shell; for disjoint shells it is
      strictly inside or outside every other shell.
    candidate : (n, n) bool
      `candidate[i, j]` is True when shell i's AABB fully contains shell
      j's, a necessary condition for shell i to contain shell j.
    labels : (len(mesh.faces),) int
      Shell index for each face, or -1 for faces in no listed shell, so a
      single ray query against the parent BVH can be bucketed per shell.
    """
    face_min = mesh.triangles.min(axis=1)
    face_max = mesh.triangles.max(axis=1)
    mins = np.array([face_min[c].min(axis=0) for c in closed_components])
    maxs = np.array([face_max[c].max(axis=0) for c in closed_components])
    points = np.array([mesh.vertices[mesh.faces[c[0], 0]] for c in closed_components])

    candidate = (mins[:, None, :] <= mins[None, :, :] + tol.merge).all(axis=2) & (
        maxs[:, None, :] >= maxs[None, :, :] - tol.merge
    ).all(axis=2)
    np.fill_diagonal(candidate, False)

    labels = np.full(len(mesh.faces), -1, dtype=np.int64)
    for shell_index, c in enumerate(closed_components):
        labels[c] = shell_index

    return mins, maxs, points, candidate, labels


def _split_solids(mesh, components, orient: bool):
    """
    Group watertight connected components into solids by containment.

    Returns
    ----------
    grouped : (n,) sequence of (m,) int
      Face groups suitable for `mesh.submesh`.
    flips : dict
      Mapping from group index to ranges of output faces that should
      be flipped to normalize exterior/cavity winding.
    """
    components = [np.asanyarray(c, dtype=np.int64) for c in components]

    # Work directly on cached edge arrays so we don't need to construct
    # submeshes just to classify components as closed or open.
    face_edges = mesh.edges.reshape((-1, 6))
    closed = []
    # `wound[k]` mirrors `closed[k]`: whether that shell is consistently wound
    wound = []
    open_components = []
    for index, component in enumerate(components):
        # a watertight shell needs at least 4 faces (a tetrahedron) so
        # skip the edge check for anything smaller
        if len(component) >= 4:
            is_tight, is_wound = is_watertight(face_edges[component].reshape((-1, 2)))
        else:
            is_tight, is_wound = False, False
        if is_tight:
            closed.append(index)
            wound.append(is_wound)
        else:
            open_components.append(index)

    # no closed shells means nothing to group or orient
    if len(closed) == 0:
        return components, {}

    # Per-shell bounds, representative points, AABB candidate matrix and the
    # per-face shell labels used to bucket a single ray query against the
    # parent BVH. Shared with the direct helper test via `_shell_bounds`.
    mins, maxs, points, candidate, labels = _shell_bounds(
        mesh, [components[i] for i in closed]
    )

    contains = _contains_matrix(mesh, labels=labels, points=points, candidate=candidate)

    degree = contains.sum(axis=0)
    roots = np.nonzero((degree % 2) == 0)[0]
    root_order = roots[np.lexsort((roots, degree[roots]))]

    parent = {}
    for child in np.nonzero((degree % 2) == 1)[0]:
        candidates = np.nonzero(
            np.logical_and(contains[:, child], degree == (degree[child] - 1))
        )[0]
        if len(candidates) == 0:
            log.debug("contained shell has no immediate parent")
            continue
        if len(candidates) > 1:
            # multiple enclosing shells at the right depth only happens for
            # malformed (touching or intersecting) input; pick the tightest
            sizes = np.prod(maxs[candidates] - mins[candidates], axis=1)
            candidates = candidates[np.argsort(sizes)]
            log.debug("contained shell has multiple immediate parents")
        parent[child] = int(candidates[0])

    flip_component = {}
    volumes = {}
    if orient:
        for shell_index, component_index in enumerate(closed):
            # signed volume is only meaningful for a consistently wound shell;
            # flipping an inconsistently wound one as a unit is noise so leave
            # it untouched rather than guess an orientation from a bad volume
            if not wound[shell_index]:
                log.debug("skipping orient on winding-inconsistent shell")
                continue
            volume = _component_volume(mesh, components[component_index])
            volumes[component_index] = volume
            if np.abs(volume) < tol.zero:
                continue
            expected = 1.0 if (degree[shell_index] % 2) == 0 else -1.0
            flip_component[component_index] = bool(np.sign(volume) != expected)

    grouped = []
    flips = {}
    used = set()

    def append_group(component_indexes):
        group_index = len(grouped)
        ranges = []
        start = 0
        group = []
        for component_index in component_indexes:
            component = components[component_index]
            stop = start + len(component)
            if flip_component.get(component_index, False):
                ranges.append((start, stop))
            group.append(component)
            start = stop
        grouped.append(np.concatenate(group))
        if len(ranges) > 0:
            flips[group_index] = ranges

    # Emit all closed solid groups first so their output indexes remain stable
    # if only_watertight later drops open components.
    for root in root_order:
        child_indexes = [k for k, v in parent.items() if v == int(root)]
        component_indexes = [closed[int(root)]] + [closed[int(c)] for c in child_indexes]
        used.update(component_indexes)
        append_group(component_indexes)

    # Preserve every closed component even for malformed containment graphs.
    # A parentless shell reaching here is emitted as its own solid, so it must
    # be oriented to a positive volume rather than as a (negative) cavity.
    for component_index in closed:
        if component_index not in used:
            # `volumes` only holds consistently wound shells, so an
            # inconsistent shell is left untouched here as well
            if orient and component_index in volumes:
                volume = volumes[component_index]
                flip_component[component_index] = bool(
                    np.abs(volume) >= tol.zero and np.sign(volume) < 0
                )
            append_group([component_index])

    # Open components are never attached to solids; the existing submesh
    # filtering/repair flags decide whether they are returned.
    grouped.extend(components[i] for i in open_components)

    return grouped, flips


def _flip_mesh_ranges(mesh, ranges) -> None:
    """
    Flip selected face rows in a mesh and keep cached face normals aligned.
    """
    if len(ranges) == 0:
        return

    flip = np.zeros(len(mesh.faces), dtype=bool)
    for start, stop in ranges:
        flip[start:stop] = True
    if not flip.any():
        return

    if "face_normals" in mesh._cache:
        normals = mesh.face_normals.copy()
        normals[flip] *= -1.0
    else:
        normals = None

    mesh.faces[flip] = np.fliplr(mesh.faces[flip])
    if normals is not None:
        mesh.face_normals = normals


def _flip_result_groups(result, groups, flips) -> None:
    """
    Apply grouped face flips to a normal split result or appended mesh result.
    """
    if isinstance(result, list):
        for group_index, ranges in flips.items():
            if group_index < len(result):
                _flip_mesh_ranges(result[group_index], ranges)
        return

    offset = 0
    ranges = []
    for group_index, group in enumerate(groups):
        for start, stop in flips.get(group_index, []):
            ranges.append((offset + start, offset + stop))
        offset += len(group)
    _flip_mesh_ranges(result, ranges)


def connected_components(
    edges, min_len: Integer = 1, nodes=None, engine: GraphEngineType = None
):
    """
    Find groups of connected nodes from an edge list.

    Parameters
    -----------
    edges : (n, 2) int
      Edges between nodes
    nodes : (m, ) int or None
      List of nodes that exist
    min_len : int
      Minimum length of a component group to return
    engine :  str or None
      Which graph engine to use (None for automatic):
      (None, 'networkx', 'scipy')


    Returns
    -----------
    components : (n,) sequence of (*,) int
      Nodes which are connected
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
        return [list(i) for i in nx.connected_components(graph)]

    def components_csgraph():
        """
        Find connected components using scipy.sparse.csgraph
        """
        # label each node
        labels = connected_component_labels(edges, node_count=node_count)

        # we have to remove results that contain nodes outside
        # of the specified node set and reindex
        contained = np.zeros(node_count, dtype=bool)
        contained[nodes] = True
        index = np.arange(node_count, dtype=np.int64)[contained]
        components = grouping.group(labels[contained], min_len=min_len)
        return [index[c] for c in components]

        return components

    # check input edges
    edges = np.asanyarray(edges, dtype=np.int64)
    # if no nodes were specified just use unique
    if nodes is None:
        nodes = np.unique(edges)

    # exit early if we have no nodes
    if len(nodes) == 0:
        return []
    elif len(edges) == 0:
        if min_len <= 1:
            return np.reshape(nodes, (-1, 1)).tolist()
        else:
            return []

    if not util.is_shape(edges, (-1, 2)):
        raise ValueError("edges must be (n, 2)!")

    # find the maximum index referenced in either nodes or edges
    counts = [0]
    if len(edges) > 0:
        counts.append(edges.max())
    if len(nodes) > 0:
        counts.append(nodes.max())
    node_count = np.max(counts) + 1

    # remove edges that don't have both nodes in the node set
    mask = np.zeros(node_count, dtype=bool)
    mask[nodes] = True
    edges_ok = mask[edges].all(axis=1)
    edges = edges[edges_ok]

    # networkx is pure python and is usually 5-10x slower than scipy
    engines = collections.OrderedDict(
        (("scipy", components_csgraph), ("networkx", components_networkx))
    )

    # if a graph engine has explicitly been requested use it
    if engine in engines:
        return engines[engine]()

    # otherwise, go through our ordered list of graph engines
    # until we get to one that has actually been installed
    for function in engines.values():
        try:
            return function()
        # will be raised if the library didn't import correctly above
        except BaseException:
            continue
    raise ImportError("no graph engines available!")


def connected_component_labels(edges, node_count=None):
    """
    Label graph nodes from an edge list, using scipy.sparse.csgraph

    Parameters
    -----------
    edges : (n, 2) int
       Edges of a graph
    node_count : int, or None
        The largest node in the graph.

    Returns
    ----------
    labels : (node_count,) int
        Component labels for each node
    """
    matrix = edges_to_coo(edges, node_count)
    _body_count, labels = csgraph.connected_components(matrix, directed=False)

    if node_count is not None:
        assert len(labels) == node_count

    return labels


def _split_traversal(traversal: NDArray, edges_tree) -> list[NDArray]:
    """
    Given a traversal as a list of nodes split the traversal
    if a sequential index pair is not in the given edges.
    Useful since the implementation of DFS we're using will
    happily return disconnected values in a flat traversal.

    Parameters
    --------------
    traversal : (m,) int
       Traversal through edges
    edges_tree : cKDTree
      A way to reconstruct original edge indices from
      sorted (n, 2) edge values. This is a slight misuse of a
      kdtree since one could just hash the tuples of the integer
      edges, but that isn't possible with numpy arrays easily
      and this allows a vectorized reconstruction.

    Returns
    ---------------
    split : sequence of (p,) int
      Traversals split into only connected paths.
    """
    # turn the (n,) traversal into (n-1, 2) edges
    # save the original order of the edges for reconstruction
    trav_edge = np.column_stack((traversal[:-1], traversal[1:]))

    # check to see if the traversal edges exists in the original edges
    # the tree is holding the sorted edges so query after sorting
    exists = edges_tree.query(np.sort(trav_edge, axis=1))[0] < 1e-10

    if exists.all():
        split = [traversal]
    else:
        # contiguous groups of edges
        blocks = grouping.blocks(exists, min_len=1, only_nonzero=True)
        split = [
            np.concatenate([trav_edge[:, 0][b], trav_edge[b[-1]][1:]]) for b in blocks
        ]

    # a traversal may be effectively closed so check
    # to see if we need to add on the first index to the end
    needs_close = np.array(
        [
            len(s) > 2
            and s[0] != s[-1]
            and edges_tree.query(sorted([s[0], s[-1]]))[0] < 1e-10
            for s in split
        ]
    )

    if needs_close.any():
        for i in np.nonzero(needs_close)[0]:
            split[i] = np.concatenate([split[i], split[i][:1]])

    if tol.strict:
        for s in split:
            check_edge = np.sort(np.column_stack((s[:-1], s[1:])), axis=1)
            assert (edges_tree.query(check_edge)[0] < 1e-10).all()

    return split


def fill_traversals(traversals: Sequence, edges: ArrayLike) -> Sequence | NDArray:
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

    Returns
    --------------
    splits : sequence of (p,) int
       Node indexes of connected traversals
    """
    # make sure edges are correct type
    edges = np.sort(edges, axis=1)
    edges_tree = cKDTree(edges)

    # if there are no traversals just return edges
    if len(traversals) == 0:
        return edges.copy()

    splits = []
    for nodes in traversals:
        # split traversals to remove edges that don't actually exist
        splits.extend(_split_traversal(traversal=nodes, edges_tree=edges_tree))

    # turn the split traversals back into (n, 2) edges
    included = util.vstack_empty([np.column_stack((i[:-1], i[1:])) for i in splits])
    if len(included) > 0:
        # sort included edges in place
        included.sort(axis=1)
        # make sure any edges not included in split traversals
        # are just added as a length 2 traversal
        splits.extend(grouping.boolean_rows(edges, included, operation=np.setdiff1d))
    else:
        # no edges were included, so our filled traversal
        # is just the original edges copied over
        splits = edges.copy()

    return splits


def traversals(edges, mode="bfs"):
    """
    Given an edge list generate a sequence of ordered depth
    first search traversals using scipy.csgraph routines.

    Parameters
    ------------
    edges : (n, 2) int
      Undirected edges of a graph
    mode :  str
      Traversal type, 'bfs' or 'dfs'

    Returns
    -----------
    traversals : (m,) sequence of (p,) int
      Ordered DFS or BFS traversals of the graph.
    """
    edges = np.array(edges, dtype=np.int64)
    if len(edges) == 0:
        return []
    elif not util.is_shape(edges, (-1, 2)):
        raise ValueError("edges are not (n, 2)!")

    # pick the traversal method
    mode = str(mode).lower().strip()
    if mode == "bfs":
        func = csgraph.breadth_first_order
    elif mode == "dfs":
        func = csgraph.depth_first_order
    else:
        raise ValueError("traversal mode must be either dfs or bfs")

    # make sure edges are sorted so we can query
    # an ordered pair later
    edges.sort(axis=1)
    # coo_matrix for csgraph routines
    graph = edges_to_coo(edges)

    # get unique nodes AND leaf node info from bincount
    bincount = np.bincount(edges.ravel())

    # a leaf node occurs exactly once
    nodes_leaf = np.nonzero(bincount == 1)[0]
    # a non-leaf node occurs more than once
    nodes = np.nonzero(bincount > 1)[0]

    # collect traversals
    traversals = []
    # keep track of which nodes have been visited
    visited = np.zeros(len(bincount) + 1, dtype=np.bool_)

    # process leaf nodes first to avoid DFS backtracking
    # starting DFS from an interior node of an open path causes
    # backtracking, which produces non-edge consecutive pairs that
    # fill_traversals then splits a single path into pieces
    for start in np.concatenate((nodes_leaf, nodes)):
        if visited[start]:
            continue

        # get an (n,) ordered traversal from this start
        ordered = func(
            graph, i_start=start, return_predecessors=False, directed=False
        ).astype(np.int64)

        # store the traversal and mark all nodes as visited
        traversals.append(ordered)
        visited[ordered] = True

    return traversals


def edges_to_coo(edges, count=None, data=None):
    """
    Given an edge list, return a boolean scipy.sparse.coo_matrix
    representing the edges in matrix form.

    Parameters
    ------------
    edges : (n, 2) int
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
    if not (len(edges) == 0 or util.is_shape(edges, (-1, 2))):
        raise ValueError("edges must be (n, 2)!")

    # if count isn't specified just set it to largest
    # value referenced in edges
    if count is None:
        count = edges.max() + 1
    count = int(count)

    # if no data is specified set every specified edge
    # to True
    if data is None:
        data = np.ones(len(edges), dtype=bool)

    return coo_matrix((data, edges.T), dtype=data.dtype, shape=(count, count))


def neighbors(edges, max_index=None, directed=False):
    """
    Find the neighbors for each node in an edgelist graph.

    TODO : re-write this with sparse matrix operations

    Parameters
    ------------
    edges : (n, 2) int
      Connected nodes
    directed : bool
      If True, only connect edges in one direction

    Returns
    ---------
    neighbors : sequence
      Vertex index corresponds to set of other vertex indices
    """
    neighbors = collections.defaultdict(set)
    if directed:
        [neighbors[edge[0]].add(edge[1]) for edge in edges]
    else:
        [
            (neighbors[edge[0]].add(edge[1]), neighbors[edge[1]].add(edge[0]))
            for edge in edges
        ]

    if max_index is None:
        max_index = edges.max() + 1
    array = [list(neighbors[i]) for i in range(max_index)]

    return array


def smooth_shade(mesh, angle: Number | None = None, facet_minarea: Number | None = 10.0):
    """
    Return a non-watertight version of the mesh which
    will render nicely with smooth shading by
    disconnecting faces at sharp angles to each other.

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Source geometry
    angle : float or None
      Angle in radians face pairs with angles
      smaller than this will appear smoothed
    facet_minarea : float or None
      Minimum area fraction to consider
      IE for `facets_minarea=25` only facets larger
      than `mesh.area / 25` will be considered.

    Returns
    ---------
    smooth : trimesh.Trimesh
      Geometry with disconnected face patches
    """
    if angle is None:
        angle = np.radians(20)

    # if the mesh has no adjacent faces return a copy
    if len(mesh.face_adjacency) == 0:
        return mesh.copy()

    # face pairs below angle threshold
    angle_ok = mesh.face_adjacency_angles < angle
    # subset of face adjacency
    adjacency = mesh.face_adjacency[angle_ok]

    # coplanar groups of faces
    facets = []
    nodes = None
    # collect coplanar regions for smoothing
    if facet_minarea is not None:
        areas = mesh.area_faces
        min_area = mesh.area / facet_minarea
        try:
            # we can survive not knowing facets
            # exclude facets with few faces
            facets = [f for f in mesh.facets if areas[f].sum() > min_area]
            if len(facets) > 0:
                # mask for removing adjacency pairs where
                # one of the faces is contained in a facet
                mask = np.ones(len(mesh.faces), dtype=bool)
                mask[np.hstack(facets)] = False
                # apply the mask to adjacency
                adjacency = adjacency[mask[adjacency].all(axis=1)]
                # nodes are no longer every faces
                nodes = np.unique(adjacency)
        except BaseException:
            log.warning("failed to calculate facets", exc_info=True)
    # run connected components on facet adjacency
    components = connected_components(adjacency, min_len=2, nodes=nodes)

    # add back coplanar groups if any exist
    if len(facets) > 0:
        components.extend(facets)

    if len(components) == 0:
        # if no components for some reason
        # just return a copy of the original mesh
        return mesh.copy()

    # add back any faces that were missed
    unique = np.unique(np.hstack(components))
    if len(unique) != len(mesh.faces):
        # things like single loose faces
        # or groups below facet_minlen
        broke = np.setdiff1d(np.arange(len(mesh.faces)), unique)
        components.extend(broke.reshape((-1, 1)))

    # get a submesh as a single appended Trimesh
    smooth = mesh.submesh(components, only_watertight=False, append=True)
    # store face indices from original mesh
    smooth.metadata["original_components"] = components
    # smoothed should have exactly the same number of faces
    if len(smooth.faces) != len(mesh.faces):
        log.warning("face count in smooth wrong!")
    return smooth


def is_watertight(
    edges: ArrayLike, edges_sorted: ArrayLike | None = None
) -> tuple[bool, bool]:
    """
    Parameters
    -----------
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

    # group sorted edges throwing away any edge
    # that doesn't appear exactly twice
    groups = grouping.group_rows(edges_sorted, require_count=2)
    # if we didn't throw away any edges that means
    # every edge shows up exactly twice
    watertight = bool((len(groups) * 2) == len(edges))

    # check that the un-sorted duplicate edges are reversed
    opposing = edges[groups].reshape((-1, 4))[:, 1:3].T
    winding = bool(np.equal(*opposing).all())

    return watertight, winding


def graph_to_svg(graph):
    """
    Turn a networkx graph into an SVG string
    using graphviz `dot`.

    Parameters
    ----------
    graph: networkx graph

    Returns
    ---------
    svg: string, pictoral layout in SVG format
    """

    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile() as dot_file:
        nx.drawing.nx_agraph.write_dot(graph, dot_file.name)
        svg = subprocess.check_output(["dot", dot_file.name, "-Tsvg"])
    return svg


def multigraph_paths(G, source, cutoff=None):
    """
    For a networkx MultiDiGraph, find all paths from a source node
    to leaf nodes. This function returns edge instance numbers
    in addition to nodes, unlike networkx.all_simple_paths.

    Parameters
    ---------------
    G : networkx.MultiDiGraph
      Graph to evaluate
    source : hashable
      Node to start traversal at
    cutoff : int
      Number of nodes to visit
      If None will visit all nodes

    Returns
    ----------
    traversals : (n,) list of [(node, edge instance index), ] paths
      Traversals of the multigraph
    """
    if cutoff is None:
        cutoff = (len(G.edges()) * len(G.nodes())) + 1

    # the path starts at the node specified
    current = [(source, 0)]
    # traversals we need to go back and do
    queue = []
    # completed paths
    traversals = []

    for _ in range(cutoff):
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
