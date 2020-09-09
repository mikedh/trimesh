"""
repair.py
-------------

Fill holes and fix winding and normals of meshes.
"""

import numpy as np

from . import graph
from . import triangles

from .constants import log
from .grouping import group_rows
from .geometry import faces_to_edges


try:
    import networkx as nx
except BaseException as E:
    # create a dummy module which will raise the ImportError
    # or other exception only when someone tries to use networkx
    from .exceptions import ExceptionModule
    nx = ExceptionModule(E)


def fix_winding(mesh):
    """
    Traverse and change mesh faces in-place to make sure winding
    is correct, with edges on adjacent faces in
    opposite directions.

    Parameters
    -------------
    mesh: Trimesh object

    Alters
    -------------
    mesh.face: will reverse columns of certain faces
    """
    # anything we would fix is already done
    if mesh.is_winding_consistent:
        return

    graph_all = nx.from_edgelist(mesh.face_adjacency)
    flipped = 0

    faces = mesh.faces.view(np.ndarray).copy()

    # we are going to traverse the graph using BFS
    # start a traversal for every connected component
    for components in nx.connected_components(graph_all):
        # get a subgraph for this component
        g = graph_all.subgraph(components)
        # get the first node in the graph in a way that works on nx's
        # new API and their old API
        start = next(iter(g.nodes()))

        # we traverse every pair of faces in the graph
        # we modify mesh.faces and mesh.face_normals in place
        for face_pair in nx.bfs_edges(g, start):
            # for each pair of faces, we convert them into edges,
            # find the edge that both faces share and then see if edges
            # are reversed in order as you would expect
            # (2, ) int
            face_pair = np.ravel(face_pair)
            # (2, 3) int
            pair = faces[face_pair]
            # (6, 2) int
            edges = faces_to_edges(pair)
            overlap = group_rows(np.sort(edges, axis=1),
                                 require_count=2)
            if len(overlap) == 0:
                # only happens on non-watertight meshes
                continue
            edge_pair = edges[overlap[0]]
            if edge_pair[0][0] == edge_pair[1][0]:
                # if the edges aren't reversed, invert the order of one face
                flipped += 1
                faces[face_pair[1]] = faces[face_pair[1]][::-1]

    if flipped > 0:
        mesh.faces = faces

    log.debug('flipped %d/%d edges', flipped, len(mesh.faces) * 3)


def fix_inversion(mesh, multibody=False):
    """
    Check to see if a mesh has normals pointing "out."

    Parameters
    -------------
    mesh:      Trimesh object
    multibody: bool, if True will try to fix normals on every body

    Alters
    -------------
    mesh.face: may reverse faces
    """
    if multibody:
        groups = graph.connected_components(mesh.face_adjacency)
        # escape early for single body
        if len(groups) == 1:
            if mesh.volume < 0.0:
                mesh.invert()
            return
        # mask of faces to flip
        flip = np.zeros(len(mesh.faces), dtype=np.bool)
        # save these to avoid thrashing cache
        tri = mesh.triangles
        cross = mesh.triangles_cross
        # indexes of mesh.faces, not actual faces
        for faces in groups:
            # calculate the volume of the submesh faces
            volume = triangles.mass_properties(
                tri[faces],
                crosses=cross[faces],
                skip_inertia=True)['volume']
            # if that volume is negative it is either
            # inverted or just total garbage
            if volume < 0.0:
                flip[faces] = True
        # one or more faces needs flipping
        if flip.any():
            # flip normals of necessary faces
            if 'face_normals' in mesh._cache:
                normals = mesh.face_normals.copy()
                normals[flip] *= -1.0
            else:
                normals = None
            # flip faces
            mesh.faces[flip] = np.fliplr(mesh.faces[flip])
            if normals is not None:
                mesh.face_normals = normals

    elif mesh.volume < 0.0:
        mesh.invert()


def fix_normals(mesh, multibody=False):
    """
    Fix the winding and direction of a mesh face and
    face normals in-place.

    Really only meaningful on watertight meshes but will orient all
    faces and winding in a uniform way for non-watertight face
    patches as well.

    Parameters
    -------------
    mesh : trimesh.Trimesh
      Mesh to fix normals on
    multibody : bool
      if True try to correct normals direction
      on every body rather than just one

    Alters
    --------------
    mesh.faces
      Will flip columns on inverted faces
    """
    # traverse face adjacency to correct winding
    fix_winding(mesh)
    # check to see if a mesh is inverted
    fix_inversion(mesh, multibody=multibody)


def broken_faces(mesh, color=None):
    """
    Return the index of faces in the mesh which break the
    watertight status of the mesh.

    Parameters
    --------------
    mesh: Trimesh object
    color: (4,) uint8, will set broken faces to this color
           None,       will not alter mesh colors

    Returns
    ---------------
    broken: (n, ) int, indexes of mesh.faces
    """
    adjacency = nx.from_edgelist(mesh.face_adjacency)
    broken = [k for k, v in dict(adjacency.degree()).items()
              if v != 3]
    broken = np.array(broken)
    if color is not None and broken.size != 0:
        # if someone passed a broken color
        color = np.array(color)
        if not (color.shape == (4,) or color.shape == (3,)):
            color = [255, 0, 0, 255]
        mesh.visual.face_colors[broken] = color
    return broken


def fill_holes(mesh):
    """
    Fill single- triangle holes on triangular meshes by adding
    new triangles to fill the holes. New triangles will have
    proper winding and normals, and if face colors exist the color
    of the last face will be assigned to the new triangles.

    Parameters
    ---------
    mesh : trimesh.Trimesh
      Mesh will be repaired in- place
    """

    def hole_to_faces(hole):
        """
        Given a loop of vertex indices  representing a hole, turn it into
        triangular faces.
        If unable to do so, return None

        Parameters
        ---------
        hole:     ordered loop of vertex indices

        Returns
        ---------
        (n, 3) new faces
        (m, 3) new vertices
        """
        hole = np.asanyarray(hole)
        # the case where the hole is just a single missing triangle
        if len(hole) == 3:
            return [hole], []
        # the hole is a quad, which we fill with two triangles
        if len(hole) == 4:
            face_A = hole[[0, 1, 2]]
            face_B = hole[[2, 3, 0]]
            return [face_A, face_B], []
        return [], []

    if len(mesh.faces) < 3:
        return False

    if mesh.is_watertight:
        return True

    # we know that in a watertight mesh every edge will be included twice
    # thus every edge which appears only once is part of a hole boundary
    boundary_groups = group_rows(
        mesh.edges_sorted, require_count=1)

    # mesh is not watertight and we have too few edges
    # edges to do a repair
    # since we haven't changed anything return False
    if len(boundary_groups) < 3:
        return False

    boundary_edges = mesh.edges[boundary_groups]
    index_as_dict = [{'index': i} for i in boundary_groups]

    # we create a graph of the boundary edges, and find cycles.
    g = nx.from_edgelist(
        np.column_stack((boundary_edges,
                         index_as_dict)))
    new_faces = []
    new_vertex = []
    for hole in nx.cycle_basis(g):
        # convert the hole, which is a polygon of vertex indices
        # to triangles and new vertices
        faces, vertex = hole_to_faces(hole=hole)
        if len(faces) == 0:
            continue
        # remeshing returns new vertices as negative indices, so change those
        # to absolute indices which won't be screwed up by the later appends
        faces = np.array(faces)
        faces[faces < 0] += len(new_vertex) + len(mesh.vertices) + len(vertex)
        new_vertex.extend(vertex)
        new_faces.extend(faces)
    new_faces = np.array(new_faces)
    new_vertex = np.array(new_vertex)

    if len(new_faces) == 0:
        # no new faces have been added, so nothing further to do
        # the mesh is NOT watertight, as boundary groups exist
        # but we didn't add any new faces to fill them in
        return False

    for face_index, face in enumerate(new_faces):
        # we compare the edge from the new face with
        # the boundary edge from the source mesh
        edge_test = face[:2]
        edge_boundary = mesh.edges[g.get_edge_data(*edge_test)['index']]

        # in a well construtced mesh, the winding is such that adjacent triangles
        # have reversed edges to each other. Here we check to make sure the
        # edges are reversed, and if they aren't we simply reverse the face
        reversed = edge_test[0] == edge_boundary[1]
        if not reversed:
            new_faces[face_index] = face[::-1]

    # stack vertices into clean (n, 3) float
    if len(new_vertex) != 0:
        new_vertices = np.vstack((mesh.vertices, new_vertex))
    else:
        new_vertices = mesh.vertices

    # try to save face normals if we can
    if 'face_normals' in mesh._cache.cache:
        cached_normals = mesh._cache.cache['face_normals']
    else:
        cached_normals = None

    # also we can remove any zero are triangles by masking here
    new_normals, valid = triangles.normals(new_vertices[new_faces])
    # all the added faces were broken
    if not valid.any():
        return False

    # this is usually the case where two vertices of a triangle are just
    # over tol.merge apart, but the normal calculation is screwed up
    # these could be fixed by merging the vertices in question here:
    # if not valid.all():
    if mesh.visual.defined and mesh.visual.kind == 'face':
        color = mesh.visual.face_colors
    else:
        color = None

    # apply the new faces and vertices
    mesh.faces = np.vstack((mesh._data['faces'], new_faces[valid]))
    mesh.vertices = new_vertices

    # dump the cache and set id to the new hash
    mesh._cache.verify()

    # save us a normals recompute if we can
    if cached_normals is not None:
        mesh.face_normals = np.vstack((cached_normals,
                                       new_normals))

    # this is usually the case where two vertices of a triangle are just
    # over tol.merge apart, but the normal calculation is screwed up
    # these could be fixed by merging the vertices in question here:
    # if not valid.all():
    if color is not None:
        # if face colors exist, assign the last face color to the new faces
        # note that this is a little cheesey, but it is very inexpensive and
        # is the right thing to do if the mesh is a single color.
        color_shape = np.shape(color)
        if len(color_shape) == 2:
            new_colors = np.tile(color[-1], (np.sum(valid), 1))
            new_colors = np.vstack((color,
                                    new_colors))
            mesh.visual.face_colors = new_colors

    log.debug('Filled in mesh with %i triangles', np.sum(valid))
    return mesh.is_watertight
