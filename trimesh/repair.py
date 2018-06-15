"""
repair.py
-------------

Fill holes and fix winding and normals of meshes.
"""

import numpy as np
import networkx as nx

from . import graph
from . import triangles

from .constants import log
from .grouping import group_rows
from .geometry import faces_to_edges


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
        graph = graph_all.subgraph(components)
        # get the first node in the graph in a way that works on nx's
        # new API and their old API
        start = next(iter(graph.nodes()))

        # we traverse every pair of faces in the graph
        # we modify mesh.faces and mesh.face_normals in place
        for face_pair in nx.bfs_edges(graph, start):
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
            edge_pair = edges[[overlap[0]]]
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
            with mesh._cache:
                # flip normals of necessary faces
                if 'face_normals' in mesh._cache:
                    mesh.face_normals[flip] *= -1.0
                # flip faces
                mesh.faces[flip] = np.fliplr(mesh.faces[flip])
            # save wangled normals
            mesh._cache.clear(exclude=['face_normals'])

    elif mesh.volume < 0.0:
        # reverse every triangles and flip every normals
        mesh.invert()


def fix_normals(mesh, multibody=False):
    """
    Fix the winding and direction of a mesh face and
    face normals in-place.

    Really only meaningful on watertight meshes, but will orient all
    faces and winding in a uniform way for non-watertight face
    patches as well.

    Parameters
    -------------
    mesh:      Trimesh object
    multibody: bool, if True try to correct normals direction
                     on every body.

    Alters
    --------------
    mesh.faces: will flip columns on inverted faces
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
    if color is not None:
        # if someone passed a broken color
        color = np.array(color)
        if not (color.shape == (4,) or color.shape == (3,)):
            color = [255, 0, 0, 255]
        mesh.visual.face_colors[broken] = color
    return broken


def fill_holes(mesh):
    """
    Fill single- triangle holes on triangular meshes by adding new triangles
    to fill the holes. New triangles will have proper winding and normals,
    and if face colors exist the color of the last face will be assigned
    to the new triangles.

    Parameters
    ---------
    mesh: Trimesh object
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

    # we know that in a watertight mesh, every edge will be included twice
    # thus, every edge which appears only once is part of the boundary of a
    # hole.
    boundary_groups = group_rows(mesh.edges_sorted, require_count=1)

    if len(boundary_groups) < 3:
        watertight = len(boundary_groups) == 0
        return watertight

    boundary_edges = mesh.edges[boundary_groups]
    index_as_dict = [{'index': i} for i in boundary_groups]

    # we create a graph of the boundary edges, and find cycles.
    graph = nx.from_edgelist(np.column_stack((boundary_edges,
                                              index_as_dict)))
    cycles = np.array(nx.cycle_basis(graph))

    new_faces = []
    new_vertex = []
    for hole in cycles:
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
        edge_test = face[0:2]
        edge_boundary = mesh.edges[graph.get_edge_data(*edge_test)['index']]

        # in a well construced mesh, the winding is such that adjacent triangles
        # have reversed edges to each other. Here we check to make sure the
        # edges are reversed, and if they aren't we simply reverse the face
        reversed = edge_test[0] == edge_boundary[1]
        if not reversed:
            new_faces[face_index] = face[::-1]

    if len(new_vertex) != 0:
        new_vertices = np.vstack((mesh.vertices, new_vertex))
    else:
        new_vertices = mesh.vertices

    # since the winding is now correct, we can get consistant normals
    # just by doing the cross products on the face edges
    mesh._cache.clear(exclude=['face_normals'])
    new_normals, valid = triangles.normals(new_vertices[new_faces])
    mesh.face_normals = np.vstack((mesh.face_normals, new_normals))
    mesh.faces = np.vstack((mesh._data['faces'], new_faces[valid]))
    mesh.vertices = new_vertices
    mesh._cache.id_set()

    # this is usually the case where two vertices of a triangle are just
    # over tol.merge apart, but the normal calculation is screwed up
    # these could be fixed by merging the vertices in question here:
    # if not valid.all():
    if mesh.visual.defined and mesh.visual.kind == 'face':
        # if face colors exist, assign the last face color to the new faces
        # note that this is a little cheesey, but it is very inexpensive and
        # is the right thing to do if the mesh is a single color.
        stored = mesh.visual._data['face_colors']
        color_shape = np.shape(stored)
        if len(color_shape) == 2:
            new_colors = np.tile(stored[-1], (np.sum(valid), 1))
            new_colors = np.vstack((stored,
                                    new_colors))
            mesh.visual.face_colors = new_colors

    log.debug('Filled in mesh with %i triangles', np.sum(valid))
    return mesh.is_watertight
