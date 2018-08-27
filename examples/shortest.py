"""
shortest.py
----------------

Given a mesh and two vertex indices find the shortest path
between the two vertices while only traveling along edges
of the mesh.
"""

import trimesh
import numpy as np

import networkx as nx

if __name__ == '__main__':

    # test on a sphere mesh
    mesh = trimesh.primitives.Sphere()

    # edges without duplication
    edges = mesh.edges_unique

    # the actual length of each unique edge
    length = np.linalg.norm(mesh.vertices[edges[:,0]] -
                            mesh.vertices[edges[:,1]],
                            axis=1)

    # create the graph with edge attributes for length
    g = nx.Graph()
    for edge, L in zip(edges, length):
        g.add_edge(*edge, length=L)

    # alternative method for weighted graph creation
    # you can also create the graph with from_edgelist and
    # a list comprehension, which is like 1.5x faster
    ga = nx.from_edgelist([(e[0], e[1], {'length': L})
                           for e, L in zip (edges, length)])

    # arbitrary indices of mesh.vertices to test with
    start = 0
    end   = int(len(mesh.vertices) / 2.0)

    # run the shortest path query using length for edge weight
    path = nx.shortest_path(g,
                            source=start,
                            target=end,
                            weight='length')


    ### VISUALIZE RESULT
    # make the sphere transparent-ish
    mesh.visual.face_colors = [100, 100, 100, 100]
    # make a scene with the two points, the path between them
    # and the original mesh we were working on
    trimesh.Scene([
        trimesh.points.PointCloud(mesh.vertices[[start, end]]),
        mesh,
        trimesh.load_path(mesh.vertices[path])]).show()
    
