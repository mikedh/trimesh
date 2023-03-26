"""
examples/nricp.py
------------------------

Register a ball to a teeth

For more information, see
"Amberg et al. 2007: Optimal Step Nonrigid ICP Algorithms for Surface Registration."

"""

import trimesh
import numpy as np
from trimesh.registration import (nricp_amberg,
                                  nricp_sumner,
                                  procrustes)
from trimesh.proximity import closest_point
from trimesh.triangles import points_to_barycentric


def slider_closure(records, pv_mesh):
    """
    Return a function used for a PyVista slider widget.
    """
    def cb(value):
        t1 = min(int(value), len(records) - 1)
        t2 = min(t1 + 1, len(records) - 1)
        t = value - t1
        pv_mesh.points = (1 - t) * records[t1] + t * records[t2]
        for i, pos in enumerate(
                pv_mesh.points[landmarks_vertex_indices[:, 0]]):
            p.add_mesh(
                pv.Sphere(
                    target.scale / 200,
                    pos),
                name=str(i),
                color='r')
            pv_mesh['distance'] = (
                1 - t) * distances[t1] + t * distances[t2]
    return cb


if __name__ == '__main__':

    # attach to trimesh logs
    trimesh.util.attach_to_log()

    # Get two meshes that have a comparable shape
    source = trimesh.load_mesh('../models/reference.obj', process=False)
    target = trimesh.load_mesh('../models/target.obj', process=False)

    # Vertex indices of landmarks source / target
    landmarks_vertex_indices = np.array([
        [177, 1633],
        [181, 1561],
        [614, 1556],
        [610, 1629],
        [114, 315],
        [398, 413],
        [812, 412],
        [227, 99],
        [241, 87],
        [674, 86],
        [660, 98],
        [362, 574],
        [779, 573],
    ])

    source_markers_vertices = source.vertices[landmarks_vertex_indices[:, 0]]
    target_markers_vertices = target.vertices[landmarks_vertex_indices[:, 1]]

    T = procrustes(source_markers_vertices, target_markers_vertices)[0]
    source.apply_transform(T)

    # Just for the sake of using barycentric coordinates...
    use_barycentric_coordinates = True
    if use_barycentric_coordinates:
        source_markers_vertices = source.vertices[landmarks_vertex_indices[:, 0]]
        source_markers_tids = closest_point(source, source_markers_vertices)[2]
        source_markers_barys = \
            points_to_barycentric(source.triangles[source_markers_tids],
                                  source_markers_vertices)
        source_landmarks = (source_markers_tids, source_markers_barys)
    else:
        source_landmarks = landmarks_vertex_indices[:, 0]

    # Parameters for nricp_amberg
    wl = 3
    max_iter = 10
    wn = 0.5
    steps_amberg = [
        # ws, wl, wn, max_iter
        [0.02, wl, wn, max_iter],
        [0.007, wl, wn, max_iter],
        [0.002, wl, wn, max_iter],
    ]

    # Parameters for nricp_sumner
    wl = 1000
    wi = 0.0001
    ws = 1
    wn = 1
    steps_sumner = [
        # wc, wi, ws, wl, wn
        [0, wi, ws, wl, wn],
        [0.1, wi, ws, wl, wn],
        [1, wi, ws, wl, wn],
        [5, wi, ws, wl, wn],
        [10, wi, ws, wl, wn],
        [50, wi, ws, wl, wn],
        [100, wi, ws, wl, wn],
        [1000, wi, ws, wl, wn],
        [10000, wi, ws, wl, wn],
    ]

    # Amberg et al. 2007
    records_amberg = nricp_amberg(
        source, target, source_landmarks=source_landmarks,
        distance_threshold=0.05,
        target_positions=target_markers_vertices,
        steps=steps_amberg, return_records=True)

    # Sumner and Popovic 2004
    records_sumner = nricp_sumner(
        source, target,
        source_landmarks=source_landmarks,
        distance_threshold=0.05,
        target_positions=target_markers_vertices,
        steps=steps_sumner, return_records=True)

    # Show the result
    try:
        import pyvista as pv
        for records, name in [(records_amberg, 'Amberg et al. 2007'),
                              (records_sumner, 'Sumner and Popovic 2004')]:
            distances = [closest_point(target, r)[1] for r in records]
            p = pv.Plotter()
            p.background_color = 'w'
            pv_mesh = pv.wrap(source)
            pv_mesh['distance'] = distances[0]
            p.add_text(name, color=(0, 0, 0))
            p.add_mesh(
                pv_mesh, color=(0.6, 0.6, 0.9), cmap='rainbow',
                clim=(0, target.scale / 100), scalars='distance',
                scalar_bar_args={'color': (0, 0, 0)})
            p.add_mesh(pv.wrap(target), style='wireframe')

            p.add_slider_widget(
                slider_closure(records=records, pv_mesh=pv_mesh),
                rng=(0, len(records)), value=0,
                color='black',
                event_type='always',
                title='step')

            for pos in target_markers_vertices:
                p.add_mesh(pv.Sphere(target.scale / 200, pos), color='g')

            p.show()

    except ImportError:
        source.vertices = records_amberg[-1]
        source.show()
        source.vertices = records_sumner[-1]
        source.show()
