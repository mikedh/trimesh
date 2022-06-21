"""
examples/nricp.py
------------------------

Register a ball to a teeth
"""

import trimesh
import numpy as np
from trimesh.registration import nricp


if __name__ == '__main__':

    # attach to trimesh logs
    trimesh.util.attach_to_log()

    tooth = trimesh.load_mesh('../models/busted.STL', process=False)
    ball = trimesh.load_mesh('../models/ballA.off', process=False)

    tooth.vertices = ball.scale * (tooth.vertices - tooth.centroid[None, :])\
        / tooth.scale + ball.centroid

    steps = [
        [0.02, 3, 0.5, 10],
        [0.007, 0.0, 0.5, 10],
        [0.002, 0.0, 0.0, 10],
    ]
    ball_landmarks = np.array([0], dtype=np.int32)
    tooth_landmarks = np.array([0], dtype=np.int32)
    result, records = nricp(ball, tooth, source_landmarks=ball_landmarks,
                            target_landmarks=tooth_landmarks, steps=steps,
                            return_records=True)
    try:
        import pyvista as pv
        p = pv.Plotter()
        p.background_color = 'w'
        pv_mesh = pv.wrap(ball)
        pv_mesh['scalars'] = records[0][1]
        p.add_mesh(pv_mesh, color=(0.6, 0.6, 0.9), scalars='scalars',
                   cmap='rainbow', clim=(0, 0.02))
        p.add_mesh(pv.wrap(tooth), style='wireframe')

        def cb(value):
            idx = min(int(value), len(records) - 1)
            pv_mesh.points = records[idx][0]
            pv_mesh['scalars'] = records[idx][1]
        p.add_slider_widget(cb, rng=(0, len(records)), value=0,
                            color='black', event_type='always')
        p.show()
    except ImportError:
        result.show()
