"""
view_callback.py
------------------

Show how to pass a callback to the scene viewer for
easy visualizations.
"""

import time
import trimesh
import numpy as np


def sinwave(scene):
    """
    A callback passed to a scene viewer which will update
    transforms in the viewer periodically.

    Parameters
    -------------
    scene : trimesh.Scene
      Scene containing geometry

    """
    # take a node
    node = s.graph.nodes_geometry[0]

    matrix = np.eye(4)
    matrix[2][3] = np.sin(time.time()) * 3
    scene.graph.update(node, matrix=matrix)


if __name__ == '__main__':
    # create some spheres
    a = trimesh.primitives.Sphere()
    b = trimesh.primitives.Sphere()

    # set some colors for the balls
    a.visual.face_colors = [255, 0, 0, 255]
    b.visual.face_colors = [0, 0, 100, 255]

    # create a scene with the two balls
    s = trimesh.Scene([a, b])

    # open the scene viewer and move a ball around
    s.show(callback=sinwave)
