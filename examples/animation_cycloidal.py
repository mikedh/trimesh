"""
animation_cycloidal.py
------------------------

A cycloidal drive from a CAD assembly animated with the scene graph,
saved as one animated glTF which plays in any viewer and rendered to video.

Three crankshafts push a pair of lobed discs around a ring of 20 rollers,
and as a disc has one fewer lobe than there are rollers the output turns
backwards one roller per crank turn.

Usage: `python animation_cycloidal.py [output directory]`
"""

import os
import sys

import numpy as np

import trimesh
from trimesh.scene.animation import Animation
from trimesh.transformations import euler_matrix, rotation_matrix

# which parts move, by geometry name: the cranks and the bearings on their
# journals turn, the discs orbit and the output carries them
CRANKS = {"camshaft", "vxb-6800-2rs"}
DISCS = {"disc_cam_A", "disc_cam_B"}
OUTPUT = {"output", "60355K430", "92320A138_spacer"}
ROLLERS = 20
# the output turns once in the film
DURATION = 14.0
CRANK_SPEED = 2.0 * np.pi * ROLLERS / DURATION


def animate(scene, fps=30):
    """
    Animate the cycloidal drive from `models/cycloidal.3DXML` in place.

    Parameters
    ------------
    scene : trimesh.Scene
      The assembly, whose parts are children of a node at the world.
    fps : float
      Keyframes per second, at least 3 so a crank turns less than half
      a revolution between keyframes.

    Returns
    ------------
    scene : trimesh.Scene
      The same scene with its animation.
    """
    graph = scene.graph
    times = np.arange(round(DURATION * fps) + 1) / fps
    crank = CRANK_SPEED * times
    nodes = list(graph.nodes_geometry)
    parts = np.array([graph[node][1] for node in nodes])
    local = np.array(
        [graph.get(node, graph.transforms.parents[node])[0] for node in nodes]
    )
    centers = np.array([scene.geometry[part].centroid for part in parts])
    centers = np.matmul(local[:, :3, :3], centers[..., None])[..., 0] + local[:, :3, 3]

    # the cranks turn on their own axes, the discs slide around a circle
    # as big as their eccentricity, and the output carries it all
    z = [0.0, 0.0, 1.0]
    matrices = np.tile(local, (len(times), 1, 1, 1))
    is_crank = np.isin(parts, list(CRANKS))
    axes = local[parts == "camshaft", :3, 3]
    distance = np.linalg.norm(centers[is_crank, None, :2] - axes[None, :, :2], axis=2)
    nearest = axes[distance.argmin(axis=1)]
    matrices[:, is_crank] = (
        rotation_matrix(crank[:, None], z, nearest[None]) @ matrices[:, is_crank]
    )
    is_disc = np.isin(parts, list(DISCS))
    eccentric = centers[is_disc] * [1.0, 1.0, 0.0]
    orbit = np.matmul(rotation_matrix(crank, z)[:, None, :3, :3], eccentric[..., None])
    moved = matrices[:, is_disc]
    moved[..., :3, 3] += orbit[..., 0] - eccentric
    matrices[:, is_disc] = moved
    carried = is_crank | is_disc | np.isin(parts, list(OUTPUT))
    matrices[:, carried] = (
        rotation_matrix(-crank[:, None] / ROLLERS, z) @ matrices[:, carried]
    )

    # the camera circles the drive once
    rotation = euler_matrix(*np.radians([70.0, 0.0, 25.0]))
    start = trimesh.scene.cameras.look_at(scene.bounds, scene.camera.fov, rotation)
    camera = rotation_matrix(2.0 * np.pi * times / DURATION, z, scene.centroid) @ start
    scene.camera_transform = camera[0]

    scene.animations.append(
        Animation(
            name="cycloidal",
            nodes=[*nodes, scene.camera.name],
            times=times,
            matrices=np.concatenate((matrices, camera[:, None]), axis=1),
        )
    )
    # a viewer which isn't playing shows the first frame
    scene.animate(0.0)
    return scene


if __name__ == "__main__":
    directory = os.getcwd()
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    here = os.path.abspath(os.path.dirname(__file__))
    scene = animate(
        trimesh.load_scene(os.path.join(here, "..", "models", "cycloidal.3DXML"))
    )
    # an animated glTF which plays in Blender, three.js and trimesh,
    # and the same film from the built in renderer
    scene.export(os.path.join(directory, "cycloidal.glb"))
    scene.render(os.path.join(directory, "cycloidal.mp4"), resolution=[1920, 1080])
