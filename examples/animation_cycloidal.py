"""
animation_cycloidal.py
------------------------

A short film of a cycloidal drive from a CAD assembly, made entirely of
scene graph animation so it saves as one animated glTF which plays in
any viewer, and renders to video.

A cycloidal drive is a reducer: three crankshafts push a pair of lobed
discs around a ring of 20 rollers, and as a disc has one fewer lobe than
there are rollers the output crawls backwards one roller per crank turn.
Here a turning stack of a disc's glowing outlines falls into one plane
as the drive grows out of it under a glass cover, the drive runs while
the camera circles it in space, then its vertices burst out spinning
faster and faster and leave it running alone: lines, meshes, points and
animation, all in one glTF.

Usage: `python animation_cycloidal.py [output directory]`
"""

import os
import sys

import numpy as np

import trimesh
from trimesh.scene.animation import Animation
from trimesh.scene.metadata import Environment, RenderSettings, Softbox
from trimesh.transformations import euler_matrix, rotation_matrix
from trimesh.visual.material import PBRMaterial

# which parts move, keyed by geometry name: the cranks and the bearings
# on their journals turn, the discs orbit and the output carries them
CRANKS = {"camshaft", "vxb-6800-2rs"}
DISCS = {"disc_cam_A", "disc_cam_B"}
OUTPUT = {"output", "60355K430", "92320A138_spacer"}
ROLLERS = 20
# the cover over the mechanism is glass, the outlines glow orange,
# colored parts are anodized and everything else is polished steel
GLASS = "fixed_lower"
GOLD = [255, 170, 40]
GLOW = [255, 150, 30, 255]
COLORS = {
    "disc_cam_A": [40, 120, 255],
    "disc_cam_B": [255, 40, 160],
    "output": GOLD,
    "fixed_top": [70, 75, 90],
}

# space is black but for a blue and a magenta nebula, a soft glow from
# above and a small bright sun, which the metal reflects in color: as a
# direction from the drive, sharpness, and linear RGB brightness
SPACE = [
    ([-0.6, 0.3, 0.7], 12.0, [1.0, 2.5, 10.0]),
    ([0.7, -0.5, 0.2], 16.0, [10.0, 1.5, 6.0]),
    ([0.0, 0.0, 1.0], 3.0, [0.12, 0.12, 0.16]),
    ([0.3, 0.6, 0.8], 300.0, [80.0, 70.0, 55.0]),
]

# the storyboard in seconds, where the output turns once in the film
DURATION = 14.0
CRANK_SPEED = 2.0 * np.pi * ROLLERS / DURATION
# the outlines fall into one plane while the drive grows out of it
STACK, GROW = (1.0, 3.0), (2.0, 3.5)
# the vertices burst out and are gone
BURST = (7.0, 9.5)
# how far the cloud grows and spins, and how the outlines stack
GROWTH, SWIRL, COPIES, GAP = 100.0, 6.0 * np.pi, 10, 9.0


def smoothstep(times, start, stop):
    """
    Zero before `start`, one after `stop` and eased between.

    Parameters
    ------------
    times : (k,) float
      Seconds.
    start : float
      When to start rising.
    stop : float
      When to arrive at one.

    Returns
    ------------
    eased : (k,) float
      From 0.0 to 1.0.
    """
    fraction = np.clip((times - start) / (stop - start), 0.0, 1.0)
    return fraction * fraction * (3.0 - 2.0 * fraction)


def animate(scene, fps=30):
    """
    Turn the cycloidal drive from `models/cycloidal.3DXML` into the
    film, in place.

    Parameters
    ------------
    scene : trimesh.Scene
      The assembly, whose parts are children of a node at the world.
    fps : float
      Keyframes per second, at least 3 so a crank never turns more
      than half a revolution between keyframes.

    Returns
    ------------
    scene : trimesh.Scene
      The same scene, shiny, in space, with its animation.
    """
    graph = scene.graph
    times = np.arange(round(DURATION * fps) + 1) / fps
    crank = CRANK_SPEED * times
    nodes = list(graph.nodes_geometry)
    parents = [graph.transforms.parents[node] for node in nodes]
    parts = np.array([graph[node][1] for node in nodes])
    # every part's edge from its parent, which is where the world is here
    local = np.array([graph.get(node, parent)[0] for node, parent in zip(nodes, parents)])
    centers = np.array([scene.geometry[part].centroid for part in parts])
    centers = np.matmul(local[:, :3, :3], centers[..., None])[..., 0] + local[:, :3, 3]
    center = scene.centroid
    # every vertex of the drive under the glass, finely spaced, as a
    # particle colored by its angle around the axis
    solid = scene.copy()
    solid.delete_geometry(GLASS)
    vertices = solid.to_mesh().subdivide_to_size(scene.extents.max() / 100).vertices
    points = vertices[np.random.default_rng(0).choice(len(vertices), 200000)]
    offset = points - center
    hue = np.arctan2(offset[:, 1], offset[:, 0]) / (2.0 * np.pi) + 0.5
    colors = trimesh.visual.color.hsv_to_rgba(
        np.column_stack((hue, np.full(len(hue), 0.8), np.ones(len(hue))))
    )

    # the outline of a disc's biggest flat face as lines
    disc = int(np.flatnonzero(parts == "disc_cam_A")[0])
    profile = scene.geometry["disc_cam_A"]
    outline = profile.outline(profile.facets[profile.facets_area.argmax()])
    outline.colors = np.tile(GLOW, (len(outline.entities), 1))

    # shiny parts in space, which the camera sees too
    for part, geometry in scene.geometry.items():
        geometry.visual = trimesh.visual.TextureVisuals(
            material=PBRMaterial(
                baseColorFactor=COLORS.get(part, [200, 205, 215]),
                metallicFactor=1.0,
                roughnessFactor=0.2 + 0.15 * (part in COLORS),
            )
        )
    scene.add_geometry(outline, geom_name="outline", node_name="outline_0")
    for index in range(1, COPIES):
        graph.update(frame_to=f"outline_{index}", geometry="outline")
    scene.geometry[GLASS].visual.material = PBRMaterial(
        baseColorFactor=[200, 220, 255],
        metallicFactor=0.0,
        roughnessFactor=0.05,
        transmissionFactor=0.9,
    )
    scene.metadata.update(
        RenderSettings(
            environment=Environment(
                up=[0, 0, 1],
                sky=[0.01, 0.01, 0.03],
                ground=[0.0, 0.0, 0.0],
                softboxes=[
                    Softbox(direction=direction, sharpness=sharpness, color=color)
                    for direction, sharpness, color in SPACE
                ],
            ),
            ground_shadow=False,
            show_environment=True,
            # bold lines and fine particles
            line_width=3.0,
            point_size=3.0,
        ).to_metadata()
    )

    # the drive: the cranks turn on their own axes, the discs slide around
    # a circle as big as their eccentricity, and the output carries it all
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

    # the outlines stack in a twisting tower above the moving disc which
    # falls into its plane, and glTF hides things by scaling them to nothing
    stack = 1.0 - smoothstep(times, *STACK)
    rise = stack[:, None] * np.arange(1, COPIES + 1)
    outlines = rotation_matrix(rise * 0.25, z, centers[disc]) @ matrices[:, [disc]]
    outlines[..., 2, 3] += rise * GAP
    outlines[stack <= 0.0, :, :3, :3] = 0.0

    # the drive grows out of the outlines' plane
    plane = trimesh.transform_points(outline.vertices[:1], local[disc])[0, 2]
    height = smoothstep(times, *GROW)
    extrude = np.tile(np.eye(4), (len(times), 1, 1))
    extrude[:, 2, 2] = height
    extrude[:, 2, 3] = plane * (1.0 - height)
    extrude[height <= 0.0, :3, :3] = 0.0
    matrices = extrude[:, None] @ matrices

    # the drive's vertices turning with it burst out from rest faster and
    # faster, spinning, past the camera until they are gone
    scene.add_geometry(trimesh.PointCloud(points, colors=colors), node_name="cloud")
    burst = np.interp(times, BURST, [0.0, 1.0]) ** 3
    size = 1.0 + GROWTH * burst
    size[(times <= BURST[0]) | (times >= BURST[1])] = 0.0
    cloud = np.tile(np.eye(4), (len(times), 1, 1))
    cloud[:, :3, :3] *= size[:, None, None]
    cloud[:, :3, 3] = center * (1.0 - size[:, None])
    cloud = rotation_matrix(-crank / ROLLERS - SWIRL * burst, z) @ cloud

    # the camera starts on the whole tower and settles on the drive as it
    # falls, while circling once
    rotation = euler_matrix(*np.radians([70.0, 0.0, 25.0]))
    tower = trimesh.transform_points(outline.bounds, outlines[0, -1])
    framed = [
        trimesh.scene.cameras.look_at(np.vstack(points), scene.camera.fov, rotation)
        for points in ([tower, scene.bounds], [scene.bounds])
    ]
    settle = smoothstep(times, *STACK)[:, None, None]
    camera = rotation_matrix(2.0 * np.pi * times / DURATION, z, center) @ (
        framed[0] * (1.0 - settle) + framed[1] * settle
    )

    scene.camera_transform = camera[0]
    # deep enough to see the whole cloud until it has flown past
    scene.camera.z_far = 2.0 * GROWTH * scene.scale
    scene.animations.extend(
        Animation(edges, times, values, name="cycloidal")
        for edges, values in [
            (list(zip(parents, nodes)), matrices),
            ([(None, "cloud")], cloud[:, None]),
            ([(None, f"outline_{index}") for index in range(COPIES)], outlines),
            ([(None, scene.camera.name)], camera[:, None]),
        ]
    )
    # the scene at rest is the first frame, which is what a viewer that
    # isn't playing the animation shows
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
    # an animated glTF which plays in Blender, three.js and trimesh, and
    # the same film from the built in renderer
    scene.export(os.path.join(directory, "cycloidal.glb"))
    scene.export(os.path.join(directory, "cycloidal.mp4"), resolution=[1920, 1080])
