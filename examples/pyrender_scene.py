"""
pyrender_scene.py
-------------------

The scene from pyrender's `examples/example.py` built with only trimesh
and its models streamed straight from pyrender's repository, saved as a
GLB which any glTF viewer shows with the same lights, camera and
materials, less pyrender's ambient light which only trimesh's viewer draws.

| pyrender                          | trimesh                                    |
|-----------------------------------|--------------------------------------------|
| `Mesh.from_trimesh(mesh)`         | `scene.add_geometry(mesh)`                 |
| `scene.add(mesh, pose=pose)`      | `scene.add_geometry(mesh, transform=pose)` |
| `DirectionalLight`, `SpotLight`   | `trimesh.scene.lighting.*Light`            |
| `PerspectiveCamera(yfov=...)`     | `trimesh.scene.Camera(fov=...)`            |
| `Scene(ambient_light=...)`        | `EnvironmentLight(cubemap=...)`            |
| `OffscreenRenderer.render(scene)` | `SceneViewer(...).to_array(...)`           |

Usage: `python pyrender_scene.py [output directory]`
"""

import os
import sys

import numpy as np
from PIL import Image

import trimesh
from trimesh.resolvers import WebResolver
from trimesh.scene import lighting
from trimesh.viewer.opengl.viewer import SceneViewer

# pyrender's models, which are loaded with the files they reference
MODELS = "https://raw.githubusercontent.com/mmatl/pyrender/master/examples/models/"
# pyrender renders 1280x960 with a 60 degree vertical fov
RESOLUTION = [1280, 960]
# the wood is the floor so there is no need for a shadow catcher
RENDER = {"ground_shadow": False}
# where the camera and both lights are
CAMERA_POSE = np.array(
    [
        [0.0, -np.sqrt(2) / 2, np.sqrt(2) / 2, 0.5],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0.4],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def build(resolver, seed=0):
    """
    The scene from pyrender's example.

    Parameters
    ------------
    resolver : trimesh.resolvers.Resolver
      Has `fuze.obj`, `drill.obj`, `wood.obj`, `WaterBottle.glb` and
      whatever they reference.
    seed : int
      For the random colors of the boxes.

    Returns
    ------------
    scene : trimesh.Scene
      The meshes, lights, camera and an almost black environment.
    """

    def load(name):
        # a model and its materials and textures from the resolver
        return trimesh.load_scene(
            trimesh.util.wrap_as_stream(resolver.get(name)),
            file_type=name.split(".")[-1],
            resolver=resolver,
        ).to_mesh()

    random = np.random.default_rng(seed)
    fuze, drill, wood, bottle = map(
        load, ["fuze.obj", "drill.obj", "wood.obj", "WaterBottle.glb"]
    )
    vertex_box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    vertex_box.visual.vertex_colors = random.uniform(size=vertex_box.vertices.shape)
    face_box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    face_box.visual.face_colors = random.uniform(size=face_box.faces.shape)

    scene = trimesh.Scene()
    shift = trimesh.transformations.translation_matrix
    for name, mesh, pose in [
        ("fuze", fuze, shift([0.1, 0.15, -fuze.bounds[0, 2]])),
        ("drill", drill, shift([0.1, 0.0, -drill.bounds[0, 2]])),
        ("wood", wood, np.eye(4)),
        (
            "bottle",
            bottle,
            np.array([[1, 0, 0, 0.1], [0, 0, -1, -0.16], [0, 1, 0, 0.13], [0, 0, 0, 1]]),
        ),
        ("vertex_box", vertex_box, shift([-0.1, 0.1, 0.05])),
        ("face_box", face_box, shift([-0.1, -0.1, 0.05])),
    ]:
        scene.add_geometry(mesh, node_name=name, transform=pose)

    # pyrender's ambient light is the same color in every direction
    ambient = [0.02, 0.02, 0.02]
    # all lights are where the camera is, in glTF units like pyrender
    scene.lights = [
        lighting.DirectionalLight(name="sun", intensity=1.0),
        lighting.SpotLight(
            name="spot",
            intensity=10.0,
            innerConeAngle=np.pi / 16,
            outerConeAngle=np.pi / 6,
        ),
        lighting.EnvironmentLight(
            name="ambient", cubemap=lighting.studio_cubemap(ambient, ambient)
        ),
    ]
    for light in scene.lights:
        scene.graph.update(frame_to=light.name, matrix=CAMERA_POSE)
    vertical = np.pi / 3.0
    horizontal = 2.0 * np.arctan(np.tan(vertical / 2.0) * RESOLUTION[0] / RESOLUTION[1])
    scene.camera = trimesh.scene.Camera(
        resolution=RESOLUTION, fov=np.degrees([horizontal, vertical])
    )
    scene.camera_transform = CAMERA_POSE
    return scene


if __name__ == "__main__":
    directory = os.getcwd()
    if len(sys.argv) > 1:
        directory = sys.argv[1]

    scene = build(WebResolver(MODELS))
    scene.export(os.path.join(directory, "pyrender_scene.glb"))

    viewer = SceneViewer(
        scene, visible=False, start_loop=False, resolution=RESOLUTION, **RENDER
    )
    color, depth, segmentation = viewer.to_array(
        include_depth=True, include_segmentation=True
    )
    viewer.close()
    Image.fromarray(color).save(os.path.join(directory, "pyrender_color.png"))
    # nearer is brighter and the background is black
    near = depth > 0.0
    brightness = np.zeros(depth.shape, dtype=np.uint8)
    brightness[near] = np.interp(depth[near], [depth[near].min(), depth.max()], [255, 64])
    Image.fromarray(brightness).save(os.path.join(directory, "pyrender_depth.png"))
    # every node a different gray like pyrender's example
    gray = ((segmentation + 1) * 20 % 256).astype(np.uint8)
    Image.fromarray(gray).save(os.path.join(directory, "pyrender_segmentation.png"))
    # and as a turntable gif, with only Pillow
    scene.render(
        os.path.join(directory, "pyrender_scene.gif"), resolution=[640, 480], **RENDER
    )
