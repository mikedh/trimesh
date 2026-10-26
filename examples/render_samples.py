"""
render_samples.py
-------------------

Render the animated glTF sample models and the examples here to 4K
video with trimesh's own OpenGL renderer. Every video is one line:
`trimesh.load_remote(url).render("film.mp4")`.

Usage: `python render_samples.py [directory]`
"""

import os
import sys
import time

import animation_cycloidal
import pyrender_scene

import trimesh

# the Khronos glTF sample models which are animated
SAMPLES = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models"
ANIMATED = [
    "AnimatedColorsCube",
    "AnimatedMorphCube",
    "BoxAnimated",
    "BrainStem",
    "CesiumMan",
    "CesiumMilkTruck",
    "ChronographWatch",
    "CommercialRefrigerator",
    "DiffuseTransmissionPlant",
    "Fox",
    "InterpolationTest",
    "IridescentDishWithOlives",
    "MorphStressTest",
    "RecursiveSkeletons",
    "RiggedFigure",
    "RiggedSimple",
    "VirtualCity",
]
FPS = 30


def load(name):
    """A scene by name: one of the examples here or a sample model."""
    if name == "cycloidal":
        here = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(here, "..", "models", "cycloidal.3DXML")
        return animation_cycloidal.animate(trimesh.load_scene(path))
    if name == "pyrender":
        return pyrender_scene.build(trimesh.resolvers.WebResolver(pyrender_scene.MODELS))
    return trimesh.load_remote(f"{SAMPLES}/{name}/glTF-Binary/{name}.glb")


if __name__ == "__main__":
    directory = os.getcwd()
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    # the examples' render settings, and the defaults for the samples
    settings = {
        "cycloidal": animation_cycloidal.RENDER,
        "pyrender": pyrender_scene.RENDER,
    }
    for name in [*ANIMATED, "cycloidal", "pyrender"]:
        scene = load(name)
        path = os.path.join(directory, f"{name}_4k.mp4")
        start = time.perf_counter()
        scene.render(path, resolution=(3840, 2160), fps=FPS, **settings.get(name, {}))
        seconds = time.perf_counter() - start
        frames = max(round(max(scene.duration, 4.0) * FPS), 1)
        print(f"{os.path.basename(path):48s} {frames / seconds:6.1f} fps")
