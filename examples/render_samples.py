"""
render_samples.py
-------------------

Render the animated glTF sample models and the examples here to video
with trimesh's own OpenGL renderer, and side by side with Blender. Every
video is one line: `trimesh.load_remote(url).export("film.mp4")`.

Usage:
  `python render_samples.py samples [directory]`: every one at 4K
  `python render_samples.py reel [directory]`: a one minute reel of
    glb animations rendered by pyglet and by Blender, with frame rates
"""

import itertools
import os
import subprocess
import sys
import time
from dataclasses import replace

import animation_cycloidal
import numpy as np
import pyglet
import pyrender_scene
from PIL import Image, ImageDraw, ImageFont

import trimesh
from trimesh.interfaces.ffmpeg import VideoSettings, VideoWriter
from trimesh.scene.cameras import current_view, look_at
from trimesh.scene.metadata import RenderSettings
from trimesh.transformations import rotation_matrix
from trimesh.viewer.render import render_frames, turntable

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
# the reel as a name, seconds to play, the animation to play when a
# file has several which move the same nodes, and whether the camera
# orbits as well, where short animations loop
REEL = [
    ("cycloidal", 12.0, None, False),
    ("BrainStem", 8.0, None, False),
    ("DiffuseTransmissionPlant", 9.0, None, False),
    ("pyrender", 6.0, None, False),
    ("Fox", 5.0, "Run", False),
    ("CesiumMan", 5.0, None, False),
    ("ChronographWatch", 6.0, None, False),
    ("CesiumMilkTruck", 6.0, None, True),
]
# each engine renders half of the 3840x2160 reel, and OpenGL renders it
# bigger and averages it down for crisp edges where Blender samples each pixel
WIDTH, HEIGHT, SUPERSAMPLE, TITLE = 1920, 2160, 2, 3.0
SETTINGS = VideoSettings(fps=30)
# the fonts the reel's labels and code are set in
FONT, CODE = (
    subprocess.run(
        ["fc-match", "-f", "%{file}", name], capture_output=True, text=True
    ).stdout
    for name in ("IBM Plex Sans:medium", "IBM Plex Mono:medium")
)


def load(name):
    """A scene by name: one of the examples here or a sample model."""
    if name == "cycloidal":
        here = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(here, "..", "models", "cycloidal.3DXML")
        return animation_cycloidal.animate(trimesh.load_scene(path))
    if name == "pyrender":
        return pyrender_scene.orbit(
            pyrender_scene.build(trimesh.resolvers.WebResolver(pyrender_scene.MODELS)),
            6.0,
        )
    return trimesh.load_remote(f"{SAMPLES}/{name}/glTF-Binary/{name}.glb")


def clips(scene):
    """The scene once per animation which moves the same nodes as another."""
    groups = {}
    for animation in scene.animations:
        groups.setdefault(animation.name, []).append(animation)
    edges = {name: {e for a in group for e in a.edges} for name, group in groups.items()}
    if not any(edges[a] & edges[b] for a, b in itertools.combinations(edges, 2)):
        return {None: scene}
    result = {}
    for name, group in groups.items():
        result[name] = scene.copy()
        result[name].animations = list(group)
    return result


def samples(directory):
    """Every animated sample model and example at 4K."""
    for name in [*ANIMATED, "cycloidal", "pyrender"]:
        for clip, scene in clips(load(name)).items():
            path = os.path.join(directory, f"{name}_{clip or 'film'}_4k.mp4")
            start = time.perf_counter()
            scene.export(path, resolution=(3840, 2160), settings=SETTINGS)
            seconds = time.perf_counter() - start
            frames = max(round(max(scene.duration, 4.0) * SETTINGS.fps), 1)
            print(f"{os.path.basename(path):48s} {frames / seconds:6.1f} fps")


def label(height, words, font=FONT):
    """Outlined words across the reel as (height, width, 4) float."""
    strip = Image.new("RGBA", (2 * WIDTH, height), (0, 0, 0, 0))
    font = ImageFont.truetype(font, height // 2)
    for center, text in words:
        # white with a dark edge reads over a white or a black background
        ImageDraw.Draw(strip).text(
            (center, height // 2),
            text,
            font=font,
            fill="white",
            anchor="mm",
            stroke_width=max(height // 30, 1),
            stroke_fill=(20, 20, 30),
        )
    return np.asarray(strip).astype(np.float64)


def overlay(frame, strip, row):
    """Blend a label strip over a float frame starting at a row."""
    alpha = strip[..., 3:] / 255.0
    rows = slice(row, row + len(strip))
    frame[rows] = frame[rows] * (1.0 - alpha) + strip[..., :3] * alpha


def reel(directory):
    """trimesh's renderer and Blender side by side through the same camera."""
    renderer = f"pyglet {pyglet.version.split('.')[0]}"
    # label heights and margins as a fraction of the reel's height
    margin = HEIGHT // 50
    with VideoWriter(os.path.join(directory, "reel.mp4"), SETTINGS) as video:
        card = np.zeros((HEIGHT, 2 * WIDTH, 3))
        title = "trimesh: glb animations rendered in pyglet or blender"
        overlay(card, label(HEIGHT // 8, [(WIDTH, title)]), HEIGHT // 3)
        code = 'scene.export("film.mp4", engine="blender")'
        overlay(card, label(HEIGHT // 12, [(WIDTH, code)], font=CODE), HEIGHT // 2)
        for _ in range(round(TITLE * SETTINGS.fps)):
            video.write(card.astype(np.uint8))

        for name, duration, animation, orbit in REEL:
            scene = clips(load(name))[animation]
            if orbit:
                # the camera a render frames the scene with, backed away to
                # fit it from every side, circling once
                camera, start = current_view(scene, resolution=(WIDTH, HEIGHT))
                axes = np.vstack((np.eye(3), -np.eye(3)))
                up = axes[np.argmax(axes @ start[:3, 1])]
                corners = trimesh.bounds.corners(scene.bounds)
                sides = [
                    trimesh.transform_points(
                        corners, rotation_matrix(angle, up, scene.centroid)
                    )
                    for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
                ]
                scene.camera = camera
                scene.camera_transform = look_at(np.vstack(sides), camera.fov, start)
                scene.animations.append(turntable(scene, duration, SETTINGS.fps))
            # lines and points are as many more pixels across in the
            # bigger OpenGL render, so they're the same once averaged
            bigger = scene.copy()
            stored = RenderSettings.from_metadata(scene.metadata)
            bigger.metadata.update(
                replace(
                    stored,
                    line_width=stored.line_width * SUPERSAMPLE,
                    point_size=stored.point_size * SUPERSAMPLE,
                ).to_metadata()
            )
            # time only the rendering: OpenGL frames are averaged down as
            # they come, and Blender renders the whole clip at once
            left, opengl_seconds = [], 0.0
            frames = iter(
                render_frames(
                    bigger,
                    resolution=(WIDTH * SUPERSAMPLE, HEIGHT * SUPERSAMPLE),
                    duration=duration,
                    fps=SETTINGS.fps,
                )
            )
            while True:
                start = time.perf_counter()
                frame = next(frames, None)
                opengl_seconds += time.perf_counter() - start
                if frame is None:
                    break
                block = frame[..., :3].reshape(
                    (HEIGHT, SUPERSAMPLE, WIDTH, SUPERSAMPLE, 3)
                )
                left.append(block.mean(axis=(1, 3)).round().astype(np.uint8))
            start = time.perf_counter()
            right = render_frames(
                scene,
                resolution=(WIDTH, HEIGHT),
                engine="blender",
                duration=duration,
                fps=SETTINGS.fps,
            )
            blender_seconds = time.perf_counter() - start

            title = label(HEIGHT // 14, [(WIDTH, name)])
            engines = label(
                HEIGHT // 11,
                [
                    (WIDTH // 2, f"{renderer}: {len(left) / opengl_seconds:.0f} fps"),
                    (
                        WIDTH + WIDTH // 2,
                        f"blender: {len(right) / blender_seconds:.1f} fps",
                    ),
                ],
            )
            for opengl, blender in zip(left, right):
                frame = np.hstack((opengl, blender[..., :3])).astype(np.float64)
                overlay(frame, title, margin)
                overlay(frame, engines, HEIGHT - len(engines) - margin)
                video.write(frame.round().astype(np.uint8))
            print(
                f"{name:28s} {len(left) / opengl_seconds:6.0f} and {len(right) / blender_seconds:5.2f} fps"
            )


if __name__ == "__main__":
    directory = os.getcwd()
    if len(sys.argv) > 2:
        directory = sys.argv[2]
    {"samples": samples, "reel": reel}[sys.argv[1]](directory)
