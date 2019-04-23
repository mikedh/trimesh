"""
Glooey widget example. Only runs with python>=3.6
"""

import pathlib

import glooey
import numpy as np

import pyglet
import trimesh
import trimesh.viewer
import trimesh.transformations as tf


here = pathlib.Path(__file__).resolve().parent


def create_scene1():
    """
    Create a scene with a Fuze bottle, some cubes, and an axis.

    Returns
    ----------
    scene : trimesh.Scene
      Object with geometry
    """
    scene = trimesh.Scene()

    # plane
    geom = trimesh.creation.box((0.5, 0.5, 0.01))
    geom.apply_translation((0, 0, -0.005))
    geom.visual.face_colors = (.6, .6, .6)
    scene.add_geometry(geom)

    # axis
    geom = trimesh.creation.axis(0.02)
    scene.add_geometry(geom)

    box_size = 0.1

    # box1
    geom = trimesh.creation.box((box_size,) * 3)
    geom.visual.face_colors = np.random.uniform(
        0, 1, (len(geom.faces), 3))
    transform = tf.translation_matrix([0.1, 0.1, box_size / 2])
    scene.add_geometry(geom, transform=transform)

    # box2
    geom = trimesh.creation.box((box_size,) * 3)
    geom.visual.face_colors = np.random.uniform(
        0, 1, (len(geom.faces), 3))
    transform = tf.translation_matrix([-0.1, 0.1, box_size / 2])
    scene.add_geometry(geom, transform=transform)

    # fuze
    geom = trimesh.load(str(here / '../models/fuze.obj'))
    transform = tf.translation_matrix([-0.1, -0.1, 0])
    scene.add_geometry(geom, transform=transform)

    # sphere
    geom = trimesh.creation.icosphere(radius=0.05)
    geom.visual.face_colors = np.random.uniform(
        0, 1, (len(geom.faces), 3))
    transform = tf.translation_matrix([0.1, -0.1, box_size / 2])
    scene.add_geometry(geom, transform=transform)

    return scene


def create_scene2():
    """
    Create a scene with a bunch of small icospheres
    """
    scene = trimesh.Scene()

    geom = trimesh.path.creation.box_outline((0.6, 0.6, 0.6))
    scene.add_geometry(geom)

    for _ in range(50):
        geom = trimesh.creation.icosphere(radius=0.01, subdivisions=2)
        geom.visual.face_colors = np.random.uniform(0, 1, (3,))
        eye = np.random.uniform(-0.3, 0.3, (3,))
        transform = tf.translation_matrix(eye)
        scene.add_geometry(geom, transform=transform)

    scene.set_camera(angles=[np.deg2rad(60), 0, 0], distance=1.5)

    return scene


if __name__ == '__main__':
    np.random.seed(0)

    # create a pyglet window
    window = pyglet.window.Window(width=1280, height=480)
    # create a glooey interface using the window
    gui = glooey.Gui(window)

    hbox = glooey.HBox()
    hbox.set_padding(5)

    # define a callback which will spin the scene
    def callback():
        if not hasattr(widget1, '_angles'):
            widget1._angles = [np.deg2rad(45), 0, 0]
        widget1._angles[2] += np.deg2rad(1)
        widget1.scene.set_camera(angles=widget1._angles, distance=1)
        widget1._draw()

    # make a widget with one scene
    scene = create_scene1()
    widget1 = trimesh.viewer.SceneWidget(scene)
    hbox.add(widget1)

    # make a widget with the other scene
    scene = create_scene2()
    widget2 = trimesh.viewer.SceneWidget(scene)
    hbox.add(widget2)

    # add the boxes with the widgets to the gui
    gui.add(hbox)

    # schedule a callback to spin the scene
    pyglet.clock.schedule_interval(lambda dt: callback(), 1 / 20)

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.Q:
            window.close()

    pyglet.app.run()
