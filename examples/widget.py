"""
Glooey widget example. Only runs with python>=3.6 and pyglet<=1.5.27 (because of Glooey).
"""

import io
import pathlib

import glooey
import numpy as np

import pyglet
import trimesh
import trimesh.viewer
import trimesh.transformations as tf
import PIL.Image


here = pathlib.Path(__file__).resolve().parent


def create_scene():
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


class Application:

    """
    Example application that includes moving camera, scene and image update.
    """

    def __init__(self):
        # create window with padding
        self.width, self.height = 480 * 3, 360
        window = self._create_window(width=self.width, height=self.height)

        gui = glooey.Gui(window)

        hbox = glooey.HBox()
        hbox.set_padding(5)

        # scene widget for changing camera location
        scene = create_scene()
        self.scene_widget1 = trimesh.viewer.SceneWidget(scene)
        self.scene_widget1._angles = [np.deg2rad(45), 0, 0]
        hbox.add(self.scene_widget1)

        # scene widget for changing scene
        scene = trimesh.Scene()
        geom = trimesh.path.creation.box_outline((0.6, 0.6, 0.6))
        scene.add_geometry(geom)
        self.scene_widget2 = trimesh.viewer.SceneWidget(scene)
        hbox.add(self.scene_widget2)

        # integrate with other widget than SceneWidget
        self.image_widget = glooey.Image()
        hbox.add(self.image_widget)

        gui.add(hbox)

        pyglet.clock.schedule_interval(self.callback, 1. / 20)
        pyglet.app.run()

    def callback(self, dt):
        # change camera location
        self.scene_widget1._angles[2] += np.deg2rad(1)
        self.scene_widget1.scene.set_camera(self.scene_widget1._angles)

        # change scene
        if len(self.scene_widget2.scene.graph.nodes) < 100:
            geom = trimesh.creation.icosphere(radius=0.01)
            geom.visual.face_colors = np.random.uniform(0, 1, (3,))
            geom.apply_translation(np.random.uniform(-0.3, 0.3, (3,)))
            self.scene_widget2.scene.add_geometry(geom)
            self.scene_widget2._draw()

        # change image
        image = np.random.randint(0,
                                  255,
                                  (self.height - 10, self.width // 3 - 10, 3),
                                  dtype=np.uint8)
        with io.BytesIO() as f:
            PIL.Image.fromarray(image).save(f, format='JPEG')
            self.image_widget.image = pyglet.image.load(filename=None, file=f)

    def _create_window(self, width, height):
        try:
            config = pyglet.gl.Config(sample_buffers=1,
                                      samples=4,
                                      depth_size=24,
                                      double_buffer=True)
            window = pyglet.window.Window(config=config,
                                          width=width,
                                          height=height)
        except pyglet.window.NoSuchConfigException:
            config = pyglet.gl.Config(double_buffer=True)
            window = pyglet.window.Window(config=config,
                                          width=width,
                                          height=height)

        @window.event
        def on_key_press(symbol, modifiers):
            if modifiers == 0:
                if symbol == pyglet.window.key.Q:
                    window.close()

        return window


if __name__ == '__main__':
    np.random.seed(0)
    Application()
