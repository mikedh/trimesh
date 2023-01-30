"""
A simple viewer using the Pyglet 2+ shader pipeline.
"""
from math import pi, sin, cos

import pyglet
from pyglet import gl
from pyglet.math import Mat4, Vec3

from trimesh.rendering import mesh_to_vertexlist


class ShaderViewer(pyglet.window.Window):
    def __init__(self,
                 scene,
                 resolution=None,
                 config=None,
                 visible=True,
                 resizable=True,
                 caption='trimesh viewer'):

        if resolution is None:
            resolution = (1024, 768)

        try:
            # Try and create a window with multisampling (antialiasing)
            config = gl.Config(
                sample_buffers=1,
                samples=4,
                depth_size=24,
                double_buffer=True)
            super(ShaderViewer, self).__init__(
                config=config,
                visible=visible,
                resizable=resizable,
                width=resolution[0],
                height=resolution[1],
                caption=caption)

        except BaseException:
            config = gl.Config(
                sample_buffers=1,
                samples=4,
                depth_size=24,
                double_buffer=True)
            super(ShaderViewer, self).__init__(
                config=config,
                visible=visible,
                resizable=resizable,
                width=resolution[0],
                height=resolution[1],
                caption=caption)

        # One-time GL setup
        gl.glClearColor(1, 1, 1, 1)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        self.on_resize(*self.size)

        # Uncomment this line for a wireframe view:
        # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        self.time = 0.0
        self.group = pyglet.graphics.Group()
        self.batch = pyglet.graphics.Batch()
        self.shader = pyglet.model.get_default_shader()

        self.add_geometry(scene)

        pyglet.clock.schedule(self.update)
        pyglet.app.run()

    def add_geometry(self, geom):
        # Create a Material and Group for the Model
        diffuse = [0.5, 0.0, 0.3, 1.0]
        ambient = [0.5, 0.0, 0.3, 1.0]
        specular = [1.0, 1.0, 1.0, 1.0]
        emission = [0.0, 0.0, 0.0, 1.0]
        shininess = 50

        material = pyglet.model.Material(
            "custom",
            diffuse,
            ambient,
            specular,
            emission,
            shininess)
        group = pyglet.model.MaterialGroup(
            material=material, program=self.shader)

        ka = mesh_to_vertexlist(geom,
                                batch=self.batch,
                                group=self.group,
                                pyglet2=True)

        """
        vertex_list = shader.vertex_list_indexed(len(vertices)//3, GL_TRIANGLES, indices, batch, group,
                                             vertices=('f', vertices),
                                             normals=('f', normals),
                                            colors=('f', material.diffuse * (len(vertices) // 3)))
        """
        vertex_list = self.shader.vertex_list_indexed(**ka)

        self.model = pyglet.model.Model(
            [vertex_list], [self.group], self.batch)

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def on_resize(self, width, height):
        self.viewport = (0, 0, *self.get_framebuffer_size())
        self.projection = Mat4.perspective_projection(
            self.aspect_ratio, z_near=0.1, z_far=255, fov=60)
        return pyglet.event.EVENT_HANDLED

    def update(self, dt):
        self.time += dt
        time = self.time
        rot_x = Mat4.from_rotation(time, Vec3(1, 0, 0))
        rot_y = Mat4.from_rotation(time / 2, Vec3(0, 1, 0))
        rot_z = Mat4.from_rotation(time / 4, Vec3(0, 0, 1))
        trans = Mat4.from_translation((0, 0, -3.0))
        self.model.matrix = trans @ rot_x @ rot_y @ rot_z


if __name__ == '__main__':
    import trimesh
    m = trimesh.creation.box()
    v = ShaderViewer(m)
