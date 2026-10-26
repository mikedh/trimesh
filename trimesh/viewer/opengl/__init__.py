"""
opengl
-----------

The OpenGL 3.3 viewer: `shaders` declares every byte the GPU reads as
numpy dtypes and generates the GLSL from them, `resources` allocates,
fills and frees every OpenGL object, and `viewer` draws scenes with
them. Only `resources` and `viewer` import pyglet, so `shaders` can be
used without a display.
"""
