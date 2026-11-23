"""
opengl
-----------

The OpenGL 3.3 viewer: `shaders` declares every byte the GPU reads as
numpy dtypes and generates the GLSL from them, and `viewer` allocates,
draws and frees every OpenGL object with them. Only `viewer` imports
pyglet, so `shaders` can be used without a display.
"""
