# empty: the legacy viewer is loaded by `trimesh.viewer.__init__` directly
# from `.pyglet1.viewer`, so importing this package does not pull in
# pyglet (the `trimesh.rendering` shim only needs `.conversion`).
