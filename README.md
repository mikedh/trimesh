trimesh.py
==========

Simple python library for loading triangular meshes (loads binary/ascii STL and Wavefront (OBJ), exports as binary STL). Can preview meshes (requires pyglet). 

    import trimesh
    
    #load_mesh can accept a filename or file object, 
    #however file objects require 'type' specified (eg. type='stl')
    mesh = trimesh.load_mesh('./models/octagonal_pocket.stl')

    #opens opengl preview window of mesh
    mesh.show()
    
In the mesh view window, dragging rotates the view, ctl + drag pans, mouse wheel scrolls, 'z' returns to the base view, 'w' toggles wireframe mode, 'c' toggles backface culling (useful if viewing non-watertight meshes).  