trimesh.py
==========

Python library for loading triangular meshes and doing simple operations on them. Included loaders are binary/ascii STL and Wavefront (OBJ), and exports as binary STL or COLLADA. If Assimp/pyassimp are available, it can also load meshes using that (which supports a huge number of formats). 

### Features
* Preview meshes (requires pyglet). 
* Calculate face adjacencies
* Calculate cross sections
* Split mesh based on face connectivity (requires networkx)
* Calculate mass properties, including volume, center of mass, and moment of inertia
* Find planar facets
* Find and fix face normals and triangle winding
* Find convex hulls of meshes
* Numerous utility functions, such as transforming points, unitizing vectors, etc. 

### Requirements
Requires numpy and networkx, additional functions are available with scipy (>.12), pyglet, and pyassimp

### Example

    import trimesh
    
    # load_mesh can accept a filename or file object, 
    # however file objects require 'file_type' specified (eg. file_type='stl')
    mesh = trimesh.load_mesh('./models/unit_cube.STL')
    
    # does basic cleanup of mesh, including merging vertices and removing
    # duplicate/degenerate faces
    mesh.process()
    
    # opens opengl preview window of mesh, with the argument enabling smoothing 
    # smoothing is disabled by default as on large meshes (>10mb) it can be slow
    mesh.show(True)
    
In the mesh view window, dragging rotates the view, ctl + drag pans, mouse wheel scrolls, 'z' returns to the base view, 'w' toggles wireframe mode, and 'c' toggles backface culling (useful if viewing non-watertight meshes).  