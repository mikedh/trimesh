trimesh
==========
[![Build Status](https://travis-ci.org/mikedh/trimesh.svg?branch=master)](https://travis-ci.org/mikedh/trimesh)

Python (2.7-3.*) library for loading and utilizing triangular meshes.

### Features
* Import binary/ASCII STL, Wavefront, and OFF
* Export binary STL, COLLADA, and OFF
* Import formats using assimp (if pyassimp installed)
* Load STEP files as meshes (if STEPtools Inc. Author Tools installed)
* Preview meshes (requires pyglet). 
* Fast loading of binary and ASCII STL files (on 234,230 face mesh, was 24.5x faster than assimp)
* Calculate face adjacencies quickly (for the same 234,230 face mesh .248 s)
* Calculate cross sections (.146 s)
* Split mesh based on face connectivity using networkx (4.96 s) or graph-tool (.584 s)
* Calculate mass properties, including volume, center of mass, and moment of inertia (.246 s)
* Find coplanar groups of faces (.454 s)
* Fix triangle winding to be consistent 
* Fix normals to be oriented 'outwards' using ray tests
* Find convex hulls of meshes (.21 s)
* Compute a rotation/translation/tessellation invariant identifier for meshes (from an FFT of the radius distribution)
* Merge duplicate meshes from identifier
* Determine if a mesh is watertight (manifold)
* Repair single triangle and single quad holes
* Uniformly sample the surface of a mesh
* Find ray-mesh intersections
* Voxelize watertight meshes
* Unit conversions
* Load 2D or 3D vector paths from DXF or SVG files
* Create meshes by extruding 2D profiles
* Numerous utility functions, such as transforming points, unitizing vectors, grouping rows, etc. 

### Installation
The easiest way to install is:

    sudo pip install git+https://github.com/mikedh/trimesh.git

### Dependencies
If you would like to use ray queries or some path functionality, install dependencies:

    sudo pip install shapely git+https://github.com/Toblerity/rtree.git svg.path 

To get the latest version of assimp/pyassimp from github:

    sudo pip install git+https://github.com/robotics/assimp_latest.git 

Rtree may not build without libspatialindex installed, get it with:

    sudo apt-get install libspatialindex* 

If you would like to use the trimesh.creation functions, meshpy is required:

    sudo pip install meshpy

### Example
    import numpy as np
    import trimesh
    
    # this list will be much longer if assimp is available
    print(trimesh.available_formats())

    # load_mesh can accept a filename or file object, 
    # however file objects require 'file_type' specified (eg. file_type='stl')
    # on load does basic cleanup of mesh, including merging vertices 
    # and removing duplicate/degenerate faces
    mesh = trimesh.load_mesh('./models/unit_cube.STL')
    
    # split mesh based on connected components
    # by default this will only return watertight meshes, but the check can be disabled
    meshes = mesh.split() 

    # first component  
    m = meshes[0]

    # assign all faces a color
    m.set_face_colors()

    # find groups of coplanar adjacent faces
    facets, facets_area = m.facets(return_area=True)

    # the largest group of faces by area    
    largest_facet = facets[np.argmax(facets_area)]

    # set all faces of the largest facet to a random color
    m.faces[largest_facet] = trimesh.color.random_color()

    # preview mesh in an opengl window
    m.show()
    

In the mesh view window, dragging rotates the view, ctl + drag pans, mouse wheel scrolls, 'z' returns to the base view, 'w' toggles wireframe mode, and 'c' toggles backface culling (useful if viewing non-watertight meshes).  