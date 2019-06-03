[![trimesh](https://trimsh.org/images/logotype-a.svg)](http://trimsh.org)

-----------
[![Build Status](https://travis-ci.org/mikedh/trimesh.svg?branch=master)](https://travis-ci.org/mikedh/trimesh) [![Build status](https://ci.appveyor.com/api/projects/status/j8h3luwvst1tkghl/branch/master?svg=true)](https://ci.appveyor.com/project/mikedh/trimesh/branch/master) [![Coverage Status](https://coveralls.io/repos/github/mikedh/trimesh/badge.svg)](https://coveralls.io/github/mikedh/trimesh) [![PyPI version](https://badge.fury.io/py/trimesh.svg)](https://badge.fury.io/py/trimesh) [![Join the chat at https://gitter.im/trimsh/Lobby](https://badges.gitter.im/trimsh/Lobby.svg)](https://gitter.im/trimsh/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)



Trimesh is a pure Python (2.7- 3.3+) library for loading and using [triangular meshes](https://en.wikipedia.org/wiki/Triangle_mesh) with an emphasis on watertight surfaces. The goal of the library is to provide a full featured and well tested Trimesh object which allows for easy manipulation and analysis, in the style of the Polygon object in the [Shapely library](https://github.com/Toblerity/Shapely).

The API is mostly stable, but this should not be relied on and is not guaranteed; install a specific version if you plan on deploying something using trimesh as a backend.

Pull requests are appreciated and responded to promptly! If you'd like to contribute, here is an [up to date list of potential enhancements](https://github.com/mikedh/trimesh/issues/199) although things not on that list are also welcome. Here are some [tips for writing mesh code in Python.](https://github.com/mikedh/trimesh/blob/master/trimesh/exchange/README.md)


## Basic Installation

The minimal requirements to import trimesh are
[numpy](http://www.numpy.org/), [scipy](http://www.scipy.org) and [networkx](https://networkx.github.io). Installing other packages mentioned adds functionality but is **not required**.

For the easiest install with *only* these minimal dependencies `pip` can generally install `trimesh` cleanly on Windows, Linux, and OSX:

```bash
pip install trimesh
```

For more functionality, like faster ray queries (`pyembree`), vector path handling (`shapely` and `rtree`), preview windows (`pyglet`), faster cache checks (`xxhash`) and more, the easiest way to get a full `trimesh` install is a [conda environment](https://conda.io/miniconda.html):

```bash
# this will install all soft dependencies available on your current platform
conda install -c conda-forge trimesh
```

If you're feeling lucky, you can try:
```bash
# will try to install things that aren't too tricky
pip install trimesh[easy]

# will try to install everything
pip install trimesh[all]
```

Further information is available in the [advanced installation documentation](https://trimsh.org/install.html).

## Quick Start

Here is an example of loading a mesh from file and colorizing its faces. Here is a nicely formatted
[ipython notebook version](https://trimsh.org/examples/quick_start.html) of this example. Also check out the [cross section example](https://trimsh.org/examples/section.html) or possibly the [integration of a function over a mesh example](https://github.com/mikedh/trimesh/blob/master/examples/integrate.ipynb).

```python
import numpy as np
import trimesh

# attach to logger so trimesh messages will be printed to console
trimesh.util.attach_to_log()

# mesh objects can be created from existing faces and vertex data
mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                       faces=[[0, 1, 2]])

# mesh objects can be loaded from a file name or from a buffer
mesh = trimesh.load('../models/featuretype.STL')

# is the current mesh watertight?
mesh.is_watertight

# what's the euler number for the mesh?
mesh.euler_number

# the convex hull is another Trimesh object that is available as a property
# lets compare the volume of our mesh with the volume of its convex hull
print(mesh.volume / mesh.convex_hull.volume)

# since the mesh is watertight, it means there is a
# volumetric center of mass which we can set as the origin for our mesh
mesh.vertices -= mesh.center_mass

# what's the moment of inertia for the mesh?
mesh.moment_inertia

# if there are multiple bodies in the mesh we can split the mesh by
# connected components of face adjacency
# since this example mesh is a single watertight body we get a list of one mesh
mesh.split()

# facets are groups of coplanar adjacent faces
# set each facet to a random color
# colors are 8 bit RGBA by default (n, 4) np.uint8
for facet in mesh.facets:
    mesh.visual.face_colors[facet] = trimesh.visual.random_color()

# preview mesh in an opengl window if you installed pyglet with pip
mesh.show()

# transform method can be passed a (4, 4) matrix and will cleanly apply the transform
mesh.apply_transform(trimesh.transformations.random_rotation_matrix())

# axis aligned bounding box is available
mesh.bounding_box.extents

# a minimum volume oriented bounding box also available
# primitives are subclasses of Trimesh objects which automatically generate
# faces and vertices from data stored in the 'primitive' attribute
mesh.bounding_box_oriented.primitive.extents
mesh.bounding_box_oriented.primitive.transform

# show the mesh appended with its oriented bounding box
# the bounding box is a trimesh.primitives.Box object, which subclasses
# Trimesh and lazily evaluates to fill in vertices and faces when requested
# (press w in viewer to see triangles)
(mesh + mesh.bounding_box_oriented).show()

# bounding spheres and bounding cylinders of meshes are also
# available, and will be the minimum volume version of each
# except in certain degenerate cases, where they will be no worse
# than a least squares fit version of the primitive.
print(mesh.bounding_box_oriented.volume, 
      mesh.bounding_cylinder.volume,
      mesh.bounding_sphere.volume)

```

## Features

* Import meshes from binary/ASCII STL, Wavefront OBJ, ASCII OFF, binary/ASCII PLY, GLTF/GLB 2.0, 3MF, XAML, 3DXML, etc.
* Import and export 2D or 3D vector paths from/to DXF or SVG files
* Import geometry files (i.e. BREP (.brep), STEP (.stp or .step), 'IGES (.igs or .iges), etc.) via Gmsh SDK
* Import skin of 3D FE Models (i.e. Abaqus (*.inp), Nastran (*.bdf), etc.) via Gmsh SDK
* Export meshes as binary STL, binary PLY, ASCII OFF, OBJ, GLTF/GLB 2.0, COLLADA, etc.
* Export mesh as FE Models (i.e. Abaqus (*.inp), Nastran (*.bdf), etc.) via Gmsh SDK
* Preview meshes using pyglet
* Preview meshes in- line in jupyter notebooks using three.js
* Automatic hashing of numpy arrays storing key data for change tracking using MD5, zlib CRC, or xxhash
* Internal caching of computed values validated from hashes
* Fast loading of binary files through importers written by defining custom numpy dtypes
* Calculate things like face adjacencies, face angles, vertex defects, etc.
* Calculate cross sections (i.e. the slicing operation used in 3D printing)
* Slice meshes with one or multiple arbitrary planes and return the resulting surface
* Split mesh based on face connectivity using networkx, graph-tool, or scipy.sparse
* Calculate mass properties, including volume, center of mass, moment of inertia, and principal components of inertia vectors and components
* Repair triangle winding and normals to be consistent 
* Convex hulls of meshes 
* Compute an identifier that is mostly rotation/translation/tessellation invariant
* Determine duplicate meshes from identifier
* Determine if a mesh is watertight, convex, etc.
* Repair single triangle and single quad holes
* Uniformly sample the surface of a mesh
* Ray-mesh queries including location, triangle index, etc.
* Boolean operations on meshes (intersection, union, difference) using OpenSCAD or Blender as backend
* Voxelize watertight meshes
* Volume mesh generation (TETgen) using Gmsh SDK
* Smooth watertight meshes using laplacian smoothing algorithms (Classic, Taubin, Humphrey)
* Subdivide faces of a mesh
* Minimum volume oriented bounding boxes for meshes
* Minimum volume bounding sphere / n-spheres
* Symbolic integration of function(x,y,z) over a triangles
* Calculate nearest point on mesh surface and signed distance
* Determine if a point lies inside or outside of a mesh using signed distance
* Primitive objects (Box, Cylinder, Sphere, Extrusion) which are subclassed Trimesh objects and have all the same features (inertia, viewers, etc)
* Simple scene graph and transform tree which can be rendered (pyglet window or three.js in a jupyter notebook) or exported.
* Numerous utility functions, such as transforming points, unitizing vectors, tracking arrays for changes, grouping rows, etc.


## Viewer

Trimesh includes an optional `pyglet` based viewer for debugging and inspecting. In the mesh view window, opened with `mesh.show()`, the following commands can be used:

* `mouse click + drag` rotates the view
* `ctl + mouse click + drag` pans the view
* `mouse wheel` zooms
* `z` returns to the base view
* `w` toggles wireframe mode
* `c` toggles backface culling
* `f` toggles between fullscreen and windowed mode
* `m` maximizes the window
* `q` closes the window
* `a` toggles an XYZ-RGB axis marker between three states: off, at world frame, or at every frame

If called from inside a `jupyter` notebook, `mesh.show()` displays an in-line preview using `three.js` to display the mesh or scene. For more complete rendering (PBR, better lighting, shaders, better off-screen support, etc) [pyrender](https://github.com/mmatl/pyrender) is designed to interoperate with `trimesh` objects.

## Containers
   
If you want to deploy something in a container that uses trimesh, automated builds containing trimesh and its dependencies are available on Docker Hub:

`docker pull mikedh/trimesh`
