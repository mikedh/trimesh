[![trimesh](https://trimesh.org/_static/images/logotype-a.svg)](http://trimesh.org)

-----------
[![Github Actions](https://github.com/mikedh/trimesh/workflows/Release%20Trimesh/badge.svg)](https://github.com/mikedh/trimesh/actions) [![codecov](https://codecov.io/gh/mikedh/trimesh/branch/main/graph/badge.svg?token=4PVRQXyl2h)](https://codecov.io/gh/mikedh/trimesh)  [![Docker Image Version (latest by date)](https://img.shields.io/docker/v/trimesh/trimesh?label=docker&sort=semver)](https://hub.docker.com/r/trimesh/trimesh/tags) [![PyPI version](https://badge.fury.io/py/trimesh.svg)](https://badge.fury.io/py/trimesh)


Trimesh is a pure Python 3.8+ library for loading and using [triangular meshes](https://en.wikipedia.org/wiki/Triangle_mesh) with an emphasis on watertight surfaces. The goal of the library is to provide a full featured and well tested Trimesh object which allows for easy manipulation and analysis, in the style of the Polygon object in the [Shapely library](https://github.com/Toblerity/Shapely).

The API is mostly stable, but this should not be relied on and is not guaranteed: install a specific version if you plan on deploying something using trimesh.

Pull requests are appreciated and responded to promptly! If you'd like to contribute, here is an [up to date list of potential enhancements](https://github.com/mikedh/trimesh/issues/1557) although things not on that list are also welcome. Here's a quick [development and contributing guide.](https://trimesh.org/contributing.html)


## Basic Installation

Keeping `trimesh` easy to install is a core goal, thus the *only* hard dependency is [numpy](http://www.numpy.org/). Installing other packages adds functionality but is not required. For the easiest install with just `numpy`, use `pip`:

```bash
pip install trimesh
```

The minimal install can load many supported formats (STL, PLY, GLTF/GLB) into numpy arrays. More functionality is available when soft dependencies are installed. This includes things like convex hulls (`scipy`), graph operations (`networkx`), faster ray queries (`embreex`), vector path handling (`shapely` and `rtree`), XML formats like 3DXML/XAML/3MF (`lxml`), preview windows (`pyglet`), faster cache checks (`xxhash`), etc.

To install `trimesh` with the soft dependencies that generally install cleanly on Linux x86_64, MacOS ARM, and Windows x86_64 using `pip`:
```bash
pip install trimesh[easy]
```

If you are supporting a different platform or are freezing your dependencies we recommend you do not use extras: depend on `trimesh scipy` versus `trimesh[easy]`. Further information is available in the [advanced installation documentation](https://trimesh.org/install.html).

## Quick Start

Here is an example of loading a mesh from file and colorizing its faces. Here is a nicely formatted
[ipython notebook version](https://trimesh.org/quick_start.html) of this example. Also check out the [cross section example](https://trimesh.org/section.html).

```python
import numpy as np
import trimesh

# attach to logger so trimesh messages will be printed to console
trimesh.util.attach_to_log()

# mesh objects can be created from existing faces and vertex data
mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                       faces=[[0, 1, 2]])

# by default, Trimesh will do a light processing, which will
# remove any NaN values and merge vertices that share position
# if you want to not do this on load, you can pass `process=False`
mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                       faces=[[0, 1, 2]],
                       process=False)

# some formats represent multiple meshes with multiple instances
# the loader tries to return the datatype which makes the most sense
# which will for scene-like files will return a `trimesh.Scene` object.
# if you *always* want a straight `trimesh.Trimesh` you can ask the
# loader to "force" the result into a mesh through concatenation
mesh = trimesh.load_mesh('models/CesiumMilkTruck.glb')

# mesh objects can be loaded from a file name or from a buffer
# you can pass any of the kwargs for the `Trimesh` constructor
# to `trimesh.load`, including `process=False` if you would like
# to preserve the original loaded data without merging vertices
# STL files will be a soup of disconnected triangles without
# merging vertices however and will not register as watertight
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

# preview mesh in an opengl window if you installed pyglet and scipy with pip
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
* Import geometry files using the GMSH SDK if installed (BREP, STEP, IGES, INP, BDF, etc)
* Export meshes as binary STL, binary PLY, ASCII OFF, OBJ, GLTF/GLB 2.0, COLLADA, etc.
* Preview meshes using pyglet or in- line in jupyter/marimo notebooks using three.js
* Automatic hashing of numpy arrays for change tracking using MD5, zlib CRC, or xxhash
* Internal caching of computed values validated from hashes
* Calculate face adjacencies, face angles, vertex defects, etc.
* Calculate cross sections, i.e. the slicing operation used in 3D printing
* Slice meshes with one or multiple arbitrary planes and return the resulting surface
* Split mesh based on face connectivity using networkx, graph-tool, or scipy.sparse
* Calculate mass properties, including volume, center of mass, moment of inertia, principal components of inertia vectors and components
* Repair simple problems with triangle winding, normals, and quad/tri holes
* Convex hulls of meshes
* Compute rotation/translation/tessellation invariant identifier and find duplicate meshes
* Determine if a mesh is watertight, convex, etc.
* Uniformly sample the surface of a mesh
* Ray-mesh queries including location, triangle index, etc.
* Boolean operations on meshes (intersection, union, difference) using Manifold3D or Blender Note that mesh booleans in general are usually slow and unreliable.
* Voxelize watertight meshes
* Smooth watertight meshes using laplacian smoothing algorithms (Classic, Taubin, Humphrey)
* Subdivide faces of a mesh
* Approximate minimum volume oriented bounding boxes for meshes
* Approximate minimum volume bounding spheres
* Calculate nearest point on mesh surface and signed distance
* Determine if a point lies inside or outside of a well constructed mesh using signed distance
* Primitive objects `Box`, `Cylinder`, `Sphere`, `Extrusion` are subclassed Trimesh objects and have all the same features (inertia, viewers, etc).
* Simple scene graph and transform tree which can be rendered (pyglet window, three.js in a jupyter/marimo notebook, [pyrender](https://github.com/mmatl/pyrender)) or exported.
* Many utility functions, like transforming points, unitizing vectors, aligning vectors, tracking numpy arrays for changes, grouping rows, etc.

## Which Mesh Format Should I Use?

Quick recommendation is to use `GLB`, and to avoid `OBJ` if possible. More discussion is in the [`documention`](https://trimesh.org/formats)


[Further information is available in the docs](https://trimesh.org)

