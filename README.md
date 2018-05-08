# trimesh #
[![Build Status](https://travis-ci.org/mikedh/trimesh.svg?branch=master)](https://travis-ci.org/mikedh/trimesh) [![Build status](https://ci.appveyor.com/api/projects/status/j8h3luwvst1tkghl?svg=true)](https://ci.appveyor.com/project/mikedh/trimesh)

Trimesh is a Python (2.7- 3.3+) library for loading and using [triangular meshes](https://en.wikipedia.org/wiki/Triangle_mesh) with an emphasis on watertight meshes. The goal of the library is to provide a fully featured Trimesh object which allows for easy manipulation and analysis, in the style of the excellent Polygon object in the [Shapely library](http://toblerity.org/shapely/manual.html).

The API is mostly stable, but this should not be relied on and is not guaranteed; install a specific version if you plan on deploying something using trimesh as a backend.

## Basic Installation


The minimal requirements to import trimesh are
[numpy](http://www.numpy.org/), [scipy](http://www.scipy.org) and
[networkx](https://networkx.github.io). Installing other packages mentioned adds functionality but is **not required**.

For the easiest install with only these minimal dependencies (slower ray queries, no vector path handling, mesh creation, viewer, etc):

```bash
pip install trimesh
```

For more functionality, the easiest way to get a full Trimesh install is a [conda environment](https://conda.io/miniconda.html):

```bash
# install modules for spatial indexing and  polygon manipulation
# these generally install cleanly on Linux, Windows, and OSX
conda install -c conda-forge rtree shapely

# install pyembree for fast ray queries
# Linux and OSX only
conda install -c conda-forge pyembree

# install Trimesh and soft dependencies that are easy to install
# these generally install cleanly on Linux, Windows, and OSX
pip install trimesh[easy]
```
 
Further information is available in the [advanced installation documentation](http://trimesh.readthedocs.io/en/latest/install.html).

## Quick Start

Here is an example of loading a mesh from file and colorizing its faces. Here is a nicely formatted
[ipython notebook version](http://github.com/mikedh/trimesh/blob/master/examples/quick_start.ipynb) of this example. Also check out the [cross section example](https://github.com/mikedh/trimesh/blob/master/examples/section.ipynb) or possibly the [integration of a function over a mesh example](https://github.com/mikedh/trimesh/blob/master/examples/integrate.ipynb).

```python
import numpy as np
import trimesh

# attach to logger so trimesh messages will be printed to console
trimesh.util.attach_to_log()

# load a file by name or from a buffer
mesh = trimesh.load('../models/featuretype.STL')

# is the current mesh watertight?
mesh.is_watertight

# what's the euler number for the mesh?
mesh.euler_number

# the convex hull is another Trimesh object that is available as a property
# lets compare the volume of our mesh with the volume of its convex hull
np.divide(mesh.volume, mesh.convex_hull.volume)

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
# colors are 8 bit RGBA by default (n,4) np.uint8
for facet in mesh.facets:
    mesh.visual.face_colors[facet] = trimesh.visual.random_color()

# preview mesh in an opengl window if you installed pyglet with pip
mesh.show()

# transform method can be passed a (4,4) matrix and will cleanly apply the transform
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

* Import binary/ASCII STL, Wavefront OBJ, ASCII OFF, binary/ASCII PLY, XAML, 3DXML, etc.
* Import additional mesh formats using [assimp](http://www.assimp.org/main_features_formats.html) (requires pyassimp or cyassimp)
* Import and export 2D or 3D vector paths from/to DXF or SVG files
* Export meshes as binary STL, binary PLY, ASCII OFF, COLLADA, dictionaries, JSON- serializable dictionaries (base64 encoded arrays), MSGPACK- serializable dictionaries (binary string arrays)
* Preview meshes (requires pyglet)
* Internal caching of computed values (validated with a zlib.adler32 CRC on face/vertex data)
* Fast loading of binary files through importers written by defining custom numpy dtypes
* Calculate face adjacencies quickly (for 234,230 face mesh .248 s)
* Calculate cross sections (.146 s)
* Split mesh based on face connectivity using networkx, graph-tool, or scipy.sparse
* Calculate mass properties, including volume, center of mass, moment of inertia, and principal components of inertia
* Find coplanar and adjacent groups of faces (.454 s)
* Fix triangle winding and normals to be consistent 
* Find convex hulls of meshes 
* Compute a rotation/translation/tessellation invariant identifier for meshes
* Determine duplicate meshes from identifier
* Determine if a mesh is watertight
* Determine if a mesh is convex
* Repair single triangle and single quad holes
* Uniformly sample the surface of a mesh
* Ray-mesh queries including location, triangle id, etc.
* Boolean operations on meshes (intersection, union, difference) using OpenSCAD or Blender as backend
* Voxelize watertight meshes
* Unit conversions
* Subdivide faces of a mesh
* Minimum volume oriented bounding boxes for meshes
* Minimum volume bounding sphere / n-spheres
* Symbolic integration of function(x,y,z) over a triangle
* Quick (sympy-numpy lambda) evaluation of symbolic integral result over a mesh 
* Calculate nearest point on mesh surface and signed distance
* Determine if a point lies inside or outside of a mesh using signed distance
* Create meshes with primitive objects (Extrude, Box, Sphere) which are subclasses of Trimesh
* Simple scene graph and transform tree which can be rendered (pyglet) or exported.
* Numerous utility functions, such as transforming points, unitizing vectors, tracking arrays for changes, grouping rows, etc.

## Use Cases

The `trimesh.Trimesh` object is most useful on single body, watertight meshes that represents a volume.

If you expect multibody geometry, you are best off dealing with them as a list of `Trimesh` objects, or as a `trimesh.Scene` object which includes things like overall bounding boxes, convex hulls, etc:

```
mesh = trimesh.load('multibody.STL')

# will split meshes into clean watertight chunks
# always returns a Scene object
scene = trimesh.scene.split_scene(mesh)

# if you want it back as a single multibody mesh, the splitting
# will heal problems with individual bodies before concatenating
multi = scene.dump().sum()

```

## Viewer

Trimesh includes an optional pyglet- based viewer for debugging/inspecting. In the mesh view window:

* dragging rotates the view
* ctl + drag pans
* mouse wheel zooms
* 'z' returns to the base view 
* 'w' toggles wireframe mode
* 'c' toggles backface culling

## Containers
   
If you want to deploy something in a container that uses trimesh, automated builds containing trimesh and its dependencies are available on Docker Hub:

`docker pull mikedh/trimesh`
