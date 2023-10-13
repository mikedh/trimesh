[![trimesh](https://trimesh.org/images/logotype-a.svg)](http://trimesh.org)

-----------
[![Github Actions](https://github.com/mikedh/trimesh/workflows/Release%20Trimesh/badge.svg)](https://github.com/mikedh/trimesh/actions) [![codecov](https://codecov.io/gh/mikedh/trimesh/branch/main/graph/badge.svg?token=4PVRQXyl2h)](https://codecov.io/gh/mikedh/trimesh)  [![Docker Image Version (latest by date)](https://img.shields.io/docker/v/trimesh/trimesh?label=docker&sort=semver)](https://hub.docker.com/r/trimesh/trimesh/tags) [![PyPI version](https://badge.fury.io/py/trimesh.svg)](https://badge.fury.io/py/trimesh)


Trimesh is a pure Python 3.7+ library for loading and using [triangular meshes](https://en.wikipedia.org/wiki/Triangle_mesh) with an emphasis on watertight surfaces. The goal of the library is to provide a full featured and well tested Trimesh object which allows for easy manipulation and analysis, in the style of the Polygon object in the [Shapely library](https://github.com/Toblerity/Shapely).

The API is mostly stable, but this should not be relied on and is not guaranteed: install a specific version if you plan on deploying something using trimesh.

Pull requests are appreciated and responded to promptly! If you'd like to contribute, here is an [up to date list of potential enhancements](https://github.com/mikedh/trimesh/issues/1557) although things not on that list are also welcome. Here's a quick [development and contributing guide.](https://trimsh.org/contributing.html)


## Basic Installation

Keeping `trimesh` easy to install is a core goal, thus the *only* hard dependency is [numpy](http://www.numpy.org/). Installing other packages adds functionality but is not required. For the easiest install with just numpy, `pip` can generally install `trimesh` cleanly on Windows, Linux, and OSX:

```bash
pip install trimesh
```

The minimal install can load many supported formats (STL, PLY, GLTF/GLB) into numpy arrays. More functionality is available when soft dependencies are installed. This includes things like convex hulls (`scipy`), graph operations (`networkx`), faster ray queries (`pyembree`), vector path handling (`shapely` and `rtree`), XML formats like 3DXML/XAML/3MF (`lxml`), preview windows (`pyglet`), faster cache checks (`xxhash`), etc. To install `trimesh` with the soft dependencies that generally install cleanly on Linux, OSX, and Windows using `pip`:
```bash
pip install trimesh[easy]
```

Further information is available in the [advanced installation documentation](https://trimsh.org/install.html).

## Quick Start

Here is an example of loading a mesh from file and colorizing its faces. Here is a nicely formatted
[ipython notebook version](https://trimsh.org/examples/quick_start.html) of this example. Also check out the [cross section example](https://trimsh.org/examples/section.html).

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
mesh = trimesh.load('models/CesiumMilkTruck.glb', force='mesh')

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
* Export meshes using the GMSH SDK if installed (Abaqus INP, Nastran BDF, etc)
* Preview meshes using pyglet or in- line in jupyter notebooks using three.js
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
* Boolean operations on meshes (intersection, union, difference) using OpenSCAD or Blender as a back end. Note that mesh booleans in general are usually slow and unreliable
* Voxelize watertight meshes
* Volume mesh generation (TETgen) using Gmsh SDK
* Smooth watertight meshes using laplacian smoothing algorithms (Classic, Taubin, Humphrey)
* Subdivide faces of a mesh
* Approximate minimum volume oriented bounding boxes for meshes
* Approximate minimum volume bounding spheres
* Calculate nearest point on mesh surface and signed distance
* Determine if a point lies inside or outside of a well constructed mesh using signed distance
* Primitive objects (Box, Cylinder, Sphere, Extrusion) which are subclassed Trimesh objects and have all the same features (inertia, viewers, etc)
* Simple scene graph and transform tree which can be rendered (pyglet window, three.js in a jupyter notebook, [pyrender](https://github.com/mmatl/pyrender)) or exported.
* Many utility functions, like transforming points, unitizing vectors, aligning vectors, tracking numpy arrays for changes, grouping rows, etc.


## Viewer

Trimesh includes an optional `pyglet` based viewer for debugging and inspecting. In the mesh view window, opened with `mesh.show()`, the following commands can be used:

* `mouse click + drag` rotates the view
* `ctl + mouse click + drag` pans the view
* `mouse wheel` zooms
* `z` returns to the base view
* `w` toggles wireframe mode
* `c` toggles backface culling
* `g` toggles an XY grid with Z set to lowest point
* `a` toggles an XYZ-RGB axis marker between: off, at world frame, or at every frame and world, and at every frame
* `f` toggles between fullscreen and windowed mode
* `m` maximizes the window
* `q` closes the window

If called from inside a `jupyter` notebook, `mesh.show()` displays an in-line preview using `three.js` to display the mesh or scene. For more complete rendering (PBR, better lighting, shaders, better off-screen support, etc) [pyrender](https://github.com/mmatl/pyrender) is designed to interoperate with `trimesh` objects.

## Projects Using Trimesh

You can check out the [Github network](https://github.com/mikedh/trimesh/network/dependents) for things using trimesh. A select few:
- Nvidia's [kaolin](https://github.com/NVIDIAGameWorks/kaolin) for deep learning on 3D geometry.
- [Cura](https://github.com/Ultimaker/Cura), a popular slicer for 3D printing.
- Berkeley's [DexNet4](https://www.youtube.com/watch?v=GBiAxoWBNho&feature=emb_logo) and related [ambidextrous.ai](https://www.ambidextrous.ai/) work with robotic grasp planning and manipulation.
- Kerfed's [Kerfed's Engine](https://kerfed.com/technology) for analyzing assembly geometry for manufacturing.
- [MyMiniFactory's](https://www.myminifactory.com/) P2Slice for preparing models for 3D printing.
- [pyrender](https://github.com/mmatl/pyrender) A library to render scenes from Python using nice looking PBR materials.
- [urdfpy](https://github.com/mmatl/urdfpy) Load URDF robot descriptions in Python.
- [moderngl-window](https://github.com/moderngl/moderngl-window) A helper to create GL contexts and load meshes.
- [vedo](https://github.com/marcomusy/vedo) Visualize meshes interactively (see example [gallery](https://github.com/marcomusy/vedo/tree/master/examples/other/trimesh/)).
- [FSLeyes](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes) View MRI images and brain data.

## Which Mesh Format Should I Use?

Quick recommendation: `GLB` or `PLY`. Every time you replace `OBJ` with `GLB` an angel gets its wings.

If you want things like by-index faces, instancing, colors, textures, etc, `GLB` is a terrific choice. GLTF/GLB is an [extremely well specified](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0) modern format that is easy and fast to parse: it has a JSON header describing data in a binary blob. It has a simple hierarchical scene graph, a great looking modern physically based material system, support in [dozens-to-hundreds of libraries](https://github.com/KhronosGroup/glTF/issues/1058), and a [John Carmack endorsment](https://www.khronos.org/news/press/significant-gltf-momentum-for-efficient-transmission-of-3d-scenes-models). Note that GLTF is a large specification, and `trimesh` only supports a subset of features: loading basic geometry is supported, NOT supported are fancier things like animations, skeletons, etc.

In the wild, `STL` is perhaps the most common format. `STL` files are extremely simple: it is basically just a list of triangles. They are robust and are a good choice for basic geometry. Binary `PLY` files are a good step up, as they support indexed faces and colors.

Wavefront `OBJ` is also pretty common: unfortunately OBJ doesn't have a widely accepted specification so every importer and exporter implements things slightly differently, making it tough to support. It also allows unfortunate things like arbitrary sized polygons, has a face representation which is easy to mess up, references other files for materials and textures, arbitrarily interleaves data, and is slow to parse. Give `GLB` or `PLY` a try as an alternative!

## How can I cite this library?

A question that comes up pretty frequently is [how to cite the library.](https://github.com/mikedh/trimesh/issues?utf8=1&q=cite) A quick BibTex recommendation:
```
@software{trimesh,
	author = {{Dawson-Haggerty et al.}},
	title = {trimesh},
	url = {https://trimsh.org/},
	version = {3.2.0},
	date = {2019-12-8},
}
```

## Containers

If you want to deploy something in a container that uses trimesh automated `debian:slim-bullseye` based builds with trimesh and most dependencies are available on [Docker Hub](https://hub.docker.com/repository/docker/trimesh/trimesh) with image tags for `latest`, git short hash for the commit in `main` (i.e. `trimesh/trimesh:0c1298d`), and version (i.e. `trimesh/trimesh:3.5.27`):

`docker pull trimesh/trimesh`

[Here's an example](https://github.com/mikedh/trimesh/tree/main/examples/docker/render) of how to render meshes using LLVMpipe and XVFB inside a container.

