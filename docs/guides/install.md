Installation
=====================

The only thing required to install `trimesh` is
[numpy](http://www.numpy.org/).

All other dependencies are \'soft,\' or trimesh will raise the exceptions (usually but not always an `ImportError`) at runtime if a function is called that requires a package that isn\'t installed. If you do the most basic install of `trimesh` it will only install `numpy`:

```
pip install trimesh
```

This will enable you to load most formats into numpy arrays: STL, PLY, OBJ, GLB, GLTF.

If you\'d like most soft dependencies which should install cleanly on Mac, Windows, and Linux, you can use the `easy` pip extra:

```
pip install trimesh[easy]
```

Or if you want the full experience, you can try the `all` extra, where packages may only be available for Linux:
```
pip install trimesh[all]
```


Conda Packages
--------------

If you prefer a `conda` environment, `trimesh` is available on `conda-forge` ([trimesh-feedstock repo](https://github.com/conda-forge/trimesh-feedstock))


If you install [Miniconda](https://conda.io/docs/install/quick.html) you can then run:

```
conda install -c conda-forge trimesh
```
      
Ubuntu-Debian Notes
-------------------

Blender and openSCAD are soft dependencies used for boolean operations with subprocess, you can get them with `apt`:

```
sudo apt-get install openscad blender
```

Dependency Overview
--------------------

Trimesh has a lot of soft-required upstream packages. We try to make sure they're active and big-ish. Here's a quick summary of what they're used for.

 
| Package | Description | Alternatives | Level |
| ------  | ---------   | ----------   | ----- |
| `numpy` | The base container for fast array types. | | `required` | 
| `scipy` | Provides convex hulls (`scipy.spatial.ConvexHull`), fast graph operations (`scipy.sparse.csgraph`), fast nearest-point queries (`scipy.spatial.cKDTree`), b-spline evaluation (`scipy.interpolate`). | Nothing comes to mind, it does a whole heck of a lot. | `easy` |  
| `lxml` | Parse XML documents. We use this over the built-in ones as it was slightly faster, and there was a format implemented which was extremely annoying to handle without the ability to get parent nodes (which `lxml` has but built-in XML doesn't). | Standard library's XML | `easy` |
| `networkx` | A nice-to-use pure Python graph library that's faster than you'd think. It implements DFS, BFS, and the usual FAANG-interview-question algorithms. A lot of the commonly run stuff in trimesh has been re-written to use `scipy.sparse.csgraph` as it's also an easy install and is way faster in most cases. But if you have a small-ish graph the API for `networkx` is way easier to "grok". | `graph-tool`, `scipy.sparse.csgraph` | `easy` |
| `shapely` | Bindings to `GEOS` for 2D spatial stuff: "set-theoretic analysis and manipulation of planar features." It lets you offset, union, and query polygons nicely. | `clipper` maybe? | `easy` | 
| `rtree` | Query ND rectangles with a spatial tree for a "broad phase" intersection. Used in polygon generation ("given N closed curves which curve contains the other curve?") and as the broad-phase for the built-in-numpy slow ray query engine. | `fcl` maybe? | `easy` |
|`requests`| Do network queries in `trimesh.exchange.load_remote`, will *only* make network requests when asked | | `easy`|
|`sympy`| Evaluate symbolic algebra | | `easy`|
|`xxhash`| Quickly hash arrays, used for our cache checking | | `easy`|
|`msgpack`| A serialization method that supports bytes-blobs. | `protobuf` | `easy`|
|`chardet`| When we fail to decode text as UTF-8 we then check with chardet which guesses an encoding. This lets us load files even with weird encodings. | | `easy`|
|`colorlog`| Printing logs with colors. | | `easy`|
|`pillow`| Reading raster images for textures, and rendering polygons into raster images. | | `easy`|
|`svg.path`| Parsing SVG path strings. | | `easy`|
|`jsonschema`| Validating our exports for formats like GLTF. | | `easy`|
|`pycollada`| Parse `dae` files. | | `easy`|
|`pyglet`| OpenGL bindings for our simple debug viewer. | | `easy`|
|`xatlas`| Unwrap meshes to generate UV coordinates quickly and well. | | `all`|
|`python-fcl`| Do collision queries between meshes | | `all`|
|`glooey`| Provide a viewer with widgets. | | `all`|
|`meshio`| Load additional mesh formats. | | `all`|
|`scikit-image`| Used in voxel ops | | `all`|
|`mapbox-earcut`| Triangulate 2D polygons | `triangle` which has an unusual license | `all`|
|`psutil`| Get current memory usage, useful for checking to see if we're going to run out of memory instantiating a giant array | | `all`|
|`ruff`| A static code analyzer that replaces `flake8`. | `flake8` | `test`|
|`autopep8`| A code formatter which fixes whitespace issues automatically. | | `test`|
|`pytest`| A test runner. | | `test`|
|`pytest-cov`| A plugin to calculate test coverage. | | `test`|
|`pyinstrument`| A sampling based profiler for performance tweaking. | | `test`|
