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


## Conda Packages

If you prefer a `conda` environment, `trimesh` is available on `conda-forge` ([trimesh-feedstock repo](https://github.com/conda-forge/trimesh-feedstock))

If you install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) you can then run:

```
conda install -c conda-forge trimesh
```

## Dependency Overview
--------------------

Trimesh has a lot of soft-required upstream packages, and we try to make sure they're actively maintained. Here's a quick summary of what they're used for:

 
| Package | Description | Alternatives | Level |
| ------  | ---------   | ----------   | ----- |
| `numpy` | The base container for fast array types. | | `required` | 
| `scipy` | Provides convex hulls (`scipy.spatial.ConvexHull`), fast graph operations (`scipy.sparse.csgraph`), fast nearest-point queries (`scipy.spatial.cKDTree`), b-spline evaluation (`scipy.interpolate`). | | `easy` |  
| `lxml` | Parse XML documents. We use this over the built-in ones as it was slightly faster, and there was a format implemented which was extremely annoying to handle without the ability to get parent nodes (which `lxml` has but built-in XML doesn't). | Standard library's XML | `easy` |
| `networkx` | Pure Python graph library that's reasonably fast and has a nice API. `scipy.sparse.csgraph` is way faster in most cases but is hard to understand and doesn't implement as many algorithms. | `graph-tool`, `scipy.sparse.csgraph` | `easy` |
| `shapely` | Bindings to `GEOS` for 2D spatial stuff: "set-theoretic analysis and manipulation of planar features" which lets you offset, union, and query polygons. | `clipper` | `easy` | 
| `rtree` | Query ND rectangles with a spatial tree for a "broad phase" intersection. Used in polygon generation ("given N closed curves which curve contains the other curve?") and as the broad-phase for the built-in-numpy slow ray query engine. | `fcl` maybe? | `easy` |
|`httpx`| Do network queries in `trimesh.exchange.load_remote`, will *only* make network requests when asked | `requests`, `aiohttp` | `easy`|
|`sympy`| Evaluate symbolic algebra | | `recommend`|
|`xxhash`| Quickly hash arrays, used for our cache checking | | `easy`|
|`charset-normalizer`| When we fail to decode text as UTF-8 we then check with charset-normalizer which guesses an encoding,  letting us load files even with weird encodings. | | `easy`|
|`colorlog`| Printing logs with colors. | | `easy`|
|`pillow`| Reading raster images for textures and render polygons into raster images. | | `easy`|
|`svg.path`| Parsing SVG path strings. | | `easy`|
|`jsonschema`| Validating our exports for formats like GLTF. | | `easy`|
|`pycollada`| Parse `dae` files. | | `easy`|
|`pyglet<2`| OpenGL bindings for our simple debug viewer. | | `recommend`|
|`xatlas`| Unwrap meshes to generate UV coordinates quickly and well. | | `recommend`|
|`python-fcl`| Do collision queries between meshes | | `recommend`|
|`glooey`| Provide a viewer with widgets. | | `recommend`|
|`meshio`| Load additional mesh formats. | | `recommend`|
|`scikit-image`| Used in voxel ops | | `recommend`|
|`mapbox-earcut`| Triangulate 2D polygons | `triangle` which has an unusual license | `easy`|
|`psutil`| Get current memory usage, useful for checking to see if we're going to run out of memory instantiating a giant array | | `recommend`|
|`ruff`| A static code analyzer and formatter that replaces `flake8` and `black`. | `flake8` | `test`|
|`pytest`| A test runner. | | `test`|
|`pytest-cov`| A plugin to calculate test coverage. | | `test`|
|`pyinstrument`| A sampling based profiler for performance tweaking. | | `test`|
|`vhacdx`| A binding for VHACD which provides convex decompositions | | `recommend`|
|`manifold3d`| A binding for the Manifold mesh boolean engine | | `recommend`|
|`openctm`| A binding for OpenCTM loaders enabling `.ctm` loading | | `recommend`|
|`cascadio`| A binding for OpenCASCADE enabling `.STEP` loading | | `recommend`|

## Adding A Dependency

If there's no way to implement something reasonably in vectorized Python or there is a mature minimal C++ or Rust implementation of something useful and complicated we may add a dependency. If it's a major, active project with few dependencies (i.e. `jinja2`) that's probably fine. Otherwise it's a lot more of a commitment than just implementing the function in Python however. An example of this is `embree`, Intel's ray check engine: it is a super complicated thing to do well and 50-100x faster than Python ray checks.

There are a few projects that we've forked into the [`trimesh`](https://github.com/trimesh/) GitHub organization which you can take a look at. The general idea of the requirements for a new compiled dependency are:

- is actively maintained and has an MIT/BSD compatible license.
- has all source code in the repository or as a submodule, i.e. no mysterious binary blobs.
- binding preferably uses [pybind11](https://pybind11.readthedocs.io/en/stable/index.html), [nanobind](https://github.com/wjakob/nanobind) or [maturin/py03](https://github.com/PyO3/maturin) for Rust projects. Cython is also OK but other options are preferable if possible. 
- uses `cibuildwheel` to publish releases configured in `pyproject.toml`.
- has unit tests which run in CI
- has minimal dependencies: ideally only `numpy`.