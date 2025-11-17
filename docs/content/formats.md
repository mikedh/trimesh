Mesh Formats
=======================

There's lots of mesh formats out there!


## Which Mesh Format Should I Use?

Quick recommendation: `GLB`. If you're looking for "the simplest possible format because I have to write an importer in another language" you should take a look at `OFF`. 

Wavefront `OBJ` is also widely used and supported across many tools, although it lacks a specification and has a slightly unusual internal data structure, which can lead to compatibility and indexing issues. Parsing is much slower than binary formats. 

## What Formats Does Trimesh Support?

No dependencies indicates a minimal install with just numpy can load the format.

| Format | Dependencies | Notes | 
| ------ | ------------ | ----- |
| `GLB`/`GLTF` | | A scene format with flat arrays (i.e. very quick loading), wide support, and a well defined specification. Most people should use this is possible. It also supports most things you'd want including texture, point clouds, etc. |
| `STL`  | | Has both ASCII and binary representations, both of which are supported. This is a very basic format, and it consists of a "triangle soup" with no shared vertices. This format is the reason why `trimesh` has the default `process=True` which merges vertices since an unindexed triangle soup isn't super useful. |
| `PLY`  | | Has both ASCII and Binary representations, both are supported. It has indexed triangles, and a header format that supports named properties. [Overview](https://paulbourke.net/dataformats/ply/index.html)
| `OBJ`  | | Wavefront OBJ, relatively slow to parse and may require re-indexing to match the `trimesh` "matching arrays" data structure. |
| `OFF`  | | Text format that is just vertices and faces in ASCII |
| `3MF`  | `lxml`, `networkx` | An XML based 3D printing focused mesh format. |
| `3DXML` | `lxml`, `networkx`, `PIL` | Dassault's XML format from CATIA/SolidWorks, the easiest way to get from Solidworks to a nice 3D scene. |
| `DAE`/`ZAE` | `lxml`, `PIL`, `pycollada` | COLLADA, an XML-based scene format that supports everything. |
| `STEP`/`STP` | `cascadio` | The only boundary representation format with open-source loaders available. `cascadio` uses OpenCASCADE to convert to GLB before loading. |
| `XAML` | `lxml` | Microsoft's 3D XAML format which exports from Solidworks |
| `DXF` | | AutoCAD's drawing format, we only support the ASCII version with 2D geometry. |
| `SVG` | | We import 2D paths as `trimesh.Path2D` from these, discarding pretty much all visual information other than the curves. |
| `XYZ` | | Simple ASCII point cloud format with just one coordinate per line. |
