Mesh Formats
=======================

There's lots of mesh formats out there!


## Which Mesh Format Should I Use?

Quick recommendation: `GLB` or `PLY`. If you're looking for "the simplest possible format because I have to write an importer in another language" you should take a look at `OFF`. 

Wavefront `OBJ` is also pretty common: unfortunately OBJ doesn't have a widely accepted specification so every importer and exporter implements things slightly differently, making it tough to support. It also allows unfortunate things like arbitrary sized polygons, has a face representation which is easy to mess up, references other files for materials and textures, arbitrarily interleaves data, and is slow to parse. Give `GLB` a try as an alternative!


| Format | Dependencies | Notes | 
| ------ | ------------ | ----- |
| `GLB`/`GLTF` | | The mesh format of choice. A little more complicated since it has a scene representation, but all vertices and faces are stored as binary buffers described in a JSON header, which means it both imports and exports very quickly. It also supports most things you'd want including texture, point clouds, etc. |
| `STL`  | | Has both ASCII and binary represenations, both of which are supported. This is probably the most basic format, and it consists of a "triangle soup" with no shared vertices. This format is the reason why `trimesh` has the default `process=True` which merges vertices since an unindexed triangle soup isn't super useful. |
| `PLY`  | | Has ASCII+Binary representations, both are supported. A nice upgrade over STL since it consists of indexed triangles. [Overview](https://paulbourke.net/dataformats/ply/index.html)