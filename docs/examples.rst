Examples
=====================

A few examples are available as rendered IPython notebooks.

A tour of some basic features: 
 `Quick start <examples/quick_start.html>`__

How to slice a mesh using trimesh: 
 `Cross sections <examples/section.html>`__

How do do signed distance and proximity queries: 
 `Nearest point queries <examples/nearest.html>`__

Find a path from one mesh vertex to another, travelling along edges of the mesh:
 `Edge graph traversal <examples/shortest.html>`__

Do simple ray- mesh queries, including finding the indexes of triangles hit by rays, the locations of points hit on the mesh surface, etc. Ray queries have the same API with two available backends, one implemented in pure numpy and one that requires pyembree but is 50x faster.
 `Ray-mesh queries <examples/ray.html>`__
