Examples
=====================

A few examples are available as rendered IPython notebooks.


	  
Quick Tour
'''''''''''''

A tour of some basic features, like inertia, viewing in notebooks, etc. 
`Quick start <examples/quick_start.html>`_


Cross Sections
'''''''''''''

How to take planar slices of a mesh, like you would for 3D printing, using trimesh.
`Cross sections <examples/section.html>`_

Proximity Queries
'''''''''''''''''

How do do signed distance and proximity queries on meshes.
`Nearest point queries <examples/nearest.html>`_

Path Finding
'''''''''''''

Find a path from one mesh vertex to another, travelling along edges of the mesh.
`Edge graph traversal <examples/shortest.html>`_

Ray Tests
'''''''''''''

Do simple ray- mesh queries, including finding the indexes of triangles hit by rays, the locations of points hit on the mesh surface, etc. Ray queries have the same API with two available backends, one implemented with just Numpy calls, and one that requires pyembree but is 50x faster (a team at Intel wrote it).
`Ray-mesh queries <examples/ray.html>`_
