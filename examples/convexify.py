'''
examples/convexify.py

Take a mesh with multiple bodies, take the convex hull
of each body, then combine them back into one mesh. 
Useful for generating collision models of an object. 
'''

import trimesh
import numpy as np

if __name__ == '__main__':
    # load the mesh from filename
    # file objects are also supported
    mesh = trimesh.load_mesh('../models/featuretype.STL')

    # split the mesh into connected components of the face adjacency graph. 
    # splitting sometimes produces non- watertight meshes
    # though the splitter will try to repair single quad and
    # single triangle holes, in our case here we are going to be 
    # taking convex hulls anyway, so there is no reason to not discard
    # the non- watertight bodies
    meshes = mesh.split(check_watertight=False)

    # the convex hull of every component
    meshes_convex = [i.convex_hull() for i in meshes]

    # combine all components into one mesh
    convex_combined = np.sum(meshes_convex)

    # open a viewer window
    convex_combined.show()
