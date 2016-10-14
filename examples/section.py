'''
examples/section.py

Slice a mesh into 2D sections, like you would do to 
write a basic 3D printing driver. 
'''

import sys
import trimesh
import numpy as np

if __name__ == '__main__':
    # print trimesh log messages to console
    trimesh.util.attach_to_log()

    # load the mesh from filename
    # file objects are also supported
    mesh = trimesh.load_mesh('../models/featuretype.STL')

    # we're going to slice the mesh into evenly spaced chunks along z
    # this takes the (2,3) bounding box and slices it into [minz, maxz]
    z_extents = mesh.bounds[:,2]

    # slice every .125 model units (eg, inches)
    z_levels  = np.arange(*z_extents, step=.125)

    # create an array to hold the section objects
    sections  = []

    for z in np.append(z_levels, 1000):
        # this will return a Path3D object, each of which will 
        # have curves in 3D space
        try:
            sections.append(mesh.section(plane_origin = [0,0,z],
                                         plane_normal = [0,0,1]))
        except ValueError:
            print('Plane doesn\'t intersect mesh at {}'.format(z))
    # summing the array of path objects will put all of the curves
    # into one Path3D object, which we can then plot easily in 3D
    combined = np.sum(sections)
    
    # we can also transform each section in space onto the XY plane
    # note that if the Path3D object couldn't find a plane which all 
    # of the entities lie on, it will raise a MeshError
    section_2D, T_matrix = sections[0].to_planar()

    # if the script is run with the option -nw, don't show the preview windows.
    if not '-nw' in sys.argv:
        combined.show()
        section_2D.show()
