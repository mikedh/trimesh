"""
save_image.py
----------------

Save an image of the isometric view of a mesh
"""

import trimesh
import math

if __name__ == '__main__':
    # load a file by name
    mesh = trimesh.load_mesh('../models/featuretype.STL')

    # Create a scene
    scene = trimesh.Scene()

    # Add the chosen geometry to the scene
    scene.add_geometry(mesh)

    # Get the bounds corners for the camera transform
    corners = scene.bounds_corners

    # Get isometric view
    r_e = trimesh.transformations.euler_matrix(
        math.radians(45),
        math.radians(45),
        math.radians(45),
        "ryxz",
    )

    # Get camera transform to look at geometry with isometric view
    t_r = scene.camera.look_at(corners, rotation=r_e)

    # Set camera transform
    scene.camera_transform = t_r

    # Render of scene as a PNG bytes
    png = scene.save_image()

    # Write the bytes to file
    with open('../models/featuretype.png', "wb") as f:
        f.write(png)
        f.close()
