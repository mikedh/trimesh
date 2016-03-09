
import sys
import numpy as np
import trimesh


if __name__ == '__main__':
    # print logged messages
    trimesh.util.attach_to_log()

    # load a mesh
    mesh  = trimesh.load('../models/featuretype.STL')

    # get a scene object containing the mesh, this is equivalent to:
    # scene = trimesh.scene.Scene(mesh)
    scene = mesh.scene()

    r = trimesh.transformations.rotation_matrix(np.radians(45.0), [0,1,0], 
                                                scene.centroid)    
    for i in range(4):
        trimesh.constants.log.info('Saving image %d', i)
        
        # rotate the camera view
        camera_new = np.dot(scene.transforms.get('camera'), r)
        scene.transforms.update('camera', matrix=camera_new)

        file_name = 'render_' + str(i) + '.png'

        # saving an image requires an opengl context, so if -nw
        # is passed don't save the image
        if not '-nw' in sys.argv:
            # save a render of the object as a png
            scene.save_image(file_name,
                             resolution=np.array([1920,1080])*2)
