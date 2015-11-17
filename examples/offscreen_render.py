import numpy as np
import trimesh


if __name__ == '__main__':    
    trimesh.util.attach_to_log()

    # load a mesh
    mesh  = trimesh.load_mesh('../models/angle_block.STL')
    # get a scene object containing the mesh
    # this is equivilant to:
    # scene = trimesh.scene.Scene(mesh)
    scene = mesh.scene()

    # loop over angles between 0 and pi
    for i, angle in enumerate(np.linspace(0, np.pi, 10)):
        trimesh.constants.log.info('Saving image %d', i)
        # move the camera of the scene around the object
        scene.set_camera(angles=[0,angle,0])
        # save a render of the object as a png
        scene.save_image('render_' + str(i) + '.png', resolution=np.array([1920,1080])*2)
    
