import os
import trimesh
import numpy as np


if __name__ == '__main__':
    # print logged messages
    trimesh.util.attach_to_log()

    # make a sphere
    mesh = trimesh.creation.icosphere()

    # get a scene object containing the mesh, this is equivalent to:
    # scene = trimesh.scene.Scene(mesh)
    scene = mesh.scene()

    # run the actual render call
    png = scene.save_image(resolution=[1920, 1080], visible=True)

    # block somehow, otherwise supervisord will restart the process
    # this is excellent if you make an application which polls some
    # queue and then sleeps for a while before exiting
    print('rendered bytes:', len(png))

    # write the render to a volume we should have docker mounted
    with open('/output/render.png', 'wb') as f:
        f.write(png)
