import trimesh
from pyglet import gl


if __name__ == '__main__':
    # print logged messages
    trimesh.util.attach_to_log()

    # make a sphere
    mesh = trimesh.creation.icosphere()

    # get a scene object containing the mesh, this is equivalent to:
    # scene = trimesh.scene.Scene(mesh)
    scene = mesh.scene()

    # set a GL config that fixes a depth buffer issue in xvfb
    window_conf = gl.Config(double_buffer=True, depth_size=24)
    # run the actual render call
    png = scene.save_image(resolution=[1920, 1080],
                           window_conf=window_conf)

    # the PNG is just bytes data
    trimesh.util.log.info('rendered bytes:', len(png))

    # write the render to a volume we should have docker mounted
    with open('/output/output.png', 'wb') as f:
        f.write(png)
