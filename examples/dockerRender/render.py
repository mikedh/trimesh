import os
import trimesh

from pyglet import gl


if __name__ == '__main__':
    # print logged messages
    trimesh.util.attach_to_log()

    root = '/trimesh/models'

    window_conf = gl.Config(double_buffer=True,
                     depth_size=24)
    
    for file_name in os.listdir(root):
        try:
            scene = trimesh.load(os.path.join(root, file_name),
                                 force='scene')
            # run the actual render call
            png = scene.save_image(resolution=[1920, 1080], visible=True, window_conf=window_conf)
            # the PNG is just bytes data
            print('rendered bytes:', len(png))
            # write the render to a volume we should have docker mounted
            out_file = '/trimesh/examples/dockerRender/{}.png'.format(
                file_name)
            with open(out_file, 'wb') as f:
                f.write(png)
        except BaseException as E:
            continue
                
