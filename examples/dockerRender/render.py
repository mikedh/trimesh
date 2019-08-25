import trimesh


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

    # the PNG is just bytes data
    print('rendered bytes:', len(png))

    # write the render to a volume we should have docker mounted
    with open('/output/render.png', 'wb') as f:
        f.write(png)

    # NOTE: if you raise an exception, supervisord will automatically
    # restart this script which is pretty useful in long running workers
