# flake8: noqa
import os


def delete_nonresult(bpy):
    # different versions of blender sometimes return the wrong mesh
    objects = bpy.context.scene.objects
    if len(objects) <= 1:
        return

    try:
        # earlier than blender <2.8
        objects[-1].select = False
        for other in objects[:-1]:
            other.select = True
        bpy.ops.object.delete()
        objects[-1].select = True
    except AttributeError:
        # blender 2.8 changed this
        ob = objects[-1]
        ob.select_set(False)
        for other in objects[:-1]:
            other.select_set(True)
        bpy.ops.object.delete()
        objects[0].select_set(True)


if __name__ == "__main__":
    import bpy
    # clear scene of default box
    bpy.ops.wm.read_homefile()
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except BaseException:
        pass
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=True)

    # get temporary files from templated locations
    mesh_pre = $MESH_PRE
    mesh_post = os.path.abspath(r'$MESH_POST')

    # When you add objects to blender, other elements are pushed back
    # by going last to first on filenames we can preserve the index
    for filename in mesh_pre[::-1]:
        bpy.ops.import_mesh.stl(filepath=os.path.abspath(filename))

    mesh = bpy.context.scene.objects[0]
    for other in bpy.context.scene.objects[1:]:
        # add boolean modifier
        mod = mesh.modifiers.new('boolean', 'BOOLEAN')
        mod.object = other
        mod.operation = '$OPERATION'
        bpy.ops.object.modifier_apply(modifier='boolean')

    delete_nonresult(bpy)
    bpy.ops.export_mesh.stl(
        filepath=mesh_post,
        use_mesh_modifiers=True)
