from .generic    import MeshScript
from ..templates import get_template

from distutils.spawn import find_executable

_blender_executable = find_executable('blender')
_blender_template = get_template('blender.py.template')

exists = _blender_executable is not None

def boolean(meshes, operation='difference'):
    if not exists:
        raise ValueError('No blender available!')
    operation = str.upper(operation)
    if operation == 'INTERSECTION': 
        operation = 'INTERSECT'

    script = _blender_template.replace('$operation', operation)
  
    with MeshScript(meshes = meshes, 
                    script = script) as blend:
        result = blend.run(_blender_executable + ' --background --python $script')

    # blender returns actively incorrect face normals
    result['face_normals'] = None
    return result


