from .generic import MeshScript

from distutils.spawn import find_executable

import inspect
import os

_blender_executable = find_executable('blender')
exists = _blender_executable is not None

def boolean(meshes, operation='difference'):
    if not exists:
        raise ValueError('No blender available!')
    operation = str.upper(operation)
    if operation == 'INTERSECTION': 
        operation = 'INTERSECT'

    current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    script = open(os.path.join(current, '../templates/blender.py.template'),'rb').read()
    script = script.replace('$operation', operation)
  
    with MeshScript(meshes = meshes, 
                    script = script) as blend:
        result = blend.run(_blender_executable + ' --background --python $script')

    # blender returns actively incorrect face normals
    result['face_normals'] = None
    return result


