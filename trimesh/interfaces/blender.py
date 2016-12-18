from .generic import MeshScript
from ..resources import get_resource

from distutils.spawn import find_executable

import os
import platform

_search_path = os.environ['PATH']
if platform.system() == 'Windows':
    _search_path += 'C:\Program Files\Blender Foundation\Blender;'
    _search_path += 'C:\Program Files (x86)\Blender Foundation\Blender;'

_blender_executable = find_executable('blender', path=_search_path)
_blender_template = get_resource('blender.py.template')

exists = _blender_executable is not None


def boolean(meshes, operation='difference'):
    if not exists:
        raise ValueError('No blender available!')
    operation = str.upper(operation)
    if operation == 'INTERSECTION':
        operation = 'INTERSECT'

    script = _blender_template.replace('$operation', operation)

    with MeshScript(meshes=meshes,
                    script=script) as blend:
        result = blend.run(_blender_executable +
                           ' --background --python $script')

    # blender returns actively incorrect face normals
    result['face_normals'] = None
    return result
