from .. import util

from .generic import MeshScript
from ..resources import get_resource
from ..constants import log

from distutils.spawn import find_executable

import os
import platform

_search_path = os.environ['PATH']
if platform.system() == 'Windows':
    # split existing path by delimiter
    _search_path = [i for i in _search_path.split(';') if len(i) > 0]
    _search_path.append('C:\Program Files\Blender Foundation\Blender')
    _search_path.append('C:\Program Files (x86)\Blender Foundation\Blender')
    _search_path = ';'.join(_search_path)
    log.debug('searching for blender in: %s', _search_path)

if platform.system() == 'Darwin':
    _search_path = [i for i in _search_path.split(':') if len(i) > 0]
    _search_path.append('/Applications/blender.app/Contents/MacOS')
    _search_path = ':'.join(_search_path)
    log.debug('searching for blender in: %s', _search_path)
    log.warning('searching for blender in: %s', _search_path)


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

    for m in util.make_sequence(result):
        # blender returns actively incorrect face normals
        m.face_normals = None

    return result
