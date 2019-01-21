import os
import platform

from .generic import MeshScript
from ..constants import log

from distutils.spawn import find_executable

_search_path = os.environ['PATH']
if platform.system() == 'Windows':
    # split existing path by delimiter
    _search_path = [i for i in _search_path.split(';') if len(i) > 0]
    _search_path.append(r'C:\Program Files')
    _search_path.append(r'C:\Program Files (x86)')
    _search_path = ';'.join(_search_path)
    log.debug('searching for vhacd in: %s', _search_path)


_vhacd_executable = None
for _name in ['vhacd', 'testVHACD']:
    _vhacd_executable = find_executable(_name, path=_search_path)
    if _vhacd_executable is not None:
        break

exists = _vhacd_executable is not None


def convex_decomposition(mesh, **kwargs):
    if not exists:
        raise ValueError('No vhacd available!')

    argstring = ' --input $mesh_0 --output $mesh_post --log $script'

    # pass through extra arguments from the input dictionary
    for key, value in kwargs.items():
        argstring += ' --{} {}'.format(str(key),
                                       str(value))

    with MeshScript(meshes=[mesh],
                    script='',
                    tmpfile_ext='obj') as vhacd:
        result = vhacd.run(_vhacd_executable + argstring)
    return result
