import numpy as np

import os

from .. import util

from ..base import Trimesh
from ..scene.scene import Scene, append_scenes
from ..constants import _log_time, log

from . import misc
from .ply import _ply_loaders
from .stl import _stl_loaders
from .misc import _misc_loaders
from .gltf import _gltf_loaders
from .assimp import _assimp_loaders
from .threemf import _three_loaders
from .openctm import _ctm_loaders
from .wavefront import _obj_loaders
from .xml_based import _xml_loaders


try:
    from ..path.io.load import load_path, path_formats
except BaseException as E:
    # save a traceback to see why path didn't import
    _path_exception = E

    def load_path(*args, **kwargs):
        """
        Dummy load path function that will raise an exception
        on use. Import of path failed, probably because a
        dependency is not installed.

        Raises
        ----------
        path_exception : Whatever failed when we imported path
        """
        raise _path_exception

    def path_formats():
        return []


def mesh_formats():
    """
    Get a list of mesh formats

    Returns
    -----------
    loaders : list
        Extensions of available mesh loaders
        i.e. 'stl', 'ply', etc.
    """
    return list(mesh_loaders.keys())


def available_formats():
    """
    Get a list of all available loaders

    Returns
    -----------
    loaders : list
        Extensions of available loaders
        i.e. 'stl', 'ply', 'dxf', etc.
    """

    loaders = np.hstack((list(compressed_loaders.keys()),
                         mesh_formats(),
                         path_formats()))
    return loaders


def load(file_obj, file_type=None, **kwargs):
    """
    Load a mesh or vectorized path into objects:
    Trimesh, Path2D, Path3D, Scene

    Parameters
    ---------
    file_obj : str, or file- like object
        The source of the data to be loadeded
    file_type: str
         What kind of file type do we have (eg: 'stl')

    Returns
    ---------
    geometry : Trimesh, Path2D, Path3D, Scene
        Loaded geometry as trimesh classes
    """
    # check to see if we're trying to load something
    # that is already a native trimesh object
    # do the check by name to avoid circular imports
    out_types = ('Trimesh', 'Path', 'Scene')
    if any(util.is_instance_named(file_obj, t)
           for t in out_types):
        log.info('Loaded called on %s object, returning input',
                 file_obj.__class__.__name__)
        return file_obj

    # parse the file arguments into clean loadable form
    (file_obj,  # file- like object
     file_type,  # str, what kind of file
     metadata,  # dict, any metadata from file name
     opened     # bool, did we open the file ourselves
     ) = _parse_file_args(file_obj, file_type)

    if isinstance(file_obj, dict):
        # if we've been passed a dict treat it as kwargs
        kwargs.update(file_obj)
        loaded = load_kwargs(kwargs)
    elif file_type in path_formats():
        # path formats get loaded with path loader
        loaded = load_path(file_obj,
                           file_type=file_type,
                           **kwargs)
    elif file_type in mesh_loaders:
        # mesh loaders use mesh loader
        loaded = load_mesh(file_obj,
                           file_type=file_type,
                           **kwargs)
    elif file_type in compressed_loaders:
        # for archives, like ZIP files
        loaded = load_compressed(file_obj,
                                 file_type=file_type,
                                 **kwargs)
    else:
        if file_type in ['svg', 'dxf']:
            # call the dummy function to raise the import error
            # this prevents the exception from being super opaque
            load_path()
        else:
            raise ValueError('File type: %s not supported',
                             file_type)

    for i in util.make_sequence(loaded):
        # check to make sure loader actually loaded something
        i.metadata.update(metadata)

    # if we opened the file in this function from a file name
    # clean up after ourselves by closing it
    if opened:
        file_obj.close()

    return loaded


@_log_time
def load_mesh(file_obj, file_type=None, **kwargs):
    """
    Load a mesh file into a Trimesh object

    Parameters
    ---------
    file_obj:  str or file-like object
    file_type: str representing file type (eg: 'stl')
    kwargs:    passed to Trimesh constructor

    Returns:
    ----------
    mesh: Trimesh object, or a list of Trimesh objects
          depending on the file format.

    """
    # turn a string into a file obj and type
    (file_obj,
     file_type,
     metadata,
     opened) = _parse_file_args(file_obj, file_type)

    # make sure we keep passed kwargs to loader
    # but also make sure loader keys override passed keys
    results = mesh_loaders[file_type](file_obj,
                                      file_type=file_type)

    if util.is_file(file_obj):
        file_obj.close()

    log.debug('loaded mesh using %s',
              mesh_loaders[file_type].__name__)

    if not isinstance(results, list):
        results = [results]

    loaded = []
    for result in results:
        kwargs.update(result)
        loaded.append(load_kwargs(kwargs))
        loaded[-1].metadata.update(metadata)
    if len(loaded) == 1:
        loaded = loaded[0]

    if opened:
        file_obj.close()

    return loaded


def load_compressed(file_obj, file_type=None):
    """
    Given a compressed archive, load all the geometry that we can from it.

    Parameters
    ----------
    file_obj: open file-like object
    file_type: str, type of file

    Returns
    ----------
    geometries: list of geometry objects
    """
    # turn a string into a file obj and type
    (file_obj,
     file_type,
     metadata,
     opened) = _parse_file_args(file_obj, file_type)

    # a dict of 'name' : file-like object
    files = util.decompress(file_obj=file_obj,
                            file_type=file_type)
    # store loaded geometries as a list
    geometries = []

    # try to save the files with meaningful metadata
    if 'file_path' in metadata:
        archive_name = metadata['file_path']
    else:
        archive_name = 'archive'

    for name, data in files.items():
        # only load formats that we support
        compressed_type = util.split_extension(name).lower()
        if compressed_type not in available_formats():
            # don't raise an exception, just try the next one
            continue
        # store the file name relative to the archive
        metadata['file_name'] = (archive_name + '/' +
                                 os.path.basename(name))
        # load the individual geometry
        geometry = load(file_obj=data,
                        file_type=compressed_type,
                        metadata=metadata)
        geometries.append(geometry)

    # if we opened the file in this function
    # clean up after ourselves
    if opened:
        file_obj.close()

    # append meshes or scenes into a single Scene object
    result = append_scenes(geometries)

    return result


def load_kwargs(*args, **kwargs):
    """
    Load geometry from a properly formatted dict or kwargs
    """
    def handle_scene():
        """
        Load a scene from our kwargs:

        class:      Scene
        geometry:   dict, name: Trimesh kwargs
        graph:      list of dict, kwargs for scene.graph.update
        base_frame: str, base frame of graph
        """
        scene = Scene()
        scene.geometry.update({k: load_kwargs(v) for
                               k, v in kwargs['geometry'].items()})
        for k in kwargs['graph']:
            if isinstance(k, dict):
                scene.graph.update(**k)
            elif util.is_sequence(k) and len(k) == 3:
                scene.graph.update(k[1], k[0], **k[2])
        if 'base_frame' in kwargs:
            scene.graph.base_frame = kwargs['base_frame']
        if 'metadata' in kwargs:
            scene.metadata.update(kwargs['metadata'])

        return scene

    def handle_trimesh_kwargs():
        if (isinstance(kwargs['vertices'], dict) or
                isinstance(kwargs['faces'], dict)):
            return Trimesh(**misc.load_dict(kwargs))
        else:
            return Trimesh(**kwargs)

    def handle_trimesh_export():
        data, file_type = kwargs['data'], kwargs['file_type']
        if not isinstance(data, dict):
            data = util.wrap_as_stream(data)
        k = mesh_loaders[file_type](data,
                                    file_type=file_type)
        return Trimesh(**k)

    # if we've been passed a single dict instead of kwargs
    # substitute the dict for kwargs
    if (len(kwargs) == 0 and
        len(args) == 1 and
            isinstance(args[0], dict)):
        kwargs = args[0]

    handlers = {handle_scene: ('graph', 'geometry'),
                handle_trimesh_kwargs: ('vertices', 'faces'),
                handle_trimesh_export: ('file_type', 'data')}

    handler = None
    for k, v in handlers.items():
        if all(i in kwargs for i in v):
            handler = k
            break
    if handler is None:
        raise ValueError('unable to determine type!')

    return handler()


def _parse_file_args(file_obj, file_type, **kwargs):
    """
    Given a file_obj and a file_type, try to turn them into a file-like object
    and a lowercase string of file type

    Parameters
    -----------
    file_obj:  str: if string represents a file path, returns
                    -------------------------------------------
                    file_obj:   an 'rb' opened file object of the path
                    file_type:  the extension from the file path

               str: if string is NOT a path, but has JSON-like special characters
                    -------------------------------------------
                    file_obj:   the same string passed as file_obj
                    file_type:  set to 'json'

               str: string is not an existing path or a JSON-like object
                    -------------------------------------------
                    ValueError will be raised as we can't do anything with input

               file like object: we cannot grab information on file_type automatically
                    -------------------------------------------
                    ValueError will be raised if file_type is None
                    file_obj:  same as input
                    file_type: same as input

               other object: like a shapely.geometry.Polygon, etc:
                    -------------------------------------------
                    file_obj:  same as input
                    file_type: if None initially, set to the class name
                               (in lower case), otherwise passed through

    file_type: str, type of file and handled according to above

    Returns
    -----------
    file_obj:  loadable object
    file_type: str, lower case of the type of file (eg 'stl', 'dae', etc)
    metadata:  dict, any metadata
    opened:    bool, did we open the file or not
    """
    metadata = {}
    opened = False
    if ('metadata' in kwargs and
            isinstance(kwargs['metadata'], dict)):
        metadata.update(kwargs['metadata'])

    if util.is_file(file_obj) and file_type is None:
        raise ValueError(
            'File type must be specified when passing file objects!')
    if util.is_string(file_obj):
        try:
            # os.path.isfile will return False incorrectly
            # if we don't give it an absolute path
            file_path = os.path.expanduser(file_obj)
            file_path = os.path.abspath(file_path)
            exists = os.path.isfile(file_path)
        except BaseException:
            exists = False

        if exists:
            metadata['file_path'] = file_path
            metadata['file_name'] = os.path.basename(file_obj)
            # if file_obj is a path that exists use extension as file_type
            if file_type is None:
                file_type = util.split_extension(file_path,
                                                 special=['tar.gz', 'tar.bz2'])
            file_obj = open(file_path, 'rb')
            opened = True
        else:
            if file_type is not None:
                return file_obj, file_type, metadata
            elif '{' in file_obj:
                # if a dict bracket is in the string, its probably a straight
                # JSON
                file_type = 'json'
            else:
                raise ValueError(
                    'File object passed as string that is not a file!')

    if file_type is None:
        file_type = file_obj.__class__.__name__

    if util.is_string(file_type) and '.' in file_type:
        # if someone has passed the whole filename as the file_type
        # use the file extension as the file_type
        if 'file_path' not in metadata:
            metadata['file_path'] = file_type
        metadata['file_name'] = os.path.basename(file_type)
        file_type = util.split_extension(file_type)
    file_type = file_type.lower()
    return file_obj, file_type, metadata, opened


compressed_loaders = {'zip': load_compressed,
                      'tar.bz2': load_compressed,
                      'tar.gz': load_compressed}
mesh_loaders = {}
# assimp has a lot of loaders, but they are all quite slow
# so we load them first and replace them with native loaders if possible
mesh_loaders.update(_assimp_loaders)
mesh_loaders.update(_stl_loaders)
mesh_loaders.update(_ctm_loaders)
mesh_loaders.update(_misc_loaders)
mesh_loaders.update(_ply_loaders)
mesh_loaders.update(_xml_loaders)
mesh_loaders.update(_obj_loaders)
mesh_loaders.update(_gltf_loaders)
mesh_loaders.update(_three_loaders)
