import os

from .. import util
from .. import visual

from ..base import Trimesh
from ..points import PointCloud
from ..scene.scene import Scene, append_scenes
from ..constants import log_time, log

from . import misc
from .ply import _ply_loaders
from .stl import _stl_loaders
from .dae import _collada_loaders
from .misc import _misc_loaders
from .gltf import _gltf_loaders
from .assimp import _assimp_loaders
from .threemf import _three_loaders
from .openctm import _ctm_loaders
from .wavefront import _obj_loaders
from .xml_based import _xml_loaders


try:
    from ..path.exchange.load import load_path, path_formats
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
    loaders = mesh_formats()
    loaders.extend(path_formats())
    loaders.extend(compressed_loaders.keys())

    return loaders


def load(file_obj,
         file_type=None,
         resolver=None,
         **kwargs):
    """
    Load a mesh or vectorized path into objects:
    Trimesh, Path2D, Path3D, Scene

    Parameters
    ---------
    file_obj : str, or file- like object
      The source of the data to be loadeded
    file_type: str
      What kind of file type do we have (eg: 'stl')
    resolver : trimesh.visual.Resolver
      Object to load referenced assets like materials and textures
    kwargs : **
      Passed to geometry __init__

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
     opened,    # bool, did we open the file ourselves
     resolver   # object to load referenced resources
     ) = parse_file_args(file_obj=file_obj,
                         file_type=file_type,
                         resolver=resolver)

    try:
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
                               resolver=resolver,
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
                raise ValueError('File type: %s not supported' %
                                 file_type)
    finally:
        # close any opened files even if we crashed out
        if opened:
            file_obj.close()

    # add load metadata ('file_name') to each loaded geometry
    for i in util.make_sequence(loaded):
        i.metadata.update(metadata)

    # if we opened the file in this function ourselves from a
    # file name clean up after ourselves by closing it
    if opened:
        file_obj.close()

    return loaded


@log_time
def load_mesh(file_obj,
              file_type=None,
              resolver=None,
              **kwargs):
    """
    Load a mesh file into a Trimesh object

    Parameters
    -----------
    file_obj : str or file object
      File name or file with mesh data
    file_type : str or None
      Which file type, e.g. 'stl'
    kwargs : dict
      Passed to Trimesh constructor

    Returns
    ----------
    mesh : trimesh.Trimesh or trimesh.Scene
      Loaded geometry data
    """

    # parse the file arguments into clean loadable form
    (file_obj,  # file- like object
     file_type,  # str, what kind of file
     metadata,  # dict, any metadata from file name
     opened,    # bool, did we open the file ourselves
     resolver   # object to load referenced resources
     ) = parse_file_args(file_obj=file_obj,
                         file_type=file_type,
                         resolver=resolver)

    try:
        # make sure we keep passed kwargs to loader
        # but also make sure loader keys override passed keys
        results = mesh_loaders[file_type](file_obj,
                                          file_type=file_type,
                                          resolver=resolver,
                                          **kwargs)

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
    finally:
        # if we failed to load close file
        if opened:
            file_obj.close()

    return loaded


def load_compressed(file_obj,
                    file_type=None,
                    resolver=None,
                    mixed=False,
                    **kwargs):
    """
    Given a compressed archive load all the geometry that
    we can from it.

    Parameters
    ----------
    file_obj : open file-like object
      Containing compressed data
    file_type : str
      Type of the archive file
    mixed : bool
      If False, for archives containing both 2D and 3D
      data will only load the 3D data into the Scene.

    Returns
    ----------
    scene : trimesh.Scene
      Geometry loaded in to a Scene object
    """

    # parse the file arguments into clean loadable form
    (file_obj,  # file- like object
     file_type,  # str, what kind of file
     metadata,  # dict, any metadata from file name
     opened,    # bool, did we open the file ourselves
     resolver   # object to load referenced resources
     ) = parse_file_args(file_obj=file_obj,
                         file_type=file_type,
                         resolver=resolver)

    try:
        # a dict of 'name' : file-like object
        files = util.decompress(file_obj=file_obj,
                                file_type=file_type)
        # store loaded geometries as a list
        geometries = []

        # so loaders can access textures/etc
        resolver = visual.resolvers.ZipResolver(files)

        # try to save the files with meaningful metadata
        if 'file_path' in metadata:
            archive_name = metadata['file_path']
        else:
            archive_name = 'archive'

        # populate our available formats
        if mixed:
            available = available_formats()
        else:
            # all types contained in ZIP archive
            contains = set(util.split_extension(n).lower()
                           for n in files.keys())
            # if there are no mesh formats available
            if contains.isdisjoint(mesh_formats()):
                available = path_formats()
            else:
                available = mesh_formats()

        for name, data in files.items():
            # only load formats that we support
            compressed_type = util.split_extension(name).lower()
            if compressed_type not in available:
                # don't raise an exception, just try the next one
                continue
            # store the file name relative to the archive
            metadata['file_name'] = (archive_name + '/' +
                                     os.path.basename(name))
            # load the individual geometry
            loaded = load(file_obj=data,
                          file_type=compressed_type,
                          resolver=resolver,
                          metadata=metadata,
                          **kwargs)

            # some loaders return multiple geometries
            if util.is_sequence(loaded):
                # if the loader has returned a list of meshes
                geometries.extend(loaded)
            else:
                # if the loader has returned a single geometry
                geometries.append(loaded)

    finally:
        # if we opened the file in this function
        # clean up after ourselves
        if opened:
            file_obj.close()

    # append meshes or scenes into a single Scene object
    result = append_scenes(geometries)

    return result


def load_remote(url, **kwargs):
    """
    Load a mesh at a remote URL into a local trimesh object.

    This must be called explicitly rather than automatically
    from trimesh.load to ensure users don't accidentally make
    network requests.

    Parameters
    ------------
    url : string
      URL containing mesh file
    **kwargs : passed to `load`
    """
    # import here to keep requirement soft
    import requests

    # download the mesh
    response = requests.get(url)
    # wrap as file object
    file_obj = util.wrap_as_stream(response.content)

    # so loaders can access textures/etc
    resolver = visual.resolvers.WebResolver(url)

    # actually load
    loaded = load(file_obj=file_obj,
                  file_type=url,
                  resolver=resolver,
                  **kwargs)
    return loaded


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
        """
        Load information with vertices and faces into a mesh
        or PointCloud object.
        """
        if (isinstance(kwargs['vertices'], dict) or
                isinstance(kwargs['faces'], dict)):
            return Trimesh(**misc.load_dict(kwargs))
        elif kwargs['faces'] is None:
            # vertices without faces returns a PointCloud
            return PointCloud(**kwargs)
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

    # function : list of expected keys
    handlers = {handle_scene: ('graph', 'geometry'),
                handle_trimesh_kwargs: ('vertices', 'faces'),
                handle_trimesh_export: ('file_type', 'data')}

    # loop through handler functions and expected key
    handler = None
    for func, expected in handlers.items():
        if all(i in kwargs for i in expected):
            # all expected kwargs exist
            handler = func
            # exit the loop as we found one
            break

    if handler is None:
        raise ValueError('unable to determine type!')

    return handler()


def parse_file_args(file_obj,
                    file_type,
                    resolver=None,
                    **kwargs):
    """
    Given a file_obj and a file_type try to turn them into a file-like
    object and a lowercase string of file type.

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

               str: string is a valid URL
                    -------------------------------------------
                    file_obj: an open 'rb' file object with retrieved data
                    file_type: from the extension

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
        raise ValueError('file_type must be set when passing file objects!')
    if util.is_string(file_obj):
        try:
            # os.path.isfile will return False incorrectly
            # if we don't give it an absolute path
            file_path = os.path.expanduser(file_obj)
            file_path = os.path.abspath(file_path)
            exists = os.path.isfile(file_path)
        except BaseException:
            exists = False

        # file obj is a string which exists on filesystm
        if exists:
            # if not passed create a resolver to find other files
            if resolver is None:
                resolver = visual.resolvers.FilePathResolver(file_path)
            # save the file name and path to metadata
            metadata['file_path'] = file_path
            metadata['file_name'] = os.path.basename(file_obj)
            # if file_obj is a path that exists use extension as file_type
            if file_type is None:
                file_type = util.split_extension(
                    file_path,
                    special=['tar.gz', 'tar.bz2'])
            # actually open the file
            file_obj = open(file_path, 'rb')
            opened = True
        else:
            if '{' in file_obj:
                # if a dict bracket is in the string, its probably a straight
                # JSON
                file_type = 'json'
            elif 'https://' in file_obj or 'http://' in file_obj:
                # we've been passed a URL, warn to use explicit function
                # and don't do network calls via magical pipeline
                raise ValueError(
                    'use load_remote to load URL: {}'.format(file_obj))
            elif file_type is None:
                raise ValueError('string is not a file: {}'.format(file_obj))

    if file_type is None:
        file_type = file_obj.__class__.__name__

    if util.is_string(file_type) and '.' in file_type:
        # if someone has passed the whole filename as the file_type
        # use the file extension as the file_type
        if 'file_path' not in metadata:
            metadata['file_path'] = file_type
        metadata['file_name'] = os.path.basename(file_type)
        file_type = util.split_extension(file_type)
        if resolver is None and os.path.exists(file_type):
            resolver = visual.resolvers.FilePathResolver(file_type)

    # all our stored extensions reference in lower case
    file_type = file_type.lower()

    # if we still have no resolver try using file_obj name
    if (resolver is None and
        hasattr(file_obj, 'name') and
            len(file_obj.name) > 0):
        resolver = visual.resolvers.FilePathResolver(file_obj.name)

    return file_obj, file_type, metadata, opened, resolver


# loader functions for compressed extensions
compressed_loaders = {'zip': load_compressed,
                      'tar.bz2': load_compressed,
                      'tar.gz': load_compressed}

# map file_type to loader function
mesh_loaders = {}
# assimp has a lot of loaders, but they are all quite slow
# load first and replace with native loaders where possible
mesh_loaders.update(_assimp_loaders)
mesh_loaders.update(_stl_loaders)
mesh_loaders.update(_ctm_loaders)
mesh_loaders.update(_misc_loaders)
mesh_loaders.update(_ply_loaders)
mesh_loaders.update(_xml_loaders)
mesh_loaders.update(_obj_loaders)
mesh_loaders.update(_collada_loaders)
mesh_loaders.update(_gltf_loaders)
mesh_loaders.update(_three_loaders)
