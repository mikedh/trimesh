import json
from tempfile import NamedTemporaryFile

from .. import util


def load_dict(file_obj, **kwargs):
    """
    Load multiple input types into kwargs for a Trimesh constructor.
    Tries to extract keys:
    'faces'
    'vertices'
    'face_normals'
    'vertex_normals'

    Parameters
    ----------
    file_obj : dict
    accepts multiple forms
          -dict: has keys for vertices and faces as (n,3) numpy arrays
          -dict: has keys for vertices/faces (n,3) arrays encoded as dicts/base64
                 with trimesh.util.array_to_encoded/trimesh.util.encoded_to_array
          -str:  json blob as dict with either straight array or base64 values
          -file object: json blob of dict
    file_type: not used

    Returns
    -----------
    loaded: dict with keys
            -vertices: (n,3) float
            -faces:    (n,3) int
            -face_normals: (n,3) float (optional)
    """
    if file_obj is None:
        raise ValueError("file_obj passed to load_dict was None!")
    if util.is_instance_named(file_obj, "Trimesh"):
        return file_obj
    if isinstance(file_obj, str):
        if "{" not in file_obj:
            raise ValueError("Object is not a JSON encoded dictionary!")
        file_obj = json.loads(file_obj.decode("utf-8"))
    elif util.is_file(file_obj):
        file_obj = json.load(file_obj)

    # what shape should the file_obj be to be usable
    mesh_file_obj = {
        "vertices": (-1, 3),
        "faces": (-1, (3, 4)),
        "face_normals": (-1, 3),
        "face_colors": (-1, (3, 4)),
        "vertex_normals": (-1, 3),
        "vertex_colors": (-1, (3, 4)),
    }

    # now go through file_obj structure and if anything is encoded as base64
    # pull it back into numpy arrays
    if isinstance(file_obj, dict):
        loaded = {}
        file_obj = util.decode_keys(file_obj, "utf-8")
        for key, shape in mesh_file_obj.items():
            if key in file_obj:
                loaded[key] = util.encoded_to_array(file_obj[key])
                if not util.is_shape(loaded[key], shape):
                    raise ValueError(
                        "Shape of %s is %s, not %s!",
                        key,
                        str(loaded[key].shape),
                        str(shape),
                    )
        if len(key) == 0:
            raise ValueError("Unable to extract any mesh file_obj!")
        return loaded
    else:
        raise ValueError("%s object passed to dict loader!", file_obj.__class__.__name__)


def load_meshio(file_obj, file_type=None, **kwargs):
    """
    Load a meshio-supported file into the kwargs for a Trimesh
    constructor.


    Parameters
    ----------
    file_obj : file object
      Contains a meshio file
    file_type : str
      File extension, aka 'vtk'

    Returns
    ----------
    loaded : dict
      kwargs for Trimesh constructor
    """
    # trimesh "file types" are really filename extensions
    file_formats = meshio.extension_to_filetypes["." + file_type]
    # load_meshio gets passed and io.BufferedReader
    # not all readers can cope with that
    # e.g., the ones that use h5m underneath
    # in that case use the associated file name instead
    mesh = None

    with NamedTemporaryFile(suffix=f".{file_type}") as temp:
        temp.write(file_obj.read())
        temp.flush()

        for file_format in file_formats:
            try:
                mesh = meshio.read(temp.name, file_format=file_format)
                break
            except BaseException:
                util.log.debug("failed to load", exc_info=True)
        if mesh is None:
            raise ValueError("Failed to load file!")

    # save file_obj as kwargs for a trimesh.Trimesh
    result = {}
    # pass kwargs to mesh constructor
    result.update(kwargs)
    # add vertices
    result["vertices"] = mesh.points
    try:
        # add faces
        result["faces"] = mesh.get_cells_type("triangle")
    except BaseException:
        util.log.warning("unable to get faces", exc_info=True)
        result["faces"] = []

    return result


_misc_loaders = {}

try:
    import meshio

    # add meshio loaders here
    _meshio_loaders = {k[1:]: load_meshio for k in meshio.extension_to_filetypes.keys()}
    _misc_loaders.update(_meshio_loaders)
except BaseException:
    _meshio_loaders = {}

try:
    import openctm

    _misc_loaders["ctm"] = openctm.load_ctm
except BaseException:
    pass
