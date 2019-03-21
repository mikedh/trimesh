
import openctm

_ctm_loaders = {}

def load_ctm(file_obj, file_type=None):
    """
    Load OpenCTM files from a file object.

    Parameters
    ----------
    file_obj : open file- like object

    Returns
    ----------
    loaded : dict
              kwargs for a Trimesh constructor:
                {vertices: (n,3) float, vertices
                 faces:    (m,3) int, indexes of vertices}
    """
    # !!load file from name
    # this should be replaced with something that
    # actually uses the file object data to support streams
    name = str(file_obj.name)
    mesh = openctm.import_mesh(name)

    # create kwargs for trimesh constructor
    result = {'vertices': mesh.vertices,
              'faces': mesh.faces}

    if mesh.normals is not None:
        result['face_normals'] = mesh.normals

    return result

# we have a library so add load_ctm
_ctm_loaders = {'ctm': load_ctm}
