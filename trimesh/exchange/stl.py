import numpy as np


class HeaderError(Exception):
    # the exception raised if an STL file object doesn't match its header
    pass


# define a numpy datatype for the data section of a binary STL file
# everything in STL is always Little Endian
# this works natively on Little Endian systems, but blows up on Big Endians
# so we always specify byteorder
_stl_dtype = np.dtype([('normals', '<f4', (3)),
                       ('vertices', '<f4', (3, 3)),
                       ('attributes', '<u2')])
# define a numpy datatype for the header of a binary STL file
_stl_dtype_header = np.dtype([('header', np.void, 80),
                              ('face_count', '<i4')])


def load_stl(file_obj, file_type=None, **kwargs):
    """
    Load an STL file from a file object.

    Parameters
    ----------
    file_obj: open file- like object
    file_type: not used

    Returns
    ----------
    loaded: kwargs for a Trimesh constructor with keys:
              vertices:     (n,3) float, vertices
              faces:        (m,3) int, indexes of vertices
              face_normals: (m,3) float, normal vector of each face
    """
    # save start of file obj
    file_pos = file_obj.tell()
    try:
        # check the file for a header which matches the file length
        # if that is true, it is almost certainly a binary STL file
        # if the header doesn't match the file length a HeaderError will be
        # raised
        return load_stl_binary(file_obj)
    except HeaderError:
        # move the file back to where it was initially
        file_obj.seek(file_pos)
        # try to load the file as an ASCII STL
        # if the header doesn't match the file length a HeaderError will be
        # raised
        return load_stl_ascii(file_obj)


def load_stl_binary(file_obj):
    """
    Load a binary STL file from a file object.

    Parameters
    ----------
    file_obj: open file- like object

    Returns
    ----------
    loaded: kwargs for a Trimesh constructor with keys:
              vertices:     (n,3) float, vertices
              faces:        (m,3) int, indexes of vertices
              face_normals: (m,3) float, normal vector of each face
    """
    # the header is always 84 bytes long, we just reference the dtype.itemsize
    # to be explicit about where that magical number comes from
    header_length = _stl_dtype_header.itemsize
    header_data = file_obj.read(header_length)
    if len(header_data) < header_length:
        raise HeaderError('Binary STL shorter than a fixed header!')

    try:
        header = np.frombuffer(header_data,
                               dtype=_stl_dtype_header)
    except BaseException:
        raise HeaderError('Binary header incorrect type')

    try:
        # save the header block as a string
        # there could be any garbage in there so wrap in try
        metadata = {'header':
                    bytes(header['header'][0]).decode('utf-8').strip()}
    except BaseException:
        metadata = {}

    # now we check the length from the header versus the length of the file
    # data_start should always be position 84, but hard coding that felt ugly
    data_start = file_obj.tell()
    # this seeks to the end of the file
    # position 0, relative to the end of the file 'whence=2'
    file_obj.seek(0, 2)
    # we save the location of the end of the file and seek back to where we
    # started from
    data_end = file_obj.tell()
    file_obj.seek(data_start)

    # the binary format has a rigidly defined structure, and if the length
    # of the file doesn't match the header, the loaded version is almost
    # certainly going to be garbage.
    len_data = data_end - data_start
    len_expected = header['face_count'] * _stl_dtype.itemsize

    # this check is to see if this really is a binary STL file.
    # if we don't do this and try to load a file that isn't structured properly
    # we will be producing garbage or crashing hard
    # so it's much better to raise an exception here.
    if len_data != len_expected:
        raise HeaderError('Binary STL has incorrect length in header!')
    blob = np.frombuffer(file_obj.read(), dtype=_stl_dtype)

    # all of our vertices will be loaded in order
    # so faces are just sequential indices reshaped.
    faces = np.arange(header['face_count'] * 3).reshape((-1, 3))

    result = {'vertices': blob['vertices'].reshape((-1, 3)),
              'face_normals': blob['normals'].reshape((-1, 3)),
              'faces': faces,
              'metadata': metadata}
    return result


def load_stl_ascii(file_obj):
    """
    Load an ASCII STL file from a file object.

    Parameters
    ----------
    file_obj: open file- like object

    Returns
    ----------
    loaded: kwargs for a Trimesh constructor with keys:
              vertices:     (n,3) float, vertices
              faces:        (m,3) int, indexes of vertices
              face_normals: (m,3) float, normal vector of each face
    """

    # the first line is the header
    header = file_obj.readline()
    # make sure header is a string, not bytes
    if hasattr(header, 'decode'):
        try:
            header = header.decode('utf-8')
        except BaseException:
            header = ''
    # save header to metadata
    metadata = {'header': header}

    # read all text into one string
    text = file_obj.read()
    # convert bytes to string
    if hasattr(text, 'decode'):
        text = text.decode('utf-8')
    # split by endsolid keyword
    text = text.lower().split('endsolid')[0]
    # create array of splits
    blob = np.array(text.strip().split())

    # there are 21 'words' in each face
    face_len = 21

    # length of blob should be multiple of face_len
    if (len(blob) % face_len) != 0:
        raise HeaderError('Incorrect length STL file!')
    face_count = int(len(blob) / face_len)

    # this offset is to be added to a fixed set of tiled indices
    offset = face_len * np.arange(face_count).reshape((-1, 1))
    normal_index = np.tile([2, 3, 4], (face_count, 1)) + offset
    vertex_index = np.tile([8, 9, 10,
                            12, 13, 14,
                            16, 17, 18], (face_count, 1)) + offset

    # faces are groups of three sequential vertices
    faces = np.arange(face_count * 3).reshape((-1, 3))
    face_normals = blob[normal_index].astype('<f8')
    vertices = blob[vertex_index.reshape((-1, 3))].astype('<f8')

    return {'vertices': vertices,
            'faces': faces,
            'metadata': metadata,
            'face_normals': face_normals}


def export_stl(mesh):
    """
    Convert a Trimesh object into a binary STL file.

    Parameters
    ---------
    mesh: Trimesh object

    Returns
    ---------
    export: bytes, representing mesh in binary STL form
    """
    header = np.zeros(1, dtype=_stl_dtype_header)
    header['face_count'] = len(mesh.faces)

    packed = np.zeros(len(mesh.faces), dtype=_stl_dtype)
    packed['normals'] = mesh.face_normals
    packed['vertices'] = mesh.triangles

    export = header.tostring()
    export += packed.tostring()

    return export


def export_stl_ascii(mesh):
    """
    Convert a Trimesh object into an ASCII STL file.

    Parameters
    ---------
    mesh : trimesh.Trimesh

    Returns
    ---------
    export : str
        Mesh represented as an ASCII STL file
    """

    # move all the data that's going into the STL file into one array
    blob = np.zeros((len(mesh.faces), 4, 3))
    blob[:, 0, :] = mesh.face_normals
    blob[:, 1:, :] = mesh.triangles

    # create a lengthy format string for the data section of the file
    format_string = 'facet normal {} {} {}\nouter loop\n'
    format_string += 'vertex {} {} {}\n' * 3
    format_string += 'endloop\nendfacet\n'
    format_string *= len(mesh.faces)

    # concatenate the header, data, and footer
    export = 'solid \n'
    export += format_string.format(*blob.reshape(-1))
    export += 'endsolid'

    return export


_stl_loaders = {'stl': load_stl,
                'stl_ascii': load_stl}
