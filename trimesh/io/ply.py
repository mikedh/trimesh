import numpy as np

from distutils.spawn import find_executable
from string import Template

import collections
import subprocess
import tempfile
import json

from .. import util

from ..resources import get_resource

# from ply specification, and additional dtypes found in the wild
ply_dtypes = {'char': 'i1',
              'uchar': 'u1',
              'short': 'i2',
              'ushort': 'u2',
              'int': 'i4',
              'int8': 'i1',
              'int16': 'i2',
              'int32': 'i4',
              'uint': 'u4',
              'uint8': 'u1',
              'uint16': 'u2',
              'uint32': 'u4',
              'float': 'f4',
              'float16': 'f2',
              'float32': 'f4',
              'double': 'f8'}


def load_ply(file_obj, *args, **kwargs):
    """
    Load a PLY file from an open file object.

    Parameters
    ---------
    file_obj : an open file- like object

    Returns
    ---------
    mesh_kwargs : dictionary of mesh info which can be passed to
                  Trimesh constructor, eg: a = Trimesh(**mesh_kwargs)
    """

    # OrderedDict which is populated from the header
    elements, is_ascii = read_ply_header(file_obj)

    # functions will fill in elements from file_obj
    if is_ascii:
        ply_ascii(elements, file_obj)
    else:
        ply_binary(elements, file_obj)

    kwargs = elements_to_kwargs(elements)
    return kwargs


def export_ply(mesh,
               encoding='binary',
               vertex_normal=None):
    """
    Export a mesh in the PLY format.

    Parameters
    ----------
    mesh : Trimesh object
    encoding : ['ascii'|'binary_little_endian']
    vertex_normal : include vertex normals

    Returns
    ----------
    export : bytes of result
    """
    # evaluate input args
    # allow a shortcut for binary
    if encoding == 'binary':
        encoding = 'binary_little_endian'
    elif encoding not in ['binary_little_endian', 'ascii']:
        raise ValueError('encoding must be binary or ascii')
    # if vertex normals aren't specifically asked for
    # only export them if they are stored in cache
    if vertex_normal is None:
        vertex_normal = 'vertex_normal' in mesh._cache

    # custom numpy dtypes for exporting
    dtype_face = [('count', '<u1'),
                  ('index', '<i4', (3))]
    dtype_vertex = [('vertex', '<f4', (3))]
    # will be appended to main dtype if needed
    dtype_vertex_normal = ('normals', '<f4', (3))
    dtype_color = ('rgba', '<u1', (4))

    # get template strings in dict
    templates = json.loads(get_resource('ply.template'))
    # start collecting elements into a string for the header
    header = templates['intro']
    header += templates['vertex']

    # if we're exporting vertex normals add them
    # to the header and dtype
    if vertex_normal:
        header += templates['vertex_normal']
        dtype_vertex.append(dtype_vertex_normal)

    # if mesh has a vertex coloradd it to the header
    if mesh.visual.kind == 'vertex' and encoding != 'ascii':
        header += templates['color']
        dtype_vertex.append(dtype_color)

    # create and populate the custom dtype for vertices
    vertex = np.zeros(len(mesh.vertices),
                      dtype=dtype_vertex)
    vertex['vertex'] = mesh.vertices
    if vertex_normal:
        vertex['normals'] = mesh.vertex_normals
    if mesh.visual.kind == 'vertex':
        vertex['rgba'] = mesh.visual.vertex_colors

    header += templates['face']
    if mesh.visual.kind == 'face' and encoding != 'ascii':
        header += templates['color']
        dtype_face.append(dtype_color)

    # put mesh face data into custom dtype to export
    faces = np.zeros(len(mesh.faces), dtype=dtype_face)
    faces['count'] = 3
    faces['index'] = mesh.faces
    if mesh.visual.kind == 'face' and encoding != 'ascii':
        faces['rgba'] = mesh.visual.face_colors

    header += templates['outro']

    header_params = {'vertex_count': len(mesh.vertices),
                     'face_count': len(mesh.faces),
                     'encoding': encoding}

    export = Template(header).substitute(header_params).encode('utf-8')

    if encoding == 'binary_little_endian':
        export += vertex.tostring()
        export += faces.tostring()
    elif encoding == 'ascii':
        # ply format is: (face count, v0, v1, v2)
        fstack = np.column_stack((np.ones(len(mesh.faces),
                                          dtype=np.int64) * 3,
                                  mesh.faces))

        # if we're exporting vertex normals they get stacked
        if vertex_normal:
            vstack = np.column_stack((mesh.vertices,
                                      mesh.vertex_normals))
        else:
            vstack = mesh.vertices

        # add the string formatted vertices and faces
        export += (util.array_to_string(vstack,
                                        col_delim=' ',
                                        row_delim='\n') +
                   '\n' +
                   util.array_to_string(fstack,
                                        col_delim=' ',
                                        row_delim='\n')).encode('utf-8')
    else:
        raise ValueError('encoding must be ascii or binary!')

    return export


def read_ply_header(file_obj):
    """
    Read the ASCII header of a PLY file, and leave the file object
    at the position of the start of data but past the header.

    Parameters
    -----------
    file_obj: open file object, positioned at the start of the file

    Returns
    -----------
    elements: OrderedDict object, with fields and data types populated
    is_ascii: bool, whether the data is ASCII or binary
    """

    if 'ply' not in str(file_obj.readline()):
        raise ValueError('This aint a ply file')

    encoding = file_obj.readline().decode('utf-8').strip().lower()
    encoding_ascii = 'ascii' in encoding

    endian = ['<', '>'][int('big' in encoding)]
    elements = collections.OrderedDict()

    while True:
        line = file_obj.readline()
        if line is None:
            raise ValueError('Header wasn\'t terminated properly!')
        line = line.decode('utf-8').strip().split()

        if 'end_header' in line:
            break

        if 'element' in line[0]:
            name, length = line[1:]
            elements[name] = {'length': int(length),
                              'properties': collections.OrderedDict()}
        elif 'property' in line[0]:
            if len(line) == 3:
                dtype, field = line[1:]
                elements[name]['properties'][
                    str(field)] = endian + ply_dtypes[dtype]
            elif 'list' in line[1]:
                dtype_count, dtype, field = line[2:]
                elements[name]['properties'][
                    str(field)] = (
                    endian +
                    ply_dtypes[dtype_count] +
                    ', ($LIST,)' +
                    endian +
                    ply_dtypes[dtype])
    return elements, encoding_ascii


def elements_to_kwargs(elements):
    """
    Given an elements data structure, extract the keyword
    arguments that a Trimesh object constructor will expect.

    Parameters
    ------------
    elements: OrderedDict object, with fields and data loaded

    Returns
    -----------
    kwargs: dict, with keys for Trimesh constructor.
            eg: mesh = trimesh.Trimesh(**kwargs)
    """
    vertices = np.column_stack([elements['vertex']['data'][i] for i in 'xyz'])
    if not util.is_shape(vertices, (-1, 3)):
        raise ValueError('Vertices were not (n,3)!')

    index_names = ['vertex_index',
                   'vertex_indices']
    face_data = elements['face']['data']
    if util.is_shape(face_data, (-1, (3, 4))):
        faces = face_data
    elif isinstance(face_data, dict):
        for i in index_names:
            if i in face_data:
                faces = face_data[i]
                break
    elif isinstance(face_data, np.ndarray):
        blob = elements['face']['data']
        # some exporters set this name to 'vertex_index'
        # and some others use 'vertex_indices', but we really
        # don't care about the name unless there are multiple properties
        if len(blob.dtype.names) == 1:
            name = blob.dtype.names[0]
        elif len(blob.dtype.names) > 1:
            for i in blob.dtype.names:
                if i in index_names:
                    name = i
                    break
        faces = elements['face']['data'][name]['f1']
    else:
        raise ValueError('Couldn\'t extract face data!')

    if not util.is_shape(faces, (-1, (3, 4))):
        raise ValueError('Faces weren\'t (n,(3|4))!')

    result = {'vertices': vertices,
              'faces': faces,
              'ply_data': elements}

    # if both vertex and face color are defined, pick the one
    # with the most going on
    f_color, f_signal = element_colors(elements['face'])
    v_color, v_signal = element_colors(elements['vertex'])
    colors = [{'face_colors': f_color},
              {'vertex_colors': v_color}]
    colors_index = np.argmax([f_signal,
                              v_signal])
    result.update(colors[colors_index])

    return result


def element_colors(element):
    """
    Given an element, try to extract RGBA color from its properties
    and return them as an (n,3|4) array.

    Parameters
    -------------
    element: dict, containing color keys

    Returns
    ------------
    colors: (n,(3|4)
    signal: float, estimate of range
    """
    keys = ['red', 'green', 'blue', 'alpha']
    candidate_colors = [element['data'][i]
                        for i in keys if i in element['properties']]

    if len(candidate_colors) >= 3:
        colors = np.column_stack(candidate_colors)
        signal = colors.ptp(axis=0).sum()
        return colors, signal

    return None, 0.0


def ply_ascii(elements, file_obj):
    """
    Load data from an ASCII PLY file into an existing elements data structure.

    Parameters
    ------------
    elements: OrderedDict object, populated from the file header.
              object will be modified to add data by this function.

    file_obj: open file object, with current position at the start
              of the data section (past the header)
    """

    # list of strings, split by newlines and spaces
    blob = file_obj.read().decode('utf-8')
    # numpy array with string type
    raw = np.array(blob.split())
    position = 0

    for key, values in elements.items():
        # will store (start, end) column index of data
        columns = collections.deque()
        # will store the total number of rows
        rows = 0

        for name, dtype in values['properties'].items():
            if '$LIST' in dtype:
                # if an element contains a list property handle it here

                # the first value is a count, followed by data
                list_count = int(raw[position + rows])

                # ignore the count and take the data
                columns.append([rows + 1,
                                rows + 1 + list_count])
                rows += list_count + 1
                # change the datatype to just the dtype for data
                values['properties'][name] = dtype.split('($LIST,)')[-1]
            else:
                # a single column data field
                columns.append([rows, rows + 1])
                rows += 1

        # total flat count of values
        count = values['length'] * rows
        # reshape the data into the specified rows
        data = raw[position:position + count].reshape((-1, rows))

        # store columns we care about by name and convert to specified data
        # type
        elements[key]['data'] = {n: data[:, c[0]:c[1]].astype(dt) for n, dt, c in zip(
            values['properties'].keys(),    # field name
            values['properties'].values(),  # data type of field
            columns)}                       # list of (start, end) column indexes

        # move up our position in the file based on how many
        # values we just read
        position += count

    if position != len(raw):
        raise ValueError('File was unexpected length!')


def ply_binary(elements, file_obj):
    """
    Load the data from a binary PLY file into the elements data structure.

    Parameters
    ------------
    elements: OrderedDict object, populated from the file header.
              object will be modified to add data by this function.

    file_obj: open file object, with current position at the start
              of the data section (past the header)
    """

    def populate_listsize(file_obj, elements):
        """
        Given a set of elements populated from the header if there are any
        list properties seek in the file the length of the list.

        Note that if you have a list where each instance is different length
        (if for example you mixed triangles and quads) this won't work at all
        """
        p_start = file_obj.tell()
        p_current = file_obj.tell()
        for element_key, element in elements.items():
            props = element['properties']
            prior_data = ''
            for k, dtype in props.items():
                if '$LIST' in dtype:
                    # every list field has two data types:
                    # the list length (single value), and the list data (multiple)
                    # here we are only reading the single value for list length
                    field_dtype = np.dtype(dtype.split(',')[0])
                    if len(prior_data) == 0:
                        offset = 0
                    else:
                        offset = np.dtype(prior_data).itemsize
                    file_obj.seek(p_current + offset)
                    size = np.frombuffer(file_obj.read(field_dtype.itemsize),
                                         dtype=field_dtype)[0]
                    props[k] = props[k].replace('$LIST', str(size))
                prior_data += props[k] + ','
            itemsize = np.dtype(', '.join(props.values())).itemsize
            p_current += element['length'] * itemsize
        file_obj.seek(p_start)

    def populate_data(file_obj, elements):
        """
        Given the data type and field information from the header,
        read the data and add it to a 'data' field in the element.
        """
        for key in elements.keys():
            items = list(elements[key]['properties'].items())
            dtype = np.dtype(items)
            data = file_obj.read(elements[key]['length'] * dtype.itemsize)
            elements[key]['data'] = np.frombuffer(data,
                                                  dtype=dtype)
        return elements

    def elements_size(elements):
        """
        Given an elements data structure populated from the header,
        calculate how long the file should be if it is intact.
        """
        size = 0
        for element in elements.values():
            dtype = np.dtype(','.join(element['properties'].values()))
            size += element['length'] * dtype.itemsize
        return size

    # some elements are passed where the list dimensions
    # are not included in the header, so this function goes
    # into the meat of the file and grabs the list dimensions
    # before we to the main data read as a single operation
    populate_listsize(file_obj, elements)

    # how many bytes are left in the file
    size_file = util.distance_to_end(file_obj)
    # how many bytes should the data structure described by
    # the header take up
    size_elements = elements_size(elements)

    # if the number of bytes is not the same the file is probably corrupt
    if size_file != size_elements:
        raise ValueError('File is unexpected length!')

    # with everything populated and a reasonable confidence the file
    # is intact, read the data fields described by the header
    populate_data(file_obj, elements)


def export_draco(mesh):
    """
    Export a mesh using Google's Draco compressed format.

    Only works if draco_encoder is in your PATH:
    https://github.com/google/draco

    Parameters
    ----------
    mesh : Trimesh object

    Returns
    ----------
    data : str or bytes, data
    """
    with tempfile.NamedTemporaryFile(suffix='.ply') as temp_ply:
        temp_ply.write(export_ply(mesh))
        temp_ply.flush()
        with tempfile.NamedTemporaryFile(suffix='.drc') as encoded:
            subprocess.check_output([draco_encoder,
                                     '-qp',  # bits of quantization for position
                                     '28',  # since our tol.merge is 1e-8, 25 bits
                                            # more has a machine epsilon
                                            # smaller than that
                                     '-i',
                                     temp_ply.name,
                                     '-o',
                                     encoded.name])
            encoded.seek(0)
            data = encoded.read()
    return data


def load_draco(file_obj, file_type=None):
    """
    Load a mesh from Google's Draco format.

    Parameters
    ----------
    file_obj  : open file- like object
    file_type : unused

    Returns
    ----------
    kwargs : dict, kwargs to construct a Trimesh object
    """

    with tempfile.NamedTemporaryFile(suffix='.drc') as temp_drc:
        temp_drc.write(file_obj.read())
        temp_drc.flush()

        with tempfile.NamedTemporaryFile(suffix='.ply') as temp_ply:
            subprocess.check_output([draco_decoder,
                                     '-i',
                                     temp_drc.name,
                                     '-o',
                                     temp_ply.name])
            temp_ply.seek(0)
            kwargs = load_ply(temp_ply)
    return kwargs


_ply_loaders = {'ply': load_ply}
_ply_exporters = {'ply': export_ply}

draco_encoder = find_executable('draco_encoder')
draco_decoder = find_executable('draco_decoder')

if draco_decoder is not None:
    _ply_loaders['drc'] = load_draco
if draco_encoder is not None:
    _ply_exporters['drc'] = export_draco
