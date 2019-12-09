import numpy as np

from distutils.spawn import find_executable
from string import Template

import json
import tempfile
import subprocess
import collections

from .. import util
from .. import visual
from .. import grouping
from .. import resources

from ..constants import log

try:
    import PIL.Image
except ImportError:
    pass

# from ply specification, and additional dtypes found in the wild
dtypes = {
    'char': 'i1',
    'uchar': 'u1',
    'short': 'i2',
    'ushort': 'u2',
    'int': 'i4',
    'int8': 'i1',
    'int16': 'i2',
    'int32': 'i4',
    'int64': 'i8',
    'uint': 'u4',
    'uint8': 'u1',
    'uint16': 'u2',
    'uint32': 'u4',
    'uint64': 'u8',
    'float': 'f4',
    'float16': 'f2',
    'float32': 'f4',
    'float64': 'f8',
    'double': 'f8'}


def load_ply(file_obj,
             resolver=None,
             fix_texture=True,
             prefer_color=None,
             *args,
             **kwargs):
    """
    Load a PLY file from an open file object.

    Parameters
    ---------
    file_obj : an open file- like object
      Source data, ASCII or binary PLY
    resolver : trimesh.visual.resolvers.Resolver
      Object which can resolve assets
    fix_texture : bool
      If True, will re- index vertices and faces
      so vertices with different UV coordinates
      are disconnected.
    prefer_color : None, 'vertex', or 'face'
      Which kind of color to prefer if both defined

    Returns
    ---------
    mesh_kwargs : dict
      Data which can be passed to
      Trimesh constructor, eg: a = Trimesh(**mesh_kwargs)
    """

    # OrderedDict which is populated from the header
    elements, is_ascii, image_name = parse_header(file_obj)

    # functions will fill in elements from file_obj
    if is_ascii:
        ply_ascii(elements, file_obj)
    else:
        ply_binary(elements, file_obj)

    # try to load the referenced image
    image = None
    if image_name is not None:
        try:
            data = resolver.get(image_name)
            image = PIL.Image.open(util.wrap_as_stream(data))
        except BaseException:
            log.warning('unable to load image!',
                        exc_info=True)

    kwargs = elements_to_kwargs(elements,
                                fix_texture=fix_texture,
                                image=image,
                                prefer_color=prefer_color)

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
    templates = json.loads(resources.get('ply.template'))
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


def parse_header(file_obj):
    """
    Read the ASCII header of a PLY file, and leave the file object
    at the position of the start of data but past the header.

    Parameters
    -----------
    file_obj : open file object
      Positioned at the start of the file

    Returns
    -----------
    elements : collections.OrderedDict
      Fields and data types populated
    is_ascii : bool
      Whether the data is ASCII or binary
    image_name : None or str
      File name of TextureFile
    """

    if 'ply' not in str(file_obj.readline()):
        raise ValueError('not a ply file!')

    # collect the encoding: binary or ASCII
    encoding = file_obj.readline().decode('utf-8').strip().lower()
    is_ascii = 'ascii' in encoding

    # big or little endian
    endian = ['<', '>'][int('big' in encoding)]
    elements = collections.OrderedDict()

    # store file name of TextureFiles in the header
    image_name = None

    while True:
        line = file_obj.readline()
        if line is None:
            raise ValueError("Header not terminated properly!")
        line = line.decode('utf-8').strip().split()

        # we're done
        if 'end_header' in line:
            break

        # elements are groups of properties
        if 'element' in line[0]:
            # we got a new element so add it
            name, length = line[1:]
            elements[name] = {
                'length': int(length),
                'properties': collections.OrderedDict()}
        # a property is a member of an element
        elif 'property' in line[0]:
            # is the property a simple single value, like:
            # `propert float x`
            if len(line) == 3:
                dtype, field = line[1:]
                elements[name]['properties'][
                    str(field)] = endian + dtypes[dtype]
            # is the property a painful list, like:
            # `property list uchar int vertex_indices`
            elif 'list' in line[1]:
                dtype_count, dtype, field = line[2:]
                elements[name]['properties'][
                    str(field)] = (
                    endian +
                    dtypes[dtype_count] +
                    ', ($LIST,)' +
                    endian +
                    dtypes[dtype])
        # referenced as a file name
        elif 'TextureFile' in line:
            # textures come listed like:
            # `comment TextureFile fuze_uv.jpg`
            index = line.index('TextureFile') + 1
            if index < len(line):
                image_name = line[index]

    return elements, is_ascii, image_name


def elements_to_kwargs(elements,
                       fix_texture,
                       image,
                       prefer_color=None):
    """
    Given an elements data structure, extract the keyword
    arguments that a Trimesh object constructor will expect.

    Parameters
    ------------
    elements : OrderedDict object
      With fields and data loaded
    fix_texture : bool
      If True, will re- index vertices and faces
      so vertices with different UV coordinates
      are disconnected.
    image : PIL.Image
      Image to be viewed
    prefer_color : None, 'vertex', or 'face'
      Which kind of color to prefer if both defined

    Returns
    -----------
    kwargs : dict
      Keyword arguments for Trimesh constructor
    """

    kwargs = {'metadata': {'ply_raw': elements}}

    vertices = np.column_stack([elements['vertex']['data'][i]
                                for i in 'xyz'])

    if not util.is_shape(vertices, (-1, 3)):
        raise ValueError('Vertices were not (n,3)!')

    try:
        face_data = elements['face']['data']
    except (KeyError, ValueError):
        # some PLY files only include vertices
        face_data = None
        faces = None

    # what keys do in-the-wild exporters use for vertices
    index_names = ['vertex_index',
                   'vertex_indices']
    texcoord = None

    if util.is_shape(face_data, (-1, (3, 4))):
        faces = face_data
    elif isinstance(face_data, dict):
        # get vertex indexes
        for i in index_names:
            if i in face_data:
                faces = face_data[i]
                break
        # if faces have UV coordinates defined use them
        if 'texcoord' in face_data:
            texcoord = face_data['texcoord']

    elif isinstance(face_data, np.ndarray):
        face_blob = elements['face']['data']
        # some exporters set this name to 'vertex_index'
        # and some others use 'vertex_indices' but we really
        # don't care about the name unless there are multiple
        if len(face_blob.dtype.names) == 1:
            name = face_blob.dtype.names[0]
        elif len(face_blob.dtype.names) > 1:
            # loop through options
            for i in face_blob.dtype.names:
                if i in index_names:
                    name = i
                    break
        # get faces
        faces = face_blob[name]['f1']

        try:
            texcoord = face_blob['texcoord']['f1']
        except (ValueError, KeyError):
            # accessing numpy arrays with named fields
            # incorrectly is a ValueError
            pass

    if faces is not None:
        # PLY stores texture coordinates per- face which is
        # slightly annoying, as we have to then figure out
        # which vertices have the same position but different UV
        expected = (faces.shape[0], faces.shape[1] * 2)
        if (image is not None and
            texcoord is not None and
                texcoord.shape == expected):

            # vertices with the same position but different
            # UV coordinates can't be merged without it
            # looking like it went through a woodchipper
            # in- the- wild PLY comes with things merged that
            # probably shouldn't be so disconnect vertices
            if fix_texture:
                # do import here
                from ..visual.texture import unmerge_faces

                # reshape to correspond with flattened faces
                uv_all = texcoord.reshape((-1, 2))
                # UV coordinates defined for every triangle have
                # duplicates which can be merged so figure out
                # which UV coordinates are the same here
                unique, inverse = grouping.unique_rows(uv_all)

                # use the indices of faces and face textures
                # to only merge vertices where the position
                # AND uv coordinate are the same
                faces, mask_v, mask_vt = unmerge_faces(
                    faces, inverse.reshape(faces.shape))
                # apply the mask to get resulting vertices
                vertices = vertices[mask_v]
                # apply the mask to get UV coordinates
                uv = uv_all[unique][mask_vt]
            else:
                # don't alter vertices, UV will look like crap
                # if it was exported with vertices merged
                uv = np.zeros((len(vertices), 2))
                uv[faces.reshape(-1)] = texcoord.reshape((-1, 2))

            # create the visuals object for the texture
            kwargs['visual'] = visual.texture.TextureVisuals(
                uv=uv, image=image)
        # faces were not none so assign them
        kwargs['faces'] = faces
    # kwargs for Trimesh or PointCloud
    kwargs['vertices'] = vertices

    # if both vertex and face color are defined pick the one
    # with the most "signal," i.e. which one is not all zeros
    colors = []
    signal = []
    if faces is not None:
        # extract face colors or None
        f_color, f_signal = element_colors(elements['face'])
        colors.append({'face_colors': f_color})
        signal.append(f_signal)
        # extract vertex colors or None
        v_color, v_signal = element_colors(elements['vertex'])
        colors.append({'vertex_colors': v_color})
        signal.append(v_signal)

        if prefer_color is None:
            # if we are in "auto-pick" mode take the one with the
            # largest  standard deviation of colors
            kwargs.update(colors[np.argmax(signal)])
        elif 'vert' in prefer_color and v_color is not None:
            # vertex colors are preferred and defined
            kwargs['vertex_colors'] = v_color
        elif 'face' in prefer_color and f_color is not None:
            # face colors are preferred and defined
            kwargs['face_colors'] = f_color
    else:
        kwargs['colors'] = element_colors(elements['vertex'])

    return kwargs


def element_colors(element):
    """
    Given an element, try to extract RGBA color from
    properties and return them as an (n,3|4) array.

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
        signal = colors.std(axis=0).sum()
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

    # get the file contents as a string
    text = str(file_obj.read().decode('utf-8'))

    # split by newlines
    lines = str.splitlines(text)

    # get each line as an array split by whitespace
    array = np.array([np.fromstring(i, sep=' ')
                      for i in lines])

    # store the line position in the file
    position = 0

    # loop through data we need
    for key, values in elements.items():
        # if the element is empty ignore it
        if 'length' not in values or values['length'] == 0:
            continue
        # will store (start, end) column index of data
        columns = collections.deque()
        # will store the total number of rows
        rows = 0

        for name, dtype in values['properties'].items():
            # we need to know how many elements are in this dtype
            if '$LIST' in dtype:
                # if an element contains a list property handle it here
                row = array[position]
                list_count = int(row[rows])
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
        # get the lines as a 2D numpy array
        data = np.vstack(array[position:position + values['length']])
        # offset position in file
        position += values['length']
        # store columns we care about by name and convert to data type
        elements[key]['data'] = {n: data[:, c[0]:c[1]].astype(dt)
                                 for n, dt, c in zip(
            values['properties'].keys(),    # field name
            values['properties'].values(),  # data type of field
            columns)}                       # list of (start, end) column indexes


def ply_binary(elements, file_obj):
    """
    Load the data from a binary PLY file into the elements data structure.

    Parameters
    ------------
    elements : OrderedDict
      Populated from the file header.
      Object will be modified to add data by this function.

    file_obj : open file object
      With current position at the start
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
        elem_pop = []
        for element_key, element in elements.items():
            props = element['properties']
            prior_data = ''
            for k, dtype in props.items():
                prop_pop = []
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
                    blob = file_obj.read(field_dtype.itemsize)
                    if len(blob) == 0:
                        # no data was read for property
                        prop_pop.append(k)
                        break
                    size = np.frombuffer(blob, dtype=field_dtype)[0]
                    props[k] = props[k].replace('$LIST', str(size))
                prior_data += props[k] + ','
            if len(prop_pop) > 0:
                # if a property was empty remove it
                for pop in prop_pop:
                    props.pop(pop)
                # if we've removed all properties from
                # an element remove the element later
                if len(props) == 0:
                    elem_pop.append(element_key)
                    continue
            # get the size of the items in bytes
            itemsize = np.dtype(', '.join(props.values())).itemsize
            # offset the file based on read size
            p_current += element['length'] * itemsize
        # move the file back to where we found it
        file_obj.seek(p_start)
        # if there were elements without properties remove them
        for pop in elem_pop:
            elements.pop(pop)

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


def export_draco(mesh, bits=28):
    """
    Export a mesh using Google's Draco compressed format.

    Only works if draco_encoder is in your PATH:
    https://github.com/google/draco

    Parameters
    ----------
    mesh : Trimesh object
      Mesh to export
    bits : int
      Bits of quantization for position
      tol.merge=1e-8 is roughly 25 bits

    Returns
    ----------
    data : str or bytes
      DRC file bytes
    """
    with tempfile.NamedTemporaryFile(suffix='.ply') as temp_ply:
        temp_ply.write(export_ply(mesh))
        temp_ply.flush()
        with tempfile.NamedTemporaryFile(suffix='.drc') as encoded:
            subprocess.check_output([draco_encoder,
                                     '-qp',
                                     str(int(bits)),
                                     '-i',
                                     temp_ply.name,
                                     '-o',
                                     encoded.name])
            encoded.seek(0)
            data = encoded.read()
    return data


def load_draco(file_obj, **kwargs):
    """
    Load a mesh from Google's Draco format.

    Parameters
    ----------
    file_obj : file- like object
      Contains data

    Returns
    ----------
    kwargs : dict
      Keyword arguments to construct a Trimesh object
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
