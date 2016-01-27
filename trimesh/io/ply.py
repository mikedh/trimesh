import numpy as np
from collections import OrderedDict

def load_ply(file_obj, *args, **kwargs):
    '''
    Load a PLY file from an open file object.
    
    Arguments
    ---------
    file_obj: an open file- like object

    Returns
    ---------
    mesh_kwargs: dictionary of mesh info which can be passed to 
                 Trimesh constructor, eg: a = Trimesh(**mesh_kwargs)
    '''
    # OrderedDict which is populated from the header
    elements = ply_read_header(file_obj)
    # some elements are passed where the list dimensions
    # are not included in the header, so this function goes 
    # into the meat of the file and grabs the list dimensions 
    # before we to the main data read as a single operation
    ply_populate_listsize(file_obj, elements)

    # how many bytes are left in the file
    size_file = size_to_end(file_obj)
    # how many bytes should the data structure described by
    # the header take up
    size_elements = ply_elements_size(elements)
    
    # if the number of bytes is not the same the file is probably corrupt
    if size_file != size_elements:
        raise ValueError('File is unexpected length!')

    # with everything populated and a reasonable confidence the file
    # is intact, read the data fields described by the header
    ply_populate_data(file_obj, elements)
    # all of the data is now stored in elements, but we need it as
    # a set of keyword arguments we can pass to the Trimesh constructor
    # will look something like {'vertices' : (data), 'faces' : (data)} 
    mesh_kwargs = ply_elements_kwargs(elements)
    return mesh_kwargs

def ply_element_colors(element):
    '''
    Given an element, try to extract RGBA color from its properties
    and return them as an (n,3|4) array.
    '''
    color_keys = ['red', 'green', 'blue', 'alpha']
    candidate_colors = [element['data'][i] for i in color_keys if i in element['properties']]

    if len(candidate_colors) >= 3:
        return np.column_stack(candidate_colors)
    return None

def ply_read_header(file_obj):
    '''
    Read the ASCII header of a PLY file, and leave the file object 
    at the position of the start of data. 
    '''
    # from ply specification
    dtypes = {'char'  : 'i1',
              'uchar' : 'u1',
              'short' : 'i2',
              'ushort': 'u2',
              'int'   : 'i4',
              'uint'  : 'u4',
              'float' : 'f4',
              'double': 'f8'}

    if not 'ply' in str(file_obj.readline()):
        raise ValueError('This aint a ply file')
    encoding = str(file_obj.readline()).strip().split()[1]

    if 'ascii' in encoding:
        raise ValueError('ASCII PLY not supported!')

    endian   = ['<', '>']['big' in encoding]
    elements = OrderedDict()

    while True:
        line = file_obj.readline()
        if line is None:
            raise ValueError('Header wasn\'t terminated properly!')
        line = line.decode('utf-8').strip().split()
        
        if 'end_header' in line:
            break

        if 'element' in line[0]:
            name, length = line[1:]
            elements[name] = {'length'     : int(length), 
                              'properties' : OrderedDict()}
        elif 'property' in line[0]:
            if len(line) == 3:
                dtype, field = line[1:]
                elements[name]['properties'][str(field)] = endian + dtypes[dtype]
            elif 'list' in line[1]:
                dtype_count, dtype, field = line[2:]
                elements[name]['properties'][str(field)] = (endian +
                                                            dtypes[dtype_count] + 
                                                            ', ($LIST,)'+
                                                            endian + 
                                                            dtypes[dtype])
    return elements

def ply_populate_listsize(file_obj, elements):
    '''
    Given a set of elements populated from the header if there are any
    list properties seek in the file the length of the list. 

    Note that if you have a list where each instance is different length
    (if for example you mixed triangles and quads) this won't work at all
    '''
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
                    offset = np.dtype(prior_data).itemsize()
                file_obj.seek(p_current+offset)
                size = np.fromstring(file_obj.read(field_dtype.itemsize), 
                                     dtype=field_dtype)[0]
                props[k] = props[k].replace('$LIST', str(size))
            prior_data += props[k] +','
        itemsize = np.dtype(', '.join(props.values())).itemsize
        p_current += element['length'] * itemsize
    file_obj.seek(p_start)
    
def ply_populate_data(file_obj, elements):
    '''
    Given the data type and field information from the header,
    read the data and add it to a 'data' field in the element.
    '''
    for key in elements.keys():
        items = list(elements[key]['properties'].items())
        dtype = np.dtype(items)
        data  = file_obj.read(elements[key]['length'] * dtype.itemsize)
        elements[key]['data'] = np.fromstring(data, dtype=dtype)
    return elements

def ply_elements_kwargs(elements):
    '''
    Given an elements data structure, extract the keyword
    arguments that a Trimesh object constructor will expect.
    '''
    vertices = np.column_stack([elements['vertex']['data'][i] for i in 'xyz'])
    faces    = elements['face']['data']['vertex_indices']['f1']
    face_colors   = ply_element_colors(elements['face'])
    vertex_colors = ply_element_colors(elements['vertex'])
    result = {'vertices' : vertices,
              'faces'    : faces,
              'face_colors'   : face_colors,
              'vertex_colors' : vertex_colors}
    return result

def ply_elements_size(elements):
    '''
    Given an elements data structure populated from the header, 
    calculate how long the file should be if it is intact.
    '''
    size = 0
    for element in elements.values():
        dtype = np.dtype(','.join(element['properties'].values()))
        size += element['length'] * dtype.itemsize
    return size

def size_to_end(file_obj):
    '''
    Given an open file object, return the number of bytes 
    to the end of the file
    '''
    position_current = file_obj.tell()
    file_obj.seek(0,2)
    position_end = file_obj.tell()
    file_obj.seek(position_current)
    size = position_end - position_current
    return size

_ply_loaders = {'ply' : load_ply}
