import numpy as np
import collections


from .. import util
from .. import visual
from .. import transformations


def xaml_load(file_obj, *args, **kwargs):
    '''
    Load a 3D XAML file.

    Arguments
    ----------
    file_obj: open file object

    Returns
    ----------
    result: dict, with keys:
            vertices:       (n,3) np.float64, points in space
            faces:          (m,3) np.int64, indices of vertices
            face_colors:    (m,4) np.uint8, RGBA colors
            vertex_normals: (n,3) np.float64, vertex normals
    '''
    def element_to_color(element):
        '''
        Turn an XML element into a (4,) np.uint8 RGBA color
        '''
        if element is None:
            return visual.DEFAULT_COLOR
        hexcolor = int(element.attrib['Color'].replace('#', ''), 16)
        opacity = float(element.attrib['Opacity'])
        rgba = [(hexcolor >> 16) & 0xFF,
                (hexcolor >> 8) & 0xFF,
                (hexcolor & 0xFF),
                opacity * 0xFF]
        rgba = np.array(rgba, dtype=np.uint8)
        return rgba

    def element_to_transform(element):
        '''
        Turn an XML element into a (4,4) np.float64 transformation matrix.
        '''
        try:
            matrix = next(element.iter(
                tag=ns + 'MatrixTransform3D')).attrib['Matrix']
            matrix = np.array(
                matrix.split(), dtype=np.float64).reshape((4, 4)).T
            return matrix
        except StopIteration:
            # this will be raised if the MatrixTransform3D isn't in the passed
            # elements tree
            return np.eye(4)

    # read the file and parse XML
    file_data = file_obj.read()
    root = etree.XML(file_data)

    # the XML namespace
    ns = root.tag.split('}')[0] + '}'

    # the linked lists our results are going in
    vertices = collections.deque()
    faces = collections.deque()
    colors = collections.deque()
    normals = collections.deque()

    # iterate through the element tree
    # the GeometryModel3D tag contains a material and geometry
    for geometry in root.iter(tag=ns + 'GeometryModel3D'):
        
        # get the diffuse and specular colors specified in the material
        color_search = './/{ns}{color}Material/*/{ns}SolidColorBrush'
        diffuse = geometry.find(color_search.format(ns=ns,
                                                    color='Diffuse'))
        specular = geometry.find(color_search.format(ns=ns,
                                                     color='Specular'))

        # convert the element into a (4,) np.uint8 RGBA color
        diffuse = element_to_color(diffuse)
        specular = element_to_color(specular)

        # to get the final transform of a component we'll have to traverse
        # all the way back to the root node and save transforms we find
        current = geometry
        # start with an identity matrix so we can skip checking for empty later
        transforms = collections.deque([np.eye(4)])
        # when the root node is reached its parent will be None and we stop
        while current is not None:
            transform_element = current.find(ns + 'ModelVisual3D.Transform')
            if transform_element is not None:
                # we are traversing the tree backwards, so append new
                # transforms left
                transforms.appendleft(element_to_transform(transform_element))
            current = current.getparent()

        # to get the final transform take the dot product of all matrices
        transform = np.linalg.multi_dot(transforms)

        # iterate through the contained mesh geometry elements
        for g in geometry.iter(tag=ns + 'MeshGeometry3D'):
            c_normals = np.array(g.attrib['Normals'].replace(',', ' ').split(),
                                 dtype=np.float64).reshape((-1, 3))

            c_vertices = np.array(g.attrib['Positions'].replace(',', ' ').split(),
                                  dtype=np.float64).reshape((-1, 3))
            # bake in the transform as we're saving
            c_vertices = transformations.transform_points(c_vertices,
                                                          transform)

            c_faces = np.array(g.attrib['TriangleIndices'].replace(',', ' ').split(),
                               dtype=np.int64).reshape((-1, 3))

            # save data to a sequence
            vertices.append(c_vertices)
            faces.append(c_faces)
            colors.append(np.tile(diffuse, (len(c_faces), 1)))
            normals.append(c_normals)

    # compile the results into clean numpy arrays
    result = dict()
    result['vertices'], result['faces'] = util.append_faces(vertices,
                                                            faces)
    result['face_colors'] = np.vstack(colors)
    result['vertex_normals'] = np.vstack(normals)

    return result


try:
    from lxml import etree
    _xml_loaders = {'xaml': xaml_load}
except ImportError:
    _xml_loaders = {}
