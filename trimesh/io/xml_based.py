import numpy as np
import networkx as nx

import collections
import json

from .. import util
from .. import visual
from .. import transformations


def load_XAML(file_obj, *args, **kwargs):
    '''
    Load a 3D XAML file.

    Parameters
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
            matrix = np.array(matrix.split(),
                              dtype=np.float64).reshape((4, 4)).T
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
        transforms = collections.deque()
        # when the root node is reached its parent will be None and we stop
        while current is not None:
            # element.find will only return elements that are direct children
            # of the current element as opposed to element.iter,
            # which will return any depth of child
            transform_element = current.find(ns + 'ModelVisual3D.Transform')
            if transform_element is not None:
                # we are traversing the tree backwards, so append new
                # transforms to the left of the deque
                transforms.appendleft(element_to_transform(transform_element))
            # we are going from the lowest level of the tree to the highest
            # this avoids having to traverse any branches that don't have
            # geometry
            current = current.getparent()

        if len(transforms) == 0:
            # no transforms in the tree mean an identity matrix
            transform = np.eye(4)
        elif len(transforms) == 1:
            # one transform in the tree we can just use
            transform = transforms.pop()
        else:
            # multiple transforms we apply all of them in order
            transform = util.multi_dot(transforms)

        # iterate through the contained mesh geometry elements
        for g in geometry.iter(tag=ns + 'MeshGeometry3D'):
            c_normals = np.array(g.attrib['Normals'].replace(',', ' ').split(),
                                 dtype=np.float64).reshape((-1, 3))

            c_vertices = np.array(
                g.attrib['Positions'].replace(
                    ',', ' ').split(), dtype=np.float64).reshape(
                (-1, 3))
            # bake in the transform as we're saving
            c_vertices = transformations.transform_points(c_vertices,
                                                          transform)

            c_faces = np.array(
                g.attrib['TriangleIndices'].replace(
                    ',', ' ').split(), dtype=np.int64).reshape(
                (-1, 3))

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


def load_3DXML(file_obj, *args, **kwargs):
    '''
    Load a 3DXML scene into kwargs.

    Parameters
    -----------
    file_obj: open file object holding 3DXML file

    Returns
    -----------
    geometries: list of dict, kwargs for Trimesh constructor
    graph:      list of dict, kwargs for Scene.graph.update
    '''
    archive = util.decompress(file_obj, file_type='zip')

    # a dictionary of file name : lxml etree
    as_etree = {}
    for k, v in archive.items():
        # wrap in try statement, as sometimes 3DXML
        # contains non- xml files, like JPG previews
        try:
            as_etree[k] = etree.XML(v.read())
        except etree.XMLSyntaxError:
            # move the file object back to the file start
            v.seek(0)

    # the file name of the root scene
    root_file = as_etree['Manifest.xml'].find('{*}Root').text
    # the etree of the scene layout
    tree = as_etree[root_file]
    # index of root element of directed acyclic graph
    root_id = tree.find('{*}ProductStructure').attrib['root']

    # load the materials libary from the materials elements
    colors = {}
    material_tree = as_etree['CATMaterialRef.3dxml']
    for MaterialDomain in material_tree.iter('{*}MaterialDomain'):
        material_id = MaterialDomain.attrib['id']
        material_file = MaterialDomain.attrib['associatedFile'].split(
            'urn:3DXML:')[-1]

        rend = as_etree[material_file].find(
            "{*}Feature[@Alias='RenderingFeature']")
        diffuse = rend.find("{*}Attr[@Name='DiffuseColor']")
        #specular = rend.find("{*}Attr[@Name='SpecularColor']")
        #emissive = rend.find("{*}Attr[@Name='EmissiveColor']")
        rgb = (
            np.array(
                json.loads(
                    diffuse.attrib['Value'])) *
            255).astype(
                np.uint8)
        colors[material_id] = rgb

    # copy indexes for instances of colors
    for MaterialDomainInstance in material_tree.iter(
            '{*}MaterialDomainInstance'):
        instance = MaterialDomainInstance.find('{*}IsInstanceOf')
        #colors[b.attrib['id']] = colors[instance.text]
        for aggregate in MaterialDomainInstance.findall('{*}IsAggregatedBy'):
            colors[aggregate.text] = colors[instance.text]

    # references which hold the 3DXML scene structure as a dict
    # element id : {key : value}
    references = collections.defaultdict(dict)

    # the 3DXML can specify different visual properties for  occurences
    view = tree.find('{*}DefaultView')
    for ViewProp in view.iter('{*}DefaultViewProperty'):
        color = ViewProp.find('{*}GraphicProperties/' +
                              '{*}SurfaceAttributes/{*}Color')
        if (color is None or
                'RGBAColorType' not in color.attrib.values()):
            continue
        rgba = np.array([color.attrib[i] for i in ['red',
                                                   'green',
                                                   'blue',
                                                   'alpha']], dtype=np.float)
        rgba = (rgba * 255).astype(np.uint8)
        for occurence in ViewProp.findall('{*}OccurenceId/{*}id'):
            reference_id = occurence.text.split('#')[-1]
            references[reference_id]['color'] = rgba

    # geometries will hold meshes
    geometries = dict()

    # get geometry
    for ReferenceRep in tree.iter(tag='{*}ReferenceRep'):
        # the str of an int that represents this meshes unique ID
        part_id = ReferenceRep.attrib['id']
        # which part file in the archive contains the geometry we care about
        part_file = ReferenceRep.attrib['associatedFile'].split(':')[-1]

        # prepare to collect actual geometry data
        mesh_faces = []
        mesh_vertices = []
        mesh_colors = []
        mesh_normals = []

        # the geometry is stored in a Rep
        for Rep in as_etree[part_file].iter('{*}Rep'):
            faces = Rep.find('{*}Faces/{*}Face')
            vertices = Rep.find('{*}VertexBuffer/{*}Positions')

            if faces is None or vertices is None:
                continue

            # these are vertex normals
            normals = Rep.find('{*}VertexBuffer/{*}Normals')
            material = Rep.find('{*}SurfaceAttributes/' +
                                '{*}MaterialApplication/' +
                                '{*}MaterialId')

            (material_file, material_id) = material.attrib['id'].split(
                'urn:3DXML:')[-1].split('#')

            # triangle strips, sequence of arbitrary length lists
            # np.fromstring is substantially faster than np.array(i.split())
            # inside the list comprehension
            strips = [np.fromstring(i, sep=' ', dtype=np.int64)
                      for i in faces.attrib['strips'].split(',')]

            # convert strips to (m,3) int
            mesh_faces.append(util.triangle_strips_to_faces(strips))

            # they mix delimiters like we couldn't figure it out from the shape :(
            # load vertices into (n, 3) float64
            mesh_vertices.append(np.fromstring(vertices.text.replace(',',
                                                                     ' '),
                                               sep=' ',
                                               dtype=np.float64).reshape((-1, 3)))
            # load vertex normals into (n, 3) float64
            mesh_normals.append(np.fromstring(normals.text.replace(',',
                                                                   ' '),
                                              sep=' ',
                                              dtype=np.float64).reshape((-1, 3)))

            # store the material information as (m,3) uint8 FACE COLORS
            mesh_colors.append(np.tile(colors[material_id],
                                       (len(mesh_faces[-1]), 1)))

        # save each mesh as the kwargs for a trimesh.Trimesh constructor
        # aka, a Trimesh object can be created with trimesh.Trimesh(**mesh)
        # this avoids needing trimesh- specific imports in this IO function
        mesh = dict()
        (mesh['vertices'],
         mesh['faces']) = util.append_faces(mesh_vertices,
                                            mesh_faces)
        mesh['vertex_normals'] = np.vstack(mesh_normals)
        mesh['face_colors'] = np.vstack(mesh_colors)

        # as far as I can tell, all 3DXML files are exported as
        # implicit milimeters (it isn't specified in the file)
        mesh['metadata'] = {'units': 'mm'}
        mesh['class'] = 'Trimesh'

        geometries[part_id] = mesh
        references[part_id]['geometry'] = part_id

    # a Reference3D maps to a subassembly or assembly
    for Reference3D in tree.iter('{*}Reference3D'):
        references[
            Reference3D.attrib['id']] = {
            'name': Reference3D.attrib['name'],
            'type': 'Reference3D'}

    # a node that is the connectivity between a geometry and the Reference3D
    for InstanceRep in tree.iter('{*}InstanceRep'):
        current = InstanceRep.attrib['id']
        instance = InstanceRep.find('{*}IsInstanceOf').text
        aggregate = InstanceRep.find('{*}IsAggregatedBy').text

        references[current].update({'aggregate': aggregate,
                                    'instance': instance,
                                    'type': 'InstanceRep'})

    # an Instance3D maps basically to a part
    for Instance3D in tree.iter('{*}Instance3D'):
        matrix = np.eye(4)
        relative = Instance3D.find('{*}RelativeMatrix')
        if relative is not None:
            relative = np.array(relative.text.split(),
                                dtype=np.float64)

            # rotation component
            matrix[:3, :3] = relative[:9].reshape((3, 3)).T
            # translation component
            matrix[:3, 3] = relative[9:]

        current = Instance3D.attrib['id']
        name = Instance3D.attrib['name']
        instance = Instance3D.find('{*}IsInstanceOf').text
        aggregate = Instance3D.find('{*}IsAggregatedBy').text

        references[current].update({'aggregate': aggregate,
                                    'instance': instance,
                                    'matrix': matrix,
                                    'name': name,
                                    'type': 'Instance3D'})

    # turn references into directed graph for path finding
    graph = nx.DiGraph()
    for k, v in references.items():
        # IsAggregatedBy points up to a parent
        if 'aggregate' in v:
            graph.add_edge(v['aggregate'], k)
        # IsInstanceOf indicates a child
        if 'instance' in v:
            graph.add_edge(k, v['instance'])

    # the 3DXML format is stored as a directed acyclic graph that needs all
    # paths from the root to a geometry to generate the tree of the scene
    paths = collections.deque()
    for geometry_id in geometries.keys():
        paths.extend(nx.all_simple_paths(graph,
                                         source=root_id,
                                         target=geometry_id))

    # the name of the root frame
    root_name = references[root_id]['name']
    # create a list of kwargs to send to the scene.graph.update function
    # start with a transform from the graphs base frame to our root name
    graph_kwargs = collections.deque([{'frame_to': root_name,
                                       'matrix': np.eye(4)}])

    # we are going to collect prettier geometry names as we traverse paths
    geom_names = {}
    # loop through every simple path and generate transforms tree
    # note that we are flattening the transform tree here
    for path_index, path in enumerate(paths):
        name = ''
        if 'name' in references[path[-3]]:
            name = references[path[-3]]['name']
            geom_names[path[-1]] = name
        # we need a unique node name for our geometry instance frame
        # due to the nature of the DAG names specified by the file may not
        # be unique, so we add an Instance3D name then append the path ids
        node_name = name + '#' + ':'.join(path)

        # pull all transformations in the path
        matrices = [references[i]['matrix']
                    for i in path if 'matrix' in references[i]]
        if len(matrices) == 0:
            matrix = np.eye(4)
        elif len(matrices) == 1:
            matrix = matrices[0]
        else:
            matrix = util.multi_dot(matrices)

        graph_kwargs.append({'matrix': matrix,
                             'frame_from': root_name,
                             'frame_to': node_name,
                             'geometry': path[-1]})

    # remap geometry names from id numbers to the name string
    # we extracted from the 3DXML tree
    geom_final = {}
    for key, value in geometries.items():
        if key in geom_names:
            geom_final[geom_names[key]] = value
    # change geometry names in graph kwargs in place
    for kwarg in graph_kwargs:
        if 'geometry' not in kwarg:
            continue
        kwarg['geometry'] = geom_names[kwarg['geometry']]

    # create the kwargs for load_kwargs
    result = {'class': 'Scene',
              'geometry': geom_final,
              'graph': graph_kwargs}

    return result


def print_element(element):
    '''
    Pretty- print an lxml.etree element
    '''
    print(etree.tostring(element, pretty_print=True).decode('utf-8'))


try:
    from lxml import etree
    _xml_loaders = {'xaml': load_XAML,
                    '3dxml': load_3DXML}

except ImportError:
    _xml_loaders = {}
