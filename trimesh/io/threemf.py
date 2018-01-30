import numpy as np
import networkx as nx

import collections

from .. import util
from .. import graph
from .. import grouping


def load_3MF(file_obj,
             **kwargs):
    """
    Load a 3MF formatted file into a Trimesh scene.

    Parameters
    ------------
    file_obj:       file object

    Returns
    ------------
    kwargs: dict, with keys 'graph', 'geometry', 'base_frame'
    """
    # dict, {name in archive: BytesIo}
    archive = util.decompress(file_obj, file_type='zip')
    # load the XML into an LXML tree
    tree = etree.XML(archive['3D/3dmodel.model'].read())

    # { mesh id : mesh name}
    id_name = {}
    # { mesh id: (n,3) float vertices}
    v_seq = {}
    # { mesh id: (n,3) int faces}
    f_seq = {}
    # components are objects that contain other objects
    # {id : [other ids]}
    components = collections.defaultdict(list)

    for obj in tree.iter('{*}object'):
        # not required, so use a get call which will return None
        # if the tag isn't populated
        name = str(obj.get('name'))
        # id is mandatory
        index = obj.attrib['id']
        # store the name by index
        id_name[index] = name

        # if the object has actual geometry data, store it
        for mesh in obj.iter('{*}mesh'):
            vertices = mesh.find('{*}vertices')
            vertices = np.array([[i.attrib['x'],
                                  i.attrib['y'],
                                  i.attrib['z']] for
                                 i in vertices.iter('{*}vertex')],
                                dtype=np.float64)
            v_seq[index] = vertices

            faces = mesh.find('{*}triangles')
            faces = np.array([[i.attrib['v1'],
                               i.attrib['v2'],
                               i.attrib['v3']] for
                              i in faces.iter('{*}triangle')],
                             dtype=np.int64)
            f_seq[index] = faces

        # components are references to other geometries
        for c in obj.iter('{*}component'):
            mesh_index = c.attrib['objectid']
            transform = _attrib_to_transform(c.attrib)
            components[index].append((mesh_index, transform))

    # load information about the scene graph
    # each instance is a single geometry
    build_items = []
    # scene graph information stored here, aka "build" the scene
    build = tree.find('{*}build')
    for item in build.iter('{*}item'):
        # get a transform from the item's attributes
        transform = _attrib_to_transform(item.attrib)
        # the index of the geometry this item instantiates
        build_items.append((item.attrib['objectid'], transform))

    metadata = {}
    if 'unit' in tree.attrib:
        metadata['units'] = tree.attrib['unit']
    else:
        # the default units, defined by the specification
        metadata['units'] = 'millimeters'

    # have one mesh per 3MF object
    # one mesh per geometry ID
    meshes = {}
    for gid in v_seq.keys():
        name = id_name[gid]
        meshes[name] = {'vertices': v_seq[gid],
                        'faces': f_seq[gid]}

    # turn the item / component representation into
    # a MultiDiGraph to compound our pain
    g = nx.MultiDiGraph()
    for gid, tf in build_items:
        g.add_edge('world', gid, matrix=tf)
    for start, group in components.items():
        for i, (gid, tf) in enumerate(group):
            g.add_edge(start, gid, matrix=tf)

    # turn the graph into kwargs for a scene graph
    # flatten the scene structure and simplify to
    # a single unique node per instance
    graph_args = []
    for path in graph.multigraph_paths(g, source='world'):
        transforms = graph.multigraph_collect(g,
                                              traversal=path,
                                              attrib='matrix')
        if len(transforms) == 1:
            transform = transforms[0]
        else:
            transform = util.multi_dot(transforms)
        name = id_name[path[-1][0]] + util.unique_id()
        geom = id_name[path[-1][0]]
        graph_args.append({'frame_from': 'world',
                           'frame_to': name,
                           'matrix': transform,
                           'geometry': geom})

    # construct the kwargs to load the scene
    kwargs = {'base_frame': 'world',
              'graph': graph_args,
              'geometry': meshes,
              'metadata': metadata}

    return kwargs


def _attrib_to_transform(attrib):
    """
    Extract a homogenous transform from a dictionary.

    Parameters
    ------------
    attrib: dict, optionally containing 'transform'

    Returns
    ------------
    transform: (4, 4) float, homogeonous transformation
    """

    transform = np.eye(4, dtype=np.float64)
    if 'transform' in attrib:
        # wangle their transform format
        values = np.array(
            attrib['transform'].split(),
            dtype=np.float64).reshape((4, 3)).T
        transform[:3, :4] = values
    return transform


# do import here to keep lxml a soft dependancy
try:
    from lxml import etree
    _three_loaders = {'3mf': load_3MF}
except ImportError:
    _three_loaders = {}
