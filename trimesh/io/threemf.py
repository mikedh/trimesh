import numpy as np

from .. import util
from .. import grouping


def load_3MF(file_obj,
             combine_bodies=True,
             **kwargs):
    """
    Load a 3MF formatted file into a Trimesh scene.
    
    3MF files can only have one body per mesh, so multibody parts
    are exported as multiple meshes.

    This looks quite dumb on things like ball bearings, where each
    ball is an element in the scene, rather than as a single mesh
    with the races and multiple balls. 
    
    If the combine_bodies option is set, the loader will combine 
    mesh groups with the same name and transform into a single mesh.

    Parameters
    ------------
    file_obj:       file object
    combine_bodies: bool, whether to combine bodies with the
                    same name or not. 

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

    # load information about vertices and faces
    for obj in tree.iter('{*}object'):
        # not required, so use a get call which will return None
        # if the tag isn't populated
        name = str(obj.get('name'))
        # id is mandatory
        index = obj.attrib['id']
        # store the name by index
        id_name[index] = name

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

    # load information about the scene graph
    # each instance is a single geometry
    geometries = []
    # and a single (4,4) homogenous transform
    transforms = []
    # scene graph information stored here, aka "build" the scene
    build = tree.find('{*}build')
    for item in build.iter('{*}item'):
        # the index of the geometry this item instantiates
        geometry = item.attrib['objectid']
        transform = np.eye(4, dtype=np.float64)
        if 'transform' in item.attrib:
            # wangle their transform format
            values = np.array(item.attrib['transform'].split(),
                              dtype=np.float64).reshape((4, 3)).T
            transform[:3, :4] = values

        transforms.append(transform)
        geometries.append(geometry)
    transforms = np.array(transforms, dtype=np.float64)

    # 3MF files can only have one body per object
    # multibody parts are exported as one mesh per body
    # which looks super dumb on things like ball bearings
    # where each ball is returned as a single mesh rather
    # than the races and multiple balls
    if combine_bodies:
        # group objects that have the same name and transform
        name_hash = np.array([hash(g) for g in geometries],
                             dtype=np.float64)
        # stack name and transform
        check = np.column_stack((transforms.reshape((-1, 16)),
                                 name_hash))
        # groups indexing transforms and geometries
        groups = grouping.group_rows(check, digits=4)
        # hash geometry groups
        geometry_hash = [hash(''.join(sorted(geometries[i] for i in g))) for
                         g in groups]
        # find the groups of bodies with the same name and transform
        (junk,
         unique,
         inverse) = np.unique(geometry_hash,
                              return_index=True,
                              return_inverse=True)
        # {mesh name : dict with vertices and faces}
        meshes = {}
        # { unique geometry ID : mesh name }
        meshes_name = {}
        for g in unique:
            group = groups[g]
            # what is the geometry ID for the group
            gid = [geometries[i] for i in group]
            # they should all share the same name
            name = id_name[gid[0]]
            # if the mesh has already been included by name
            # append a unique id
            if name in meshes:
                name += util.unique_id()
            # append the group into a single mesh
            v, f = util.append_faces([v_seq[i] for i in gid],
                                     [f_seq[i] for i in gid])
            meshes_name[g] = name
            meshes[name] = {'vertices': v,
                            'faces': f}
        # create a graph with our combined meshes
        graph = []
        for i, group in zip(inverse, groups):
            name = meshes_name[unique[i]]
            graph.append({'frame_from': 'world',
                          'frame_to': util.unique_id(),
                          'geometry': name,
                          'matrix': transforms[group[0]]})

    else:
        # we are not combining bodies so our scene will
        # have one mesh per 3MF object
        graph = []
        for gid, transform in zip(geometries, transforms):
            # refer to mesh by given name and geometry ID
            # as names may be duplicated but geometry ID's aren't
            mesh_name = id_name[gid] + '-' + gid
            graph.append({'frame_from': 'world',
                          'frame_to': util.unique_id(),
                          'geometry': mesh_name,
                          'matrix': transform})
        # one mesh per geometry ID
        meshes = {}
        for gid, mesh_name in id_name.items():
            mesh_name += '-' + gid
            meshes[mesh_name] = {'vertices': v_seq[gid],
                                 'faces': f_seq[gid]}
    # construct the kwargs to load the scene
    kwargs = {'base_frame': 'world',
              'graph': graph,
              'geometry': meshes}

    return kwargs


# do import here to keep lxml a soft dependancy
try:
    from lxml import etree
    _three_loaders = {'3mf': load_3MF}
except ImportError:
    _three_loaders = {}
