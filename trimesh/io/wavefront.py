import numpy as np

from ..constants import log
from .. import util


def load_wavefront(file_obj, **kwargs):
    """
    Loads an ascii Wavefront OBJ file_obj into kwargs
    for the Trimesh constructor.

    Vertices with the same position but different normals or uvs
    are split into multiple vertices.

    Colors are discarded.

    Parameters
    ----------
    file_obj : file object
                   Containing a wavefront file

    Returns
    ----------
    loaded : dict
                kwargs for Trimesh constructor
    """

    # make sure text is UTF-8 with only \n newlines
    text = file_obj.read()
    if hasattr(text, 'decode'):
        text = text.decode('utf-8')
    text = text.replace('\r\n', '\n').replace('\r', '\n') + ' \n'

    meshes = []

    def append_mesh():
        # append kwargs for a Trimesh constructor
        # to our list of meshes
        if len(current['f']) > 0:
            # get vertices as clean numpy array
            vertices = np.array(current['v'],
                                dtype=np.float64).reshape((-1, 3))
            # do the same for faces
            faces = np.array(current['f'],
                             dtype=np.int64).reshape((-1, 3))

            # get keys and values of remap as numpy arrays
            # we are going to try to preserve the order as
            # much as possible by sorting by remap key
            keys, values = (np.array(list(remap.keys())),
                            np.array(list(remap.values())))

            try:
                # if we sort keys as strings they will be an
                # ordering like (1/1/1, 10/10/10) vs (1/1/1, 2/2/2)
                # so try to convert to int before sorting
                split = np.array([i.split('/')[0] for i in keys],
                                 dtype=np.int)
                order = split.argsort()
            except BaseException:
                # we can still use arbitrary order as a fallback
                order = keys.argsort()

            # new order of vertices
            vert_order = values[order]

            # we need to mask to preserve index relationship
            # between faces and vertices
            face_order = np.zeros(len(vertices),
                                  dtype=np.int64)
            face_order[vert_order] = np.arange(len(vertices),
                                               dtype=np.int64)

            # apply the ordering and put into kwarg dict
            loaded = {'vertices': vertices[vert_order],
                      'faces': face_order[faces],
                      'metadata': {}}

            # handle vertex normals
            if len(current['vn']) > 0:
                normals = np.array(current['vn'],
                                   dtype=np.float64).reshape((-1, 3))
                loaded['vertex_normals'] = normals[vert_order]

            # handle vertex texture
            if len(current['vt']) > 0:
                texture = np.array(current['vt'], dtype=np.float64)
                # make sure vertex texture is the right shape
                # AKA (len(vertices), dimension)
                try:
                    texture = texture.reshape((len(vertices), -1))
                    # save vertex texture with correct ordering
                    loaded['metadata']['vertex_texture'] = texture[vert_order]
                except ValueError:
                    log.warning(
                        'Texture information seems broken: %s' % file_obj.name
                    )

            # build face groups information
            # faces didn't move around so we don't have to reindex
            if len(current['g']) > 0:
                face_groups = np.zeros(len(current['f']) // 3,
                                       dtype=np.int64)
                for idx, start_f in current['g']:
                    face_groups[start_f:] = idx
                loaded['metadata']['face_groups'] = face_groups

            # we're done, append the loaded mesh kwarg dict
            meshes.append(loaded)

    attribs = {k: [] for k in ['v', 'vt', 'vn']}
    current = {k: [] for k in ['v', 'vt', 'vn', 'f', 'g']}
    # remap vertex indexes {str key: int index}
    remap = {}
    next_idx = 0
    group_idx = 0

    for line in text.split("\n"):
        line_split = line.strip().split()
        if len(line_split) < 2:
            continue
        if line_split[0] in attribs:
            # v, vt, or vn
            # vertex, vertex texture, or vertex normal
            # only parse 3 values, ignore colors
            attribs[line_split[0]].append([float(x)
                                           for x in line_split[1:4]])
        elif line_split[0] == 'f':
            # a face
            ft = line_split[1:]
            if len(ft) == 4:
                # hasty triangulation of quad
                ft = [ft[0], ft[1], ft[2], ft[2], ft[3], ft[0]]
            for f in ft:
                # loop through each vertex reference of a face
                # we are reshaping later into (n,3)
                if f not in remap:
                    remap[f] = next_idx
                    next_idx += 1
                    # faces are "vertex index"/"vertex texture"/"vertex normal"
                    # you are allowed to leave a value blank, which .split
                    # will handle by nicely maintaining the index
                    f_split = f.split('/')
                    current['v'].append(attribs['v'][int(f_split[0]) - 1])
                    if len(f_split) > 1 and f_split[1] != '':
                        current['vt'].append(
                            attribs['vt'][int(f_split[1]) - 1])
                    if len(f_split) > 2:
                        current['vn'].append(
                            attribs['vn'][int(f_split[2]) - 1])
                current['f'].append(remap[f])
        elif line_split[0] == 'o':
            # defining a new object
            append_mesh()
            # reset current to empty lists
            current = {k: [] for k in current.keys()}
            remap = {}
            next_idx = 0
            group_idx = 0

        elif line_split[0] == 'g':
            # defining a new group
            group_idx += 1
            current['g'].append((group_idx, len(current['f']) // 3))

    if next_idx > 0:
        append_mesh()

    return meshes


def export_wavefront(mesh,
                     include_normals=True,
                     include_texture=True):
    """
    Export a mesh as a Wavefront OBJ file

    Parameters
    -----------
    mesh: Trimesh object

    Returns
    -----------
    export: str, string of OBJ format output
    """
    # store the multiple options for formatting
    # a vertex index for a face
    face_formats = {('v',): '{}',
                    ('v', 'vn'): '{}//{}',
                    ('v', 'vt'): '{}/{}',
                    ('v', 'vn', 'vt'): '{}/{}/{}'}
    # we are going to reference face_formats with this
    face_type = ['v']

    export = 'v '
    export += util.array_to_string(mesh.vertices,
                                   col_delim=' ',
                                   row_delim='\nv ',
                                   digits=8) + '\n'

    if include_normals and 'vertex_normals' in mesh._cache:
        # if vertex normals are stored in cache export them
        # these will have been autogenerated if they have ever been called
        face_type.append('vn')
        export += 'vn '
        export += util.array_to_string(mesh.vertex_normals,
                                       col_delim=' ',
                                       row_delim='\nvn ',
                                       digits=8) + '\n'

    if (include_texture and
        'vertex_texture' in mesh.metadata and
            len(mesh.metadata['vertex_texture']) == len(mesh.vertices)):
        # if vertex texture exists and is the right shape export here
        face_type.append('vt')
        export += 'vt '
        export += util.array_to_string(mesh.metadata['vertex_texture'],
                                       col_delim=' ',
                                       row_delim='\nvt ',
                                       digits=8) + '\n'

    # the format for a single vertex reference of a face
    face_format = face_formats[tuple(face_type)]
    faces = 'f ' + util.array_to_string(mesh.faces + 1,
                                        col_delim=' ',
                                        row_delim='\nf ',
                                        value_format=face_format)
    # add the exported faces to the export
    export += faces

    return export


_obj_loaders = {'obj': load_wavefront}
_obj_exporters = {'obj': export_wavefront}
