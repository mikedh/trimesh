import os.path

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

from ..constants import log
from .. import util
from .. import visual


def _get_vertex_colors(uv, mtllibs, resolver):
    """
    Turn loaded UV and materials into vertex colors.

    Parameters
    ------------
    uv : (n, 2) float
      Texture coordinates
    mttlibs : list of dict
      Material data including image names
    resolver : trimesh.visual.resolvers.Resolver
      Object to load data referenced by name

    Returns
    -----------
    colors : (n, 4) float or None
      Loaded RGBA colors
    """
    if Image is None:
        log.warning('pillow is required to load texture')
        return

    success = False
    vertex_colors = np.zeros((len(uv), 4), dtype=np.uint8)

    for mtl in mtllibs:
        # get the data for the image
        texture_data = resolver.get(mtl['map_Kd'])
        # load the actual image
        texture = Image.open(util.wrap_as_stream(texture_data))
        # append the loaded colors
        vertex_colors += visual.uv_to_color(uv, texture)
        success = True

    # nothing actually loaded
    if not success:
        return
    return vertex_colors


def parse_mtl(mtl):
    """
    Parse a loaded MTL file.

    Parameters
    -------------
    mtl : str or bytes
      Data from an MTL file

    Returns
    ------------
    data : dict
      Has keys map_Kd
    """
    # decode bytes if necessary
    if hasattr(mtl, 'decode'):
        mtl = mtl.decode('utf-8')

    # initial data layout
    data = {'newmtl': None,
            'map_Kd': None}

    # use universal newline splitting
    for line in str.splitlines(mtl.strip()):
        # clean leading/trailing whitespace and split
        line_split = line.strip().split()
        # needs to be at least two values
        if len(line_split) <= 1:
            continue
        # store the keys
        if line_split[0] in data:
            data[line_split[0]] = line_split[1]

    return data


def load_wavefront(file_obj, resolver=None, **kwargs):
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
    resolver : trimesh.visual.Resolver or None
      For loading referenced files, like MTL or textures
    kwargs : **
      Passed to trimesh.Trimesh.__init__

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

            if len(current['usemtl']) > 0:
                current_mtllibs = [mtllibs[k] for k in current['usemtl']
                                   if k in mtllibs]
                try:
                    vertex_colors = _get_vertex_colors(
                        uv=loaded['metadata']['vertex_texture'],
                        mtllibs=current_mtllibs,
                        resolver=resolver)
                    loaded['vertex_colors'] = vertex_colors
                    loaded['metadata']['mtllibs'] = current_mtllibs
                except BaseException:
                    log.error('failed to load texture!', exc_info=True)

            # this mesh is done so append the loaded mesh kwarg dict
            meshes.append(loaded)

    attribs = {k: [] for k in ['v', 'vt', 'vn']}
    current = {k: [] for k in ['v', 'vt', 'vn', 'f', 'g', 'usemtl']}
    mtllibs = {}
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
        elif line_split[0] == 'mtllib':

            # a resolver to load files
            if resolver is None:
                resolver = visual.resolvers.FilePathResolver(file_obj.name)

            # the name of the referenced material file
            mtl_name = line_split[1]
            # bytes containing MTL data
            try:
                mtl_data = resolver.get(mtl_name)
            except:
                log.error('unable to load material:', mtl_name)
                continue

            # loaded into a dict
            mtllib = parse_mtl(mtl_data)

            if 'newmtl' in mtllib:
                mtllibs[mtllib['newmtl']] = mtllib

        elif line_split[0] == 'usemtl':
            current['usemtl'].append(line_split[1])

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
