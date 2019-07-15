import numpy as np

try:
    import PIL.Image as Image
except ImportError:
    pass

from .. import util

from ..visual.color import to_float
from ..visual.texture import unmerge_faces, TextureVisuals
from ..visual.material import SimpleMaterial

from ..constants import log, tol


def load_obj(file_obj, resolver=None, **kwargs):
    """
    Load a Wavefront OBJ file into kwargs for a trimesh.Scene
    object.

    Parameters
    --------------
    file_obj : file like object
      Contains OBJ data
    resolver : trimesh.visual.resolvers.Resolver
      Allow assets such as referenced textures and
      material files to be loaded

    Returns
    -------------
    kwargs : dict
      Keyword arguments which can be loaded by
      trimesh.exchange.load.load_kwargs into a trimesh.Scene
    """

    # get text as bytes or string blob
    text = file_obj.read()

    # if text was bytes decode into string
    text = util.decode_text(text)

    # add leading and trailing newlines so we can use the
    # same logic even if they jump directly in to data lines
    text = '\n{}\n'.format(text.strip().replace('\r\n', '\n'))

    # Load Materials
    materials = {}
    mtl_position = text.find('mtllib')
    if mtl_position >= 0:
        # take the line of the material file after `mtllib`
        # which should be the file location of the .mtl file
        mtl_path = text[mtl_position + 6:text.find('\n', mtl_position)].strip()
        try:
            # use the resolver to get the data
            material_kwargs = parse_mtl(resolver[mtl_path],
                                        resolver=resolver)
            # turn parsed kwargs into material objects
            materials = {k: SimpleMaterial(**v)
                         for k, v in material_kwargs.items()}
        except IOError:
            # usually the resolver couldn't find the asset
            log.warning('unable to load materials from: {}'.format(mtl_path))
        except BaseException:
            # something else happened so log a warning
            log.warning('unable to load materials from: {}'.format(mtl_path),
                        exc_info=True)

    # Load Vertices
    # aggressivly reduce blob to only part with vertices
    # the first position of a vertex in the text blob
    v_start = text.find('\nv ')
    # we only really need to search from the start of the file
    # up to the location of out our first vertex but we
    # are going to use this check for "do we have texture"
    # determination later so search the whole stupid file
    vn_start = text.find('\nvn ')
    vt_start = text.find('\nvt ')
    # positions of first locations filtered by existence
    starts = [i for i in [v_start, vt_start, vn_start] if i >= 0]
    if len(starts) > 0:
        start = min(starts)
        # search for the first newline past the last vertex
        v_end = text.find('\n', text.rfind('\nv ') + 3)
        # we only need to search from the last
        # vertex up until the end of the file
        vt_end = text.find('\n', text.rfind('\nvt ', v_end) + 4)
        vn_end = text.find('\n', text.rfind('\nvn ', v_end) + 4)
        # take the last position of any vertex property
        end = max(i for i in [v_end, vt_end, vn_end] if i > 0)
        # make a giant string numpy array of each "word"
        words = np.array(text[start:end].replace('+e', 'e').replace('-e', 'e').split())
    else:
        # none of the vertex keys exist so the file is empty
        words = np.array([])

    # find indexes of the "vertex" keys
    # this avoids having to loop through the giant vertex array
    v_idx = np.nonzero(words == 'v')[0].reshape((-1, 1))
    # do the type conversion with built- in map/list/float
    # vs np.astype, which is roughly 2x slower and these
    # are the most expensive operations in the whole loader
    # due to the fact that they are executed on every vertex
    # In [17]: %timeit np.array(list(map(float, a.ravel().tolist())),
    #                           dtype=np.float64)
    #           1 loop, best of 3: 208 ms per loop
    # In [18]: %timeit a.astype(np.float64)
    #          1 loop, best of 3: 375 ms per loop
    #
    # get vertices as list of strings
    v_list = words[v_idx + np.arange(1, 4)].ravel().tolist()
    # run the conversion using built- in functions then re-numpy it
    v = np.array(list(map(float, v_list)),
                 dtype=np.float64).reshape((-1, 3))

    # check will generally only be run in unit tests
    # so we are allowed to do things that are slow
    if tol.strict:
        # check to make sure our subsetting
        # didn't miss any vertices
        assert len(v) == text.count('\nv ')

    # vertex colors are stored right after the vertices
    vc = None
    try:
        # try just one line, which will raise before
        # we try to do the whole array
        words[v_idx[0] + np.arange(4, 7)].astype(np.float64)
        # we made it past one line, try to get a color for every vertex
        vc_list = words[v_idx + np.arange(4, 7)]
        if len(vc_list) == len(v):
            # only get vertex colors if they have the correct shape
            vc = np.array(list(map(float, vc_list.ravel().tolist())),
                          dtype=np.float64).reshape(v.shape)
    except BaseException:
        # we don't have colors of correct shape
        pass

    vt = None
    if vt_start >= 0:
        # if we have vertex textures specified convert to numpy array
        vt_idx = np.nonzero(words == 'vt')[0].reshape((-1, 1))
        # only bother if we got the right number of indexes
        if len(vt_idx) > 0:
            vt_list = words[vt_idx + np.arange(1, 3)].ravel().tolist()
            vt = np.array(list(map(float, vt_list)),
                          dtype=np.float64).reshape((-1, 2))

    vn = None
    if vn_start >= 0:
        # if we have vertex normals specified convert to numpy array
        vn_idx = np.nonzero(words == 'vn')[0].reshape((-1, 1))
        if len(vn_idx) > 0:
            vn_list = words[vn_idx + np.arange(1, 4)].ravel().tolist()
            vn = np.array(list(map(float, vn_list)),
                          dtype=np.float64).reshape((-1, 3))

    # Pre-Process Face Text
    # Rather than looking at each line in a loop we're
    # going to split lines by directives which indicate
    # a new mesh, specifically 'usemtl' and 'o' keys
    # search for materials, objects, faces, or groups
    starters = ['\nusemtl ', '\no ', '\nf ', '\ng ', '\ns ']
    f_start = len(text)
    # first index of material, object, face, group, or smoother
    for st in starters:
        search = text.find(st, 0, f_start)
        # if not contained find will return -1
        if search < 0:
            continue
        # subtract the length of the key from the position
        # to make sure it's included in the slice of text
        current = search
        if current < f_start:
            f_start = current
    # index in blob of the newline after the last face
    f_end = text.find('\n', text.rfind('\nf ') + 3)
    # get the chunk of the file that has face information
    if f_end >= 0:
        # clip to the newline after the last face
        f_chunk = text[f_start:f_end]
    else:
        # no newline after last face
        f_chunk = text[f_start:]

    if tol.strict:
        # check to make sure our subsetting didn't miss any faces
        assert f_chunk.count('\nf ') == text.count('\nf ')

    # start with undefined objects and material
    current_object = None
    current_material = None
    # where we're going to store result tuples
    # containing (material, object, face lines)
    face_tuples = []

    # two things cause new meshes to be created: objects and materials
    # first divide faces into groups split by material and objects
    # face chunks using different materials will be treated
    # as different meshes
    for m_chunk in f_chunk.split('\nusemtl '):
        # if empty continue
        if len(m_chunk) == 0:
            continue
        # find the first newline in the chunk
        # everything before it will be the usemtl direction
        newline = m_chunk.find('\n')
        # remove internal double spaces because why wouldn't that be OK
        current_material = ' '.join(m_chunk[:newline].strip().split())
        # material chunk contains multiple objects
        o_split = m_chunk.split('\no ')
        if len(o_split) > 1:
            for o_chunk in o_split:
                # set the object label
                current_object = o_chunk[:o_chunk.find('\n')].strip()
                # find the first face in the chunk
                f_idx = o_chunk.find('\nf ')
                # if we have any faces append it to our search tuple
                if f_idx >= 0:
                    face_tuples.append(
                        (current_material,
                         current_object,
                         o_chunk[f_idx:]))
        else:
            # if there are any faces in this chunk add them
            f_idx = m_chunk.find('\nf ')
            if f_idx >= 0:
                face_tuples.append(
                    (current_material,
                     current_object,
                     m_chunk[f_idx:]))

    # Load Faces
    # now we have clean- ish faces grouped by material and object
    # so now we have to turn them into numpy arrays and kwargs
    # for trimesh mesh and scene objects
    geometry = {}
    for material, current_object, chunk in face_tuples:
        # do wangling in string form
        # we need to only take the face line before a newline
        # using builtin functions in a list comprehension
        # is pretty fast relative to other options
        # this operation is the only one that is O(len(faces))
        # [i[:i.find('\n')] ... requires a conditional
        face_lines = [i.split('\n')[0] for i in chunk.split('\nf ')[1:]]
        # then we are going to replace all slashes with spaces
        joined = ' '.join(face_lines).replace('/', ' ')
        # the fastest way to get to a numpy array
        # processes the whole string at once into a 1D array
        # also wavefront is 1- indexed (vs 0- indexed) so offset
        array = np.fromstring(
            joined, sep=' ', dtype=np.int64) - 1

        # get the number of columns rounded and converted to int
        columns = int(np.round(
            float(len(array) / len(face_lines))))

        # make sure we have the right number of values
        if len(array) == (columns * len(face_lines)):
            # reshape to columns
            array = array.reshape((-1, columns))
            # how many elements are in the first line of faces
            # i.e '13/1/13 14/1/14 2/1/2 1/2/1' is 4
            group_count = len(face_lines[0].strip().split())
            # how many elements are there for each vertex reference
            # i.e. '12/1/13' is 3
            per_ref = int(columns / group_count)
            # create an index mask we can use to slice vertex references
            index = np.arange(group_count) * per_ref
            # slice the faces out of the blob array
            faces = array[:, index]

            # TODO: probably need to support 8 and 12 columns for quads
            # or do something more general
            faces_tex, normal_idx = None, None
            if columns == 6:
                # if we have two values per vertex the second
                # one is index of texture coordinate (`vt`)
                # count how many delimiters are in the first face line
                # to see if our second value is texture or normals
                count = face_lines[0].count('/')
                if count == columns:
                    # case where each face line looks like:
                    # ' 75//139 76//141 77//141'
                    # which is vertex/nothing/normal
                    normal_idx = array[:, index + 1]
                elif count == int(columns / 2):
                    # case where each face line looks like:
                    # '75/139 76/141 77/141'
                    # which is vertex/texture
                    faces_tex = array[:, index + 1]
                else:
                    log.warning('face lines are weird: {}'.format(
                        face_lines[0]))
            elif columns == 9:
                # if we have three values per vertex
                # second value is always texture
                faces_tex = array[:, index + 1]
                # third value is reference to vertex normal (`vn`)
                normal_idx = array[:, index + 2]
        else:
            # if we had something annoying like mixed in quads
            # or faces that differ per-line we have to loop
            log.warning(
                'faces are mixed tri/quad/*, try to not do this!')
            # TODO: allow fallback, and find a mesh we can test it on
            faces, faces_tex, normal_idx = _parse_faces(face_lines)

        name = current_object
        if name is None or len(name) == 0 or name in geometry:
            name = '{}_{}'.format(name, util.unique_id())

        # try to get usable texture
        mesh = kwargs.copy()
        if faces_tex is not None:
            # convert faces referencing vertices and
            # faces referencing vertex texture to new faces
            # where each face
            new_faces, mask_v, mask_vt = unmerge_faces(faces, faces_tex)

            if tol.strict:
                # we should NOT have messed up the faces
                # note: this is EXTREMELY slow due to all the
                # float comparisons so only run this in unit tests
                assert np.allclose(v[faces], v[mask_v][new_faces])
                # faces should all be in bounds of vertives
                assert new_faces.max() < len(v[mask_v])

            try:
                # survive index errors as sometimes we
                # want materials without UV coordinates
                uv = vt[mask_vt]
            except BaseException:
                uv = None

            # mask vertices and use new faces

            mesh.update({'vertices': v[mask_v].copy(),
                         'faces': new_faces})

        else:
            # otherwise just use unmasked vertices
            uv = None
            mesh.update({'vertices': v.copy(),
                         'vertex_normals': vn,
                         'faces': faces})

            # check to make sure indexes are in bounds
            if tol.strict:
                assert faces.max() < len(v)

            # if we have vertex colors pass them
            if vc is not None:
                mesh['vertex_colors'] = vc
            # if we have vertex normals pass them
            if vn is not None and np.shape(normal_idx) == faces.shape:
                # do the crazy unmerging logic for split indices
                new_faces, mask_v, mask_vn = unmerge_faces(
                    faces, normal_idx)

                mesh['vertex_normals'] = vn[mask_vn]
                mesh['vertices'] = mesh['vertices'][mask_v]
                mesh['faces'] = new_faces
                if 'vertex_colors' in mesh:
                    mesh['vertex_colors'] = mesh['vertex_colors'][mask_v]

        if materials is not None and material in materials:
            visual = TextureVisuals(
                uv=uv, material=materials[material])
        else:
            log.warning('specified material not loaded!')
            visual = None

        mesh['visual'] = visual

        # store geometry by name
        geometry[name] = mesh

    if len(geometry) == 1:
        return next(iter(geometry.values()))

    # add an identity transform for every geometry
    graph = [{'geometry': k, 'frame_to': k, 'matrix': np.eye(4)}
             for k in geometry.keys()]

    # convert to scene kwargs
    result = {'geometry': geometry,
              'graph': graph}

    return result


def parse_mtl(mtl, resolver=None):
    """
    Parse a loaded MTL file.

    Parameters
    -------------
    mtl : str or bytes
      Data from an MTL file
    resolver : trimesh.visual.Resolver
      Fetch assets by name from files, web, or other

    Returns
    ------------
    mtllibs : list of dict
      Each dict has keys: newmtl, map_Kd, Kd
    """
    # decode bytes into string if necessary
    if hasattr(mtl, 'decode'):
        mtl = mtl.decode('utf-8')

    # current material
    material = None
    # materials referenced by name
    materials = {}
    # use universal newline splitting
    lines = str.splitlines(str(mtl).strip())

    for line in lines:
        # split by white space
        split = line.strip().split()
        # needs to be at least two values
        if len(split) <= 1:
            continue
        # the first value is the parameter name
        key = split[0]
        # start a new material
        if key == 'newmtl':
            # material name extracted from line like:
            # newmtl material_0
            if material is not None:
                # save the old material by old name and remove key
                materials[material.pop('newmtl')] = material
            # start a fresh new material
            material = {'newmtl': ' '.join(split[1:])}

        elif key == 'map_Kd':
            # represents the file name of the texture image
            try:
                file_data = resolver.get(split[1])
                # load the bytes into a PIL image
                # an image file name
                material['image'] = Image.open(
                    util.wrap_as_stream(file_data))
            except BaseException:
                log.warning('failed to load image', exc_info=True)

        elif key in ['Kd', 'Ka', 'Ks']:
            # remap to kwargs for SimpleMaterial
            mapped = {'Kd': 'diffuse',
                      'Ka': 'ambient',
                      'Ks': 'specular'}
            try:
                # diffuse, ambient, and specular float RGB
                material[mapped[key]] = [float(x) for x in split[1:]]
            except BaseException:
                log.warning('failed to convert color!', exc_info=True)

        elif material is not None:
            # save any other unspecified keys
            material[key] = split[1:]
    # reached EOF so save any existing materials
    if material:
        materials[material.pop('newmtl')] = material

    return materials


def _parse_faces(lines):
    """
    Use a slow but more flexible looping method to process
    face lines as a fallback option to faster vectorized methods.

    Parameters
    -------------
    lines : (n,) str
      List of lines with face information

    Returns
    -------------
    faces : (m, 3) int
      Clean numpy array of face triangles
    """

    # collect vertex, texture, and vertex normal indexes
    v, vt, vn = [], [], []

    # loop through every line starting with a face
    for line in lines:
        # remove leading newlines then
        # take first bit before newline then split by whitespace
        split = line.strip().split('\n')[0].split()
        # split into: ['76/558/76', '498/265/498', '456/267/456']
        if len(split) == 4:
            # triangulate quad face
            split = [split[0],
                     split[1],
                     split[2],
                     split[2],
                     split[3],
                     split[0]]
        elif len(split) != 3:
            log.warning(
                'face has {} elements! skipping!'.format(len(split)))
            continue

        # f is like: '76/558/76'
        for f in split:
            # vertex, vertex texture, vertex normal
            split = f.split('/')
            # we always have a vertex reference
            v.append(int(split[0]))

            # faster to try/except than check in loop
            try:
                vt.append(int(split[1]))
            except BaseException:
                pass
            try:
                # vertex normal is the third index
                vn.append(int(split[2]))
            except BaseException:
                pass

    # shape into triangles and switch to 0-indexed
    faces = np.array(v, dtype=np.int64).reshape((-1, 3)) - 1
    faces_tex, normals = None, None
    if len(vt) == len(v):
        faces_tex = np.array(vt, dtype=np.int64).reshape((-1, 3)) - 1
    if len(vn) == len(v):
        normals = np.array(vn, dtype=np.int64).reshape((-1, 3)) - 1

    return faces, faces_tex, normals


def export_obj(mesh,
               include_normals=True,
               include_texture=True,
               include_color=True):
    """
    Export a mesh as a Wavefront OBJ file

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Mesh to be exported

    Returns
    -----------
    export : str
      OBJ format output
    """
    # store the multiple options for formatting
    # vertex indexes for faces
    face_formats = {('v',): '{}',
                    ('v', 'vn'): '{}//{}',
                    ('v', 'vt'): '{}/{}',
                    ('v', 'vn', 'vt'): '{}/{}/{}'}
    # we are going to reference face_formats with this
    face_type = ['v']

    # OBJ includes vertex color as RGB elements on the same line
    if include_color and mesh.visual.kind in ['vertex', 'face']:
        # create a stacked blob with position and color
        v_blob = np.column_stack((
            mesh.vertices,
            to_float(mesh.visual.vertex_colors[:, :3])))
    else:
        # otherwise just export vertices
        v_blob = mesh.vertices

    # add the first vertex key and convert the array
    export = 'v ' + util.array_to_string(v_blob,
                                         col_delim=' ',
                                         row_delim='\nv ',
                                         digits=8) + '\n'

    # only include vertex normals if they're already stored
    if include_normals and 'vertex_normals' in mesh._cache:
        # if vertex normals are stored in cache export them
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


_obj_loaders = {'obj': load_obj}
_obj_exporters = {'obj': export_obj}
