import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

from trimesh import util
from trimesh import visual
from trimesh.constants import log


TOL_ZERO = 1e-12


def parse_mtl(mtl, resolver=None):
    """
    Parse a loaded MTL file.

    Parameters
    -------------
    mtl : str or bytes
      Data from an MTL file

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
            name = split[1]

            if material is not None:
                # save the old material by old name and remove key
                materials[material.pop('newmtl')] = material

            # start a fresh new material
            material = {'newmtl': split[1]}

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


def unmerge(faces, faces_tex):
    """
    Textured meshes can come with faces referencing vertex
    indices (`v`) and an array the same shape which references
    vertex texture indices (`vt`).

    Parameters
    -------------
    faces : (n, d) int
      References vertex indices
    faces_tex : (n, d) int
      References a list of UV coordinates

    Returns
    -------------
    new_faces : (m, d) int
      New faces for masked vertices
    mask_v : (p,) int
      A mask to apply to vertices
    mask_vt : (p,) int
      A mask to apply to vt array to get matching UV coordinates
    """
    # stack into pairs of (vertex index, texture index)
    stack = np.column_stack((faces.reshape(-1),
                             faces_tex.reshape(-1)))
    # find unique pairs: we're trying to avoid merging
    # vertices that have the same position but different
    # texture coordinates
    unique, inverse = trimesh.grouping.unique_rows(stack)

    # only take the unique pairts
    pairs = stack[unique]
    # try to maintain original vertex order
    order = pairs[:, 0].argsort()
    # apply the order to the pairs
    pairs = pairs[order]

    # the mask for vertices, and mask for vt to generate uv coordinates
    mask_v, mask_uv = pairs.T

    # we re-ordered the vertices to try to maintain
    # the original vertex order as much as possible
    # so to reconstruct the faces we need to remap
    remap = np.zeros(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))

    # the faces are just the inverse with the new order
    new_faces = remap[inverse].reshape((-1, 3))

    # we should NOT have messed up the faces
    # note: this is EXTREMELY slow due to the numerous
    # float comparisons so only use in unit tests
    if True or trimesh.tol.strict:
        assert np.allclose(v[faces], v[mask_v][new_faces])

    return new_faces, mask_v, mask_uv


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
            log.warning('only triangle and quad faces supported!')
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


if __name__ == '__main__':

    """
    OBJ is a free form hippy hot tub of a format which allows pretty much anything. Unfortunatly, for this reason it is extremely popular.

    Our current loader supports a lot of things and is vectorized in a bunch of nice places and is decently performant. However, it is pretty convoluted and is tough to "grok" enough to develop on.

 This PR includes a mostly fresh pass at an OBJ loader. In my testing on large files, it was roughly 3x faster than the previous loader. Most of the gains are from doing string preprocessing operations, and processing only relevant substrings as much as possible, through `str.find` and `str.rfind`. For small meshes, it is similar or slightly slower than the old loader.

    There is also a fallback method which loops through every face.

    Scope
    -------------
    - [x] Load vertices (`v`)
    - [ ] Vertex colors (on the same line as `v`)
    - [x] Vertex normals (`vn`)
    - [x] Vertex texture coordinates (`vt`)
    - [x] Triangular and quad faces
    - [x] Multiple materials
    - [x] Multiple objects (`o`)
    - [ ] Face groups `g`
    - [ ] Smoothing groups `s`
    - [x] Useable kwargs
    - [x] Usable kwargs with texture
    - [ ] Uses names from OBJ in scene

    Splitting And Return Type
    ----------------------------
    Rather than return a single mesh, this returns a scene containing
    a new mesh split at every `usemtl` or `o` tag.
    """

    name = 'models/fuze.obj'
    #name = 'src.obj'
    ##name = 'models/cube_compressed.obj'
    name = 'model.obj'
    name = 'airplane/models/model_normalized.obj'

    with open(name, 'r') as f:
        text = f.read()

    import time
    import trimesh
    import cProfile
    import pstats
    import io
    trimesh.util.attach_to_log()

    tic = [time.time()]

    # benchmark against the old loader
    with open(name, 'r') as f:
        r = trimesh.exchange.wavefront.load_wavefront(f)
    tic.append(time.time())

    pr = cProfile.Profile()
    pr.enable()

    # create a fun little resolver
    resolver = trimesh.visual.resolvers.FilePathResolver(name)

    # Load Materials
    materials = None
    mtl_position = text.find('mtllib')
    if mtl_position >= 0:
        # take the line of the material file after `mtllib`
        # which should be the file location of the .mtl file
        mtl_path = text[mtl_position + 6:text.find('\n', mtl_position)]
        # use the resolver to get the data, then parse the MTL
        material_kwargs = parse_mtl(resolver[mtl_path], resolver=resolver)
        materials = {k: visual.texture.SimpleMaterial(**v)
                     for k, v in material_kwargs.items()}

    # Load Vertices
    # aggressivly reduce blob to only part with vertices
    # the first position of a vertex in the text blob
    v_start = text.find('\nv ') - 3
    # we only need to search from the start of the file
    # up to the location of out our first vertex
    vn_start = text.find('\nvn ', 0, v_start) - 4
    vt_start = text.find('\nvt ', 0, v_start) - 4
    start = min(i for i in [v_start, vt_start, vn_start] if i > 0)
    # search for the first newline past the last vertex
    v_end = text.find('\n', text.rfind('\nv ') + 3)
    # we only need to search from the last
    # vertex up until the end of the file
    vt_end = text.find('\n', text.rfind('\nvt ', v_end) + 4)
    vn_end = text.find('\n', text.rfind('\nvn ', v_end) + 4)
    # take the last position of any vertex property
    end = max(i for i in [v_end, vt_end, vn_end] if i > 0)
    # make a giant string numpy array of each "word"
    words = np.array(text[start:end].split())

    # find indexes of the three values after a "vertex" key
    # this strategy avoids having to loop through the giant
    # vertex array but does discard vertex colors if specified
    v_idx = np.nonzero(words == 'v')[0].reshape((-1, 1))
    # do the type conversion with built- in map/list/float
    # vs np.astype, which is roughly 2x slower and these
    # are some of the most expensive operations in the whole loader
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
    v = np.array(list(map(float, v_list)), dtype=np.float64).reshape((-1, 3))

    # vertex colors are stored right after the vertices
    vc = None
    try:
        # try just one line, which will raise before
        # we try to do the whole array
        words[v_idx[0] + np.arange(4, 7)].astype(np.float64)
        # we made it past one line, try to get a color for every vertex
        vc_list = words[v_idx + np.arange(4, 7)].ravel().tolist()
        vc = np.array(list(map(float, vc_list)), dtype=np.float64).reshape((-1, 3))
    except BaseException:
        pass

    vt = None
    if vt_end >= 0:
        # if we have vertex textures specified convert to numpy array
        vt_idx = np.nonzero(words == 'vt')[0].reshape((-1, 1))
        vt_list = words[vt_idx + np.arange(1, 3)].ravel().tolist()
        vt = np.array(list(map(float, vt_list)), dtype=np.float64).reshape((-1, 2))

    vn = None
    if vn_end >= 0:
        # if we have vertex normals specified convert to numpy array
        vn_idx = np.nonzero(words == 'vn')[0].reshape((-1, 1))
        if len(vn_idx) == len(v):
            vn_list = words[vn_idx + np.arange(1, 4)].ravel().tolist()
            vn = np.array(list(map(float, vn_list)), dtype=np.float64).reshape((-1, 3))

    # Pre-Process Face Text
    # Rather than looking at each line in a loop we're
    # going to split lines by directives which indicate
    # a new mesh, specifically 'usemtl' and 'o' keys
    # search for materials, objects, faces, or groups
    starters = ['\nusemtl ', '\no ', '\nf ', '\ng ', '\ns ']
    f_start = len(text)
    # first index of material, object, face, group, or smoother
    for st in starters:
        current = text.find(st, 0, f_start)
        if current < 0:
            continue
        current += len(st)
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
        current_material = m_chunk[:newline].strip()
        # material chunk contains multiple objects
        o_split = m_chunk.split('\no ')
        if len(o_split) > 1:
            for o_chunk in o_split:
                # set the object label
                current_object = o_chunk[
                    :o_chunk.find('\n')].strip()
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
    kwargs = []
    for material, obj, chunk in face_tuples:
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
            # faces are going to be the first value
            index = np.arange(3) * int(columns / 3.0)
            # slice the faces out of the blob array
            faces = array[:, index]

            texture, normals = None, None
            if columns == 6:
                # if we have two values per vertex the second
                # one is index of texture coordinate (`vt`)
                faces_tex = array[:, index + 1]
            elif columns == 9:
                # if we have three values per vertex
                # second value is always texture
                faces_tex = array[:, index + 1]
                # third value is reference to vertex normal (`vn`)
                normals = array[:, index + 2]
        else:
            # if we had something annoying like mixed in quads
            # or faces that differ per-line we have to loop
            log.warning('inconsistent faces!')
            # TODO: allow fallback, and find a mesh we can test it on
            assert False
            faces, faces_tex, normals = _parse_faces(face_lines)

        visual = None
        if faces_tex is not None:
            # texture is referencing vt
            faces, mask_v, mask_vt = unmerge(faces=faces, faces_tex=faces_tex)

            try:
                visual = trimesh.visual.TextureVisuals(
                    uv=vt[mask_vt], material=materials[material])
            except BaseException:
                visual = None
            kwargs.append({'vertices': v[mask_v],
                           'vertex_normals': normals,
                           'visual': visual,
                           'faces': faces})

    tic.append(time.time())

    # ... do something ...
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    print('\n\nOld loader: {:0.3f} ms\nNew loader: {:0.3f} ms\nImprovement: {factor:0.3f}x'.format(
        *np.diff(tic) * 1000, factor=np.divide(*np.diff(tic))))

    m = trimesh.Scene([trimesh.Trimesh(**k) for k in kwargs])
