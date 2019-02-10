import io
import uuid

import numpy as np

try:
    import collada
except:
    pass

from .. import visual

def load_collada(file_obj, **kwargs):
    """Load a COLLADA (.dae) file into a list of trimesh kwargs.

    Parameters
    ----------
    file_obj : file object
      Containing a COLLADA file
    resolver : trimesh.visual.Resolver or None
      For loading referenced files, like texture images
    kwargs : **
      Passed to trimesh.Trimesh.__init__

    Returns
    -------
    loaded : list of dict
      kwargs for Trimesh constructor
    """
    # Get meshes
    meshes = []
    c = collada.Collada(file_obj)
    for node in c.scene.nodes:
        _parse_node(node, np.eye(4), meshes)
    return meshes

def export_collada(mesh, **kwargs):
    """Export a mesh or a list of meshes as a COLLADA .dae file.

    Parameters
    -----------
    mesh: Trimesh object or list of Trimesh objects
        The mesh(es) to export.

    Returns
    -----------
    export: str, string of COLLADA format output
    """
    meshes = mesh
    if not isinstance(mesh, list) and not isinstance(mesh, tuple):
        meshes = [mesh]

    c = collada.Collada()
    nodes = []
    for i, m in enumerate(meshes):

        # Load uv, colors, materials
        uv = None
        colors = None
        mat = _unparse_material(None)
        if m.visual.defined:
            if m.visual.kind == 'texture':
                mat = _unparse_material(m.visual.material)
                uv = m.visual.uv
            elif m.visual.kind == 'vertex':
                colors = (m.visual.vertex_colors / 255.0)[:,:3]
        c.effects.append(mat.effect)
        c.materials.append(mat)

        # Create geometry object
        vertices = collada.source.FloatSource('verts-array', m.vertices.flatten(), ('X', 'Y', 'Z'))
        normals = collada.source.FloatSource('normals-array', m.vertex_normals.flatten(), ('X', 'Y', 'Z'))
        input_list = collada.source.InputList()
        input_list.addInput(0, 'VERTEX', '#verts-array')
        input_list.addInput(1, 'NORMAL', '#normals-array')
        arrays = [vertices, normals]
        if uv is not None:
            texcoords = collada.source.FloatSource('texcoords-array', uv.flatten(), ('U', 'V'))
            input_list.addInput(2, 'TEXCOORD', '#texcoords-array')
            arrays.append(texcoords)
        if colors is not None:
            colors = collada.source.FloatSource('colors-array', colors.flatten(), ('R', 'G', 'B'))
            input_list.addInput(3, 'COLOR', '#colors-array')
            arrays.append(colors)
        geom = collada.geometry.Geometry(
            c, uuid.uuid4().hex, uuid.uuid4().hex, arrays
        )
        indices = np.repeat(m.faces.flatten(), len(arrays))

        matref = 'material{}'.format(i)
        triset = geom.createTriangleSet(indices, input_list, matref)
        geom.primitives.append(triset)
        c.geometries.append(geom)

        matnode = collada.scene.MaterialNode(matref, mat, inputs=[])
        geomnode = collada.scene.GeometryNode(geom, [matnode])
        node = collada.scene.Node('node{}'.format(i), children=[geomnode])
        nodes.append(node)
    scene = collada.scene.Scene('scene', nodes)
    c.scenes.append(scene)
    c.scene = scene

    b = io.BytesIO()
    c.write(b)
    b.seek(0)
    return b.read()

def _parse_node(node, parent_matrix, meshes):
    """Recursively parse COLLADA scene nodes.
    """

    # Parse mesh node
    if isinstance(node, collada.scene.GeometryNode):
        geometry = node.geometry
        material_nodes = node.materials

        # Create material map
        material_map = {}
        for n in material_nodes:
            s = n.symbol
            effect = n.target.effect
            material_map[s] = _parse_material(effect)

        # Iterate over primitives of geometry
        for primitive in geometry.primitives:
            if isinstance(primitive, collada.triangleset.TriangleSet):
                vertex = primitive.vertex
                vertex_index = primitive.vertex_index
                vertices = vertex[vertex_index].reshape(len(vertex_index)*3,3)

                # Get normals if present
                normals = None
                if primitive.normal is not None:
                    normal = primitive.normal
                    normal_index = primitive.normal_index
                    normals = normal[normal_index].reshape(len(normal_index)*3,3)

                # Get colors if present
                colors = None
                s = primitive.sources
                if ('COLOR' in s and len(s['COLOR']) > 0 and len(primitive.index) > 0):
                    color = s['COLOR'][0][4].data
                    color_index = primitive.index[:,:,s['COLOR'][0][0]]
                    colors = color[color_index].reshape(len(color_index)*3,3)

                faces = np.arange(vertices.shape[0]).reshape(vertices.shape[0]//3,3)

                # Transform by parent matrix value
                vertices = np.dot(parent_matrix[:3,:3], vertices.T).T + parent_matrix[:3,3]
                if normals is not None:
                    normals = np.dot(parent_matrix[:3,:3], normals.T).T

                # Get UV coordinates if possible
                vis = None
                if colors is None and primitive.material in material_map:
                    material = material_map[primitive.material]
                    uv = None
                    if len(primitive.texcoordset) > 0:
                        texcoord = primitive.texcoordset[0]
                        texcoord_index = primitive.texcoord_indexset[0]
                        uv = texcoord[texcoord_index].reshape((len(texcoord_index)*3,2))
                    vis = visual.texture.TextureVisuals(uv=uv, material=material)


                meshes.append({
                    'vertices' : vertices,
                    'faces': faces,
                    'vertex_normals': normal,
                    'vertex_colors': colors,
                    'visual': vis,
                })

    # Recurse down tree
    elif isinstance(node, collada.scene.Node):
        if node.children is not None:
            for c in node.children:
                _parse_node(c, node.matrix, meshes)

def _parse_material(effect):
    """Turn a COLLADA effect into a trimesh material.
    """

    # Compute base color
    baseColorFactor = np.ones(4)
    baseColorTexture = None
    if isinstance(effect.diffuse, collada.material.Map):
        baseColorTexture = effect.diffuse.sampler.surface.img.pilimage
    elif effect.diffuse is not None:
        baseColorFactor = effect.diffuse

    # Compute emission color
    emissiveFactor = np.zeros(3)
    emissiveTexture = None
    if isinstance(effect.emission, collada.material.Map):
        emissiveTexture = effect.diffuse.sampler.surface.img.pilimage
    elif effect.emission is not None:
        emissiveFactor = effect.emission[:3]

    # Compute roughness
    roughnessFactor = 1.0
    if (not isinstance(effect.shininess, collada.material.Map)
        and effect.shininess is not None):
        roughnessFactor = np.sqrt(2.0 / (2.0 + effect.shininess))

    # Compute metallic factor
    metallicFactor = 0.0

    # Compute normal texture
    normalTexture = None
    if effect.bumpmap is not None:
        normalTexture = effect.bumpmap.sampler.surface.img.pilimage

    return visual.texture.PBRMaterial(
        emissiveFactor=emissiveFactor,
        emissiveTexture=emissiveTexture,
        normalTexture=normalTexture,
        baseColorTexture=baseColorTexture,
        baseColorFactor=baseColorFactor,
        metallicFactor=metallicFactor,
        roughnessFactor=roughnessFactor
    )

def _unparse_material(material):
    """Turn a trimesh material into a COLLADA material.
    """
    # TODO EXPORT TEXTURES
    if isinstance(material, visual.texture.PBRMaterial):
        diffuse = material.baseColorFactor
        if diffuse is not None:
            diffuse = list(diffuse)
        emission = material.emissiveFactor
        if emission is not None:
            emission = [*emission, 1.0]
        else:
            emission = list(emission)
        shininess = material.roughnessFactor
        if shininess is not None:
            shininess = 2.0 / shininess**2 - 2.0

        effect = collada.material.Effect(
            uuid.uuid4().hex, params=[], shadingtype='phong',
            diffuse=diffuse, emission=emission,
            specular=[1.0, 1.0, 1.0], shininess=float(shininess)
        )
        material = collada.material.Material(
            uuid.uuid4().hex, 'pbrmaterial', effect
        )
    else:
        effect = collada.material.Effect(
            uuid.uuid4().hex, params=[], shadingtype='phong'
        )
        material = collada.material.Material(
            uuid.uuid4().hex, 'defaultmaterial', effect
        )
    return material


_collada_loaders = { 'dae' : load_collada }
_collada_exporters = { 'dae' : export_collada }
