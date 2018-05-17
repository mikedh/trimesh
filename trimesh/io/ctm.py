import numpy as np
from .openctm import *

from .. import triangles

def load_ctm(file_obj, file_type=None):
    '''
    Load a OpenCTM file from a file object.
    Parameters
    ----------
    file_obj: open file- like object
    Returns
    ----------
    loaded: kwargs for a Trimesh constructor with keys:
              vertices:     (n,3) float, vertices
              faces:        (m,3) int, indexes of vertices
              face_normals: (m,3) float, normal vector of each face
    '''

    ctm = ctmNewContext(CTM_IMPORT)
    ctmLoad(ctm, bytes(file_obj.name, encoding='utf-8'))

    # Get the mesh properties
    vertexCount = ctmGetInteger(ctm, CTM_VERTEX_COUNT)
    triangleCount = ctmGetInteger(ctm, CTM_TRIANGLE_COUNT)
    hasNormals = ctmGetInteger(ctm, CTM_HAS_NORMALS)

    # Get indices
    pindices = ctmGetIntegerArray(ctm, CTM_INDICES)

    # Get vertices
    pvertices = ctmGetFloatArray(ctm, CTM_VERTICES)

    # Get normals
    if ctmGetInteger(ctm, CTM_HAS_NORMALS) == CTM_TRUE:
        print('has face normals!')
        pnormals = ctmGetFloatArray(ctm, CTM_NORMALS)
    else:
        pnormals = None

    # Create vertices, faces and normals
    vertices = []
    for i in range(vertexCount):
        vertices.append([pvertices[i * 3], pvertices[i * 3 + 1], pvertices[i * 3 + 2]])
    faces = []
    for i in range(triangleCount):
        faces.append([pindices[i * 3], pindices[i * 3 + 1], pindices[i * 3 + 2]])
    face_normals =[]
    if pnormals:
        i = 0
        for v in vertices:
            face_normals.append([pnormals[i], pnormals[i + 1], pnormals[i + 2]])
            i += 3
    else:
        # calculate them
        triangls = []
        for i, face in enumerate(faces):
            triangls.append([vertices[face[0]], vertices[face[1]], vertices[face[2]]])
        face_normals, _ = triangles.normals(triangles=np.array(triangls))

    result = {'vertices': np.array(vertices),
              'face_normals': np.array(face_normals),
              'faces': np.array(faces)}
    return result

_ctm_loaders = {'ctm': load_ctm}
