'''
An interface to the openCASCADE BREP engine, using the occmodel bindings
'''

import numpy as np

from .base import Trimesh
from .util import three_dimensionalize
from .constants import *

# openCASCADE bindings
from occmodel import Face, Solid
from geotools import Transform

def solid_to_mesh(solid, process=True):
    occ_mesh = solid.createMesh()
    occ_mesh.optimize()
    faces    = np.array(list(occ_mesh.triangles)).reshape((-1,3)).astype(int)
    vertices = np.array(list(occ_mesh.vertices)).reshape((-1,3)).astype(float)
    mesh     = Trimesh(vertices=vertices, faces=faces, process=process)
    if process and (not mesh.is_watertight): 
        log.warning('Mesh returned from openCASCADE isn\'t watertight!')
    return mesh

def extrude_polygon(polygon,
                    vectors    = None,
                    height     = None,
                    height_pad = [0,0]):

    shell = np.array(polygon.exterior.coords)
    holes = [np.array(i.coords) for i in polygon.interiors]    
 
    if vectors is None: 
        pad_signed = np.sign(height) * np.array(height_pad)
        vectors    = np.array([[0.0, 0.0, -pad_signed[0]], 
                               [0.0, 0.0,  pad_signed[1] + height]])
    else:
        vectors = np.array(vectors)

    shell_3D      = three_dimensionalize(shell, False)
    shell_3D[:,2] = vectors[0][2]

    shell_face  = Face().createPolygonal(shell_3D)
    shell_solid = Solid().extrude(shell_face, *vectors)

    if holes is None: return shell_solid

    pad_vector  = np.diff(vectors, axis=0)[0]
    cut_vectors = vectors + [-pad_vector, pad_vector]

    for hole in holes:
        cut_face  = Face().createPolygonal(three_dimensionalize(hole, False))
        cut_solid = Solid().extrude(cut_face, *cut_vectors)
        shell_solid.cut(cut_solid)    
    return shell_solid
