import trimesh
import unittest
import logging
from collections import deque
import os
import numpy as np
import json

TEST_DIR  = '../models'
TOL_ZERO  = 1e-9
TOL_CHECK = 1e-4
log = logging.getLogger('trimesh')
log.addHandler(logging.NullHandler)

'''
class VectorTests(unittest.TestCase):
    def setUp(self):
        self.vector_dim = (100,3)
        self.vectors = np.random.random(self.vector_dim)
        self.vectors = trimesh.unitize(self.vectors)

    def test_unitize(self):
        self.vectors[0:10] = [0,0,0]
        self.vectors = trimesh.unitize(self.vectors)
        self.assertTrue(np.shape(self.vectors) == self.vector_dim)
        
        norms = np.sum(self.vectors ** 2, axis=1) ** 2
        nonzero = norms > TOL_ZERO
        unit_vector = np.abs(norms[nonzero] - 1.0) < TOL_ZERO
        self.assertTrue(np.all(unit_vector))

    def test_group(self):
        tol_angle = np.radians(10)
        tol_dist  = np.tan(tol_angle) * 2

        self.vectors[0:10]  = [0.0, 0.0, 0.0]
        self.vectors[10:20] = [0.0, 0.0, 1.0]
       
        vectors, aligned = trimesh.group_vectors(self.vectors, 
                                                 TOL_ANGLE = tol_angle,
                                                 include_negative = True)
        self.assertTrue(len(vectors) == len(aligned))

        for vector, group in zip(vectors, aligned):
            dists_pos = np.sum((self.vectors[[group]] - vector)**2, axis=1)**.5
            dists_neg = np.sum((self.vectors[[group]] + vector)**2, axis=1)**.5
            dist_ok = np.logical_or((dists_pos < tol_dist), (dists_neg < tol_dist))
            self.assertTrue(np.all(dist_ok))
'''

class MeshTests(unittest.TestCase):
    def setUp(self):
        meshes = deque()
        for filename in os.listdir(TEST_DIR):
            location = os.path.abspath(os.path.join(TEST_DIR, filename))
            meshes.append(trimesh.load_mesh(location))
        self.meshes = list(meshes)

    def test_meshes(self):
        for mesh in self.meshes:
            self.assertTrue(len(mesh.faces) > 0)
            self.assertTrue(len(mesh.vertices) > 0)
            
class MassTests(unittest.TestCase):
    def setUp(self):
        # inertia numbers pulled from solidworks
        self.truth  = json.load(open('mass_properties.json', 'rb'))
        self.meshes = dict()
        for data in self.truth:
            filename = data['filename']
            location = os.path.abspath(os.path.join(TEST_DIR, filename))
            self.meshes[filename] = trimesh.load_mesh(location)

    def test_mass(self):
        def check_parameter(a,b):
            check = np.less(np.abs(np.array(a)-np.array(b)), TOL_CHECK)
            if not np.all(check):
                print a, b, check
            self.assertTrue(np.all(check))

        for truth in self.truth:
            calculated = self.meshes[truth['filename']].mass_properties()
            for parameter in calculated.keys():
                if not (parameter in truth): continue
                check_parameter(calculated[parameter], truth[parameter])

if __name__ == '__main__':
    formatter = logging.Formatter("[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s", "%Y-%m-%d %H:%M:%S")
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    handler_stream.setLevel(logging.DEBUG)
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    log.addHandler(handler_stream)
    np.set_printoptions(precision=4, suppress=True)
    unittest.main()

    density        = 0.036127
    inertia_actual = [[0.008312,0.000000,0.000000], [0.000000,0.012823,0.000807], [0.000000,0.000807, 0.009005]]
    
