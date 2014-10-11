import trimesh
import unittest
import logging
from collections import deque
import os
import numpy as np
import json

TEST_DIR  = '../models'
TOL_ZERO  = 1e-9
TOL_CHECK = 1e-2
log = logging.getLogger('trimesh')
log.addHandler(logging.NullHandler())

class VectorTests(unittest.TestCase):
    def setUp(self):
        self.test_dim = (100,3)

    def test_unitize_multi(self):
        vectors = np.ones(self.test_dim)
        vectors[0] = [0,0,0]
        vectors, valid = trimesh.unitize(vectors, check_valid=True)
        
        self.assertFalse(valid[0])
        self.assertTrue(np.all(valid[1:]))
        
        length       = np.sum(vectors[1:] ** 2, axis=1) ** 2
        length_check = np.abs(length - 1.0) < TOL_ZERO
        self.assertTrue(np.all(length_check))

class MeshTests(unittest.TestCase):
    def setUp(self):
        meshes = deque()
        for filename in os.listdir(TEST_DIR):
            log.info('Attempting to load %s', filename)
            location = os.path.abspath(os.path.join(TEST_DIR, filename))
            meshes.append(trimesh.load_mesh(location))
            meshes[-1].metadata['filename'] = filename
        self.meshes = list(meshes)

    def test_meshes(self):
        for mesh in self.meshes:

            log.info('Testing %s', mesh.metadata['filename'])
            self.assertTrue(len(mesh.faces) > 0)
            self.assertTrue(len(mesh.vertices) > 0)
            
            mesh.process()
            split     = mesh.split()
            facets    = mesh.facets()
            section   = mesh.cross_section(normal=[0,0,1], origin=mesh.centroid)
            adjacency = mesh.face_adjacency()
            hull      = mesh.convex_hull()
            
            mesh.generate_face_colors()
            mesh.generate_vertex_colors()
            #mesh.fix_normals()
            
class MassTests(unittest.TestCase):
    def setUp(self):
        # inertia numbers pulled from solidworks
        self.truth  = json.load(open('mass_properties.json', 'r'))
        self.meshes = dict()
        for data in self.truth:
            filename = data['filename']
            location = os.path.abspath(os.path.join(TEST_DIR, filename))
            self.meshes[filename] = trimesh.load_mesh(location)

    def test_mass(self):
        def check_parameter(a,b):
            check = np.all(np.less(np.abs(np.array(a)-np.array(b)), TOL_CHECK))
            return check

        for truth in self.truth:
            calculated = self.meshes[truth['filename']].mass_properties(density=truth['density'])
            parameter_count = 0
            for parameter in calculated.keys():
                if not (parameter in truth): continue
                parameter_ok = check_parameter(calculated[parameter], truth[parameter])
                if not parameter_ok:
                    log.error('Parameter %s failed on file %s!', parameter, truth['filename'])
                self.assertTrue(parameter_ok)
                parameter_count += 1
            log.info('%i mass parameters confirmed for %s', parameter_count, truth['filename'])  
                
if __name__ == '__main__':
    formatter = logging.Formatter("[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s", "%Y-%m-%d %H:%M:%S")
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    handler_stream.setLevel(logging.DEBUG)
    log.setLevel(logging.DEBUG)
    log.addHandler(handler_stream)
    np.set_printoptions(precision=4, suppress=True)
    unittest.main()

