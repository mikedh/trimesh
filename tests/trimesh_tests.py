import trimesh
import unittest
import logging
import time
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

    def test_align(self):
        log.info('Testing vector alignment')
        target = np.array([0,0,1])
        for i in range(100):
            vector  = trimesh.unitize(np.random.random(3) - .5)
            T       = trimesh.geometry.align_vectors(vector, target)
            result  = np.dot(T, np.append(vector, 1))[0:3]
            aligned = np.abs(result-target).sum() < TOL_ZERO
            self.assertTrue(aligned)

    def test_horn(self):
        log.info('Testing absolute orientation')
        for i in range(10):
            points_A = (np.random.random(self.test_dim) - .5) * 100
            angle    = 4*np.pi*(np.random.random() - .5)
            vector   = trimesh.unitize(np.random.random(3) - .5)
            offset   = 100*(np.random.random(3)-.5)
            T        = trimesh.transformations.rotation_matrix(angle, vector)
            T[0:3,3] = offset
            points_B = trimesh.geometry.transform_points(points_A, T)
            M, error = trimesh.geometry.absolute_orientation(points_A, points_B, return_error=True)
            self.assertTrue(np.all(error < TOL_ZERO))

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

            tic = [time.time()]
            split     = mesh.split()
            tic.append(time.time())
            facets    = mesh.facets()
            tic.append(time.time())

            trimesh._has_gt = False

            split     = mesh.split()            
            tic.append(time.time())
            facets    = mesh.facets()
            tic.append(time.time())

            trimesh._has_gt = True

            times = np.diff(tic)

            log.info('Graph-tool sped up split by %f and facets by %f', (times[2] / times[0]), (times[3] / times[1]))

            section   = mesh.cross_section(normal=[0,0,1], origin=mesh.centroid)
            hull      = mesh.convex_hull()

            sample    = mesh.sample(1000)
            self.assertTrue(sample.shape == (1000,3))
            
            mesh.generate_face_colors()
            mesh.generate_vertex_colors()

    def test_hash(self):
        for mesh in self.meshes:
            if not mesh.is_watertight(): 
                log.warn('Hashing non- watertight mesh (%s) produces garbage!',
                         mesh.metadata['filename'])
                continue
            log.info('Hashing %s', mesh.metadata['filename'])
            result = deque()
            for i in xrange(10):
                mesh.rezero()
                matrix = trimesh.transformations.random_rotation_matrix()
                matrix[0:3,3] = (np.random.random(3)-.5)*20
                mesh.transform(matrix)
                result.append(mesh.identifier())

            ok = (np.abs(np.diff(result, axis=0)) < 1e-3).all()
            if not ok:
                log.error('Hashes on %s differ after transform! diffs:\n %s\n', 
                          mesh.metadata['filename'],
                          str(np.diff(result, axis=0)))
            self.assertTrue(ok)


            
    def test_fix_normals(self):
        for mesh in self.meshes[:2]:
            mesh.fix_normals()

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
    try: 
        from colorlog import ColoredFormatter
        formatter = ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(filename)17s:%(lineno)-4s  %(blue)4s%(message)s",
            datefmt = None,
            reset   = True,
            log_colors = {'DEBUG':    'cyan',
                          'INFO':     'green',
                          'WARNING':  'yellow',
                          'ERROR':    'red',
                          'CRITICAL': 'red' } )
    except: 

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s", "%Y-%m-%d %H:%M:%S")

    log_level      = logging.DEBUG
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    handler_stream.setLevel(log_level)
    log.setLevel(log_level)
    log.addHandler(handler_stream)
    np.set_printoptions(precision=4, suppress=True)
    unittest.main()
