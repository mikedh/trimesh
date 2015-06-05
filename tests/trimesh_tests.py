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
            ext = os.path.splitext(filename)[-1][1:].lower() 
            if not ext in trimesh.available_formats():
                continue

            log.info('Attempting to load %s', filename)
            location = os.path.abspath(os.path.join(TEST_DIR, filename))
            meshes.append(trimesh.load_mesh(location))
            meshes[-1].metadata['filename'] = filename
        self.meshes = list(meshes)

    def test_meshes(self):

        has_gt = trimesh.graph_ops._has_gt
        if not has_gt:
            log.warn('No graph-tool to test!')

        for mesh in self.meshes:
            log.info('Testing %s', mesh.metadata['filename'])
            self.assertTrue(len(mesh.faces) > 0)
            self.assertTrue(len(mesh.vertices) > 0)
            
            mesh.process()

            tic = [time.time()]

            if has_gt:                
                split     = trimesh.graph_ops.split_gt(mesh)
                tic.append(time.time())
                facets    = trimesh.graph_ops.facets_gt(mesh)
                tic.append(time.time())

            split     = trimesh.graph_ops.split_nx(mesh) 
            tic.append(time.time())
            facets    = trimesh.graph_ops.facets_nx(mesh)
            tic.append(time.time())

            if has_gt:
                times = np.diff(tic)
                log.info('Graph-tool sped up split by %f and facets by %f', 
                         (times[2] / times[0]), (times[3] / times[1]))

            section   = mesh.section(plane_normal=[0,0,1], plane_origin=mesh.centroid)
            hull      = mesh.convex_hull()
            sample    = mesh.sample(1000)
            self.assertTrue(sample.shape == (1000,3))
            
            mesh.set_face_colors()
            mesh.verify_vertex_colors()

    def test_hash(self):
        for mesh in self.meshes:
            if not mesh.is_watertight(): 
                log.warn('Hashing non- watertight mesh (%s) produces garbage!',
                         mesh.metadata['filename'])
                continue
            log.info('Hashing %s', mesh.metadata['filename'])
            result = deque()
            for i in range(10):
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

    def test_fill_holes(self):
        for mesh in self.meshes[:5]:
            if not mesh.is_watertight(): continue
            mesh.faces = mesh.faces[1:-1]
            self.assertFalse(mesh.is_watertight())
            mesh.fill_holes()
            self.assertTrue(mesh.is_watertight())
            
    def test_fix_normals(self):
        for mesh in self.meshes[-2:]:
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
    trimesh.util.attach_stream_to_log()
    unittest.main()
