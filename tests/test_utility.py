import trimesh
import unittest
import logging
import time
import os
import sys
import inspect
import numpy as np
import json
from collections import deque

import generic as g

TEST_DIM = (100,3)
TOL_ZERO  = 1e-9
TOL_CHECK = 1e-2

log = logging.getLogger('trimesh')
log.addHandler(logging.NullHandler())

_QUICK = '-q' in sys.argv

class VectorTests(unittest.TestCase):
    def setUp(self):
        self.test_dim = TEST_DIM

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
            points_B = trimesh.points.transform_points(points_A, T)
            M, error = trimesh.points.absolute_orientation(points_A, points_B, return_error=True)
            self.assertTrue(np.all(error < TOL_ZERO))

class UtilTests(unittest.TestCase):
    def test_track(self):
        a = trimesh.util.tracked_array(np.random.random(TEST_DIM))
        modified = deque()
        modified.append(int(a.md5(), 16))
        a[0][0] = 10
        modified.append(int(a.md5(), 16))
        a[1] = 5
        modified.append(int(a.md5(), 16))
        a[2:] = 2
        modified.append(int(a.md5(), 16))
        self.assertTrue((np.diff(modified) != 0).all())

        modified = deque()
        modified.append(int(a.md5(), 16))
        b = a[[0,1,2]]
        modified.append(int(a.md5(), 16))
        c = a[1:]
        modified.append(int(a.md5(), 16))
        self.assertTrue((np.diff(modified) == 0).all())

class SceneTests(unittest.TestCase):
    def setUp(self):
        filename = os.path.join(g.dir_models, 'box.STL')
        mesh = trimesh.load(filename)
        split = mesh.split()
        scene = trimesh.scene.Scene(split)
        self.scene = scene

    def test_scene(self):
        duplicates = self.scene.duplicate_nodes()


class IOTest(unittest.TestCase):
    def test_dae(self):
        a = trimesh.load_mesh(os.path.abspath(os.path.join(g.dir_models, 
                                                           'ballA.off')))
        r = a.export(file_type='dae')
 
class ContainsTest(unittest.TestCase):
    def setUp(self):
        self.sphere = trimesh.load_mesh(os.path.abspath(os.path.join(g.dir_models, 
                                                                     'unit_sphere.STL')))    
    def test_equal(self):
        samples = (np.random.random((1000,3))-.5)*5
        radius = np.linalg.norm(samples, axis=1)

        margin = .025
        truth_in = radius < 1.0 - margin
        truth_out = radius > 1.0 + margin

        contains = self.sphere.contains(samples)
        
        assert contains[truth_in].all()
        assert not contains[truth_out].any()
   
class RayTests(unittest.TestCase):
    def setUp(self):
        data = g.data['ray_data']
        self.meshes = [trimesh.load_mesh(location(f)) for f in data['filenames']]
        self.rays   = data['rays']
        self.truth  = data['truth']

    def test_rays(self):
        for mesh, ray_test, truth in zip(self.meshes, self.rays, self.truth):
            hit_id      = mesh.ray.intersects_id(ray_test)
            hit_loc     = mesh.ray.intersects_location(ray_test)
            hit_any     = mesh.ray.intersects_any(ray_test)
            hit_any_tri = mesh.ray.intersects_any_triangle(ray_test)

            for i in range(len(ray_test)):
                self.assertTrue(len(hit_id[i])  == truth['count'][i])
                #self.assertTrue(len(hit_loc[i]) == truth['count'][i])

    def test_rps(self):
        dimension = (1000,3)
        sphere    = trimesh.load_mesh(location('unit_sphere.STL'))

        rays_ori = np.random.random(dimension)
        rays_dir = np.tile([0,0,1], (dimension[0], 1))
        rays_ori[:,2] = -5
        rays = np.column_stack((rays_ori, rays_dir)).reshape((-1,2,3))
        # force ray object to allocate tree before timing it
        tree = sphere.triangles_tree()
        tic = time.time()
        sphere.ray.intersects_id(rays)
        toc = time.time()
        rps = dimension[0] / (toc-tic)
        log.info('Measured %f rays/second', rps)

    def test_contains(self):
        mesh = trimesh.load_mesh(location('unit_cube.STL'))
        scale = 1+(trimesh.constants.tol.merge*2)

        test_on  = mesh.contains(mesh.vertices)
        test_in  = mesh.contains(mesh.vertices * (1.0/scale))
        test_out = mesh.contains(mesh.vertices * scale)
        
        #assert test_on.all()
        assert test_in.all()
        assert not test_out.any()

class MassTests(unittest.TestCase):
    def setUp(self):
        # inertia numbers pulled from solidworks
        self.truth = g.data['mass_properties']
        self.meshes = dict()
        for data in self.truth:
            filename = data['filename']
            location = os.path.abspath(os.path.join(g.dir_models, filename))
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
   
def location(name):
    return os.path.abspath(os.path.join(g.dir_models, name))
                
if __name__ == '__main__':
    trimesh.util.attach_to_log()
    unittest.main()
    
