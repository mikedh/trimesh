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

TEST_DIM = (100, 3)
TOL_ZERO = 1e-9
TOL_CHECK = 1e-2

log = logging.getLogger('trimesh')
log.addHandler(logging.NullHandler())

_QUICK = '-q' in sys.argv


class VectorTests(unittest.TestCase):

    def setUp(self):
        self.test_dim = TEST_DIM

    def test_unitize_multi(self):
        vectors = np.ones(self.test_dim)
        vectors[0] = [0, 0, 0]
        vectors, valid = trimesh.unitize(vectors, check_valid=True)

        self.assertFalse(valid[0])
        self.assertTrue(np.all(valid[1:]))

        length = np.sum(vectors[1:] ** 2, axis=1) ** 2
        length_check = np.abs(length - 1.0) < TOL_ZERO
        self.assertTrue(np.all(length_check))

    def test_align(self):
        log.info('Testing vector alignment')
        target = np.array([0, 0, 1])
        for i in range(100):
            vector = trimesh.unitize(np.random.random(3) - .5)
            T = trimesh.geometry.align_vectors(vector, target)
            result = np.dot(T, np.append(vector, 1))[0:3]
            aligned = np.abs(result - target).sum() < TOL_ZERO
            self.assertTrue(aligned)

    def test_horn(self):
        log.info('Testing absolute orientation')
        for i in range(10):
            points_A = (np.random.random(self.test_dim) - .5) * 100
            angle = 4 * np.pi * (np.random.random() - .5)
            vector = trimesh.unitize(np.random.random(3) - .5)
            offset = 100 * (np.random.random(3) - .5)
            T = trimesh.transformations.rotation_matrix(angle, vector)
            T[0:3, 3] = offset
            points_B = trimesh.transformations.transform_points(points_A, T)
            M, error = trimesh.points.absolute_orientation(
                points_A, points_B, return_error=True)
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
        b = a[[0, 1, 2]]
        modified.append(int(a.md5(), 16))
        c = a[1:]
        modified.append(int(a.md5(), 16))
        self.assertTrue((np.diff(modified) == 0).all())

    def test_bounds_tree(self):
        for attempt in range(3):
            for dimension in [2, 3]:
                t = g.np.random.random((1000, 3, dimension))
                bounds = g.np.column_stack((t.min(axis=1), t.max(axis=1)))
                tree = g.trimesh.util.bounds_tree(bounds)
                self.assertTrue(0 in tree.intersection(bounds[0]))

    def test_strips(self):
        '''
        Test our conversion of triangle strips to face indexes.
        '''
        
        def strips_to_faces(strips):
            '''
            A slow but straightfoward version of the function to test against
            '''
            faces = g.collections.deque()
            for s in strips:
                s = g.np.asanyarray(s, dtype=g.np.int)
                # each triangle is defined by one new vertex
                tri = g.np.column_stack([g.np.roll(s, -i) for i in range(3)])[:-2]
                # we need to flip ever other triangle
                idx = (g.np.arange(len(tri)) % 2).astype(bool)
                tri[idx] = g.np.fliplr(tri[idx])
                faces.append(tri)
            # stack into one (m,3) array
            faces = g.np.vstack(faces)
            return faces
        
        
        # test 4- triangle strip
        s = [g.np.arange(6)]
        f = g.trimesh.util.triangle_strips_to_faces(s)
        assert (f == g.np.array([[0, 1, 2],
                               [3, 2, 1],
                               [2, 3, 4],
                               [5, 4, 3]])).all()
        assert len(f) + 2  == len(s[0])   
        assert (f == strips_to_faces(s)).all()
        
        # test single triangle
        s = [g.np.arange(3)]
        f = g.trimesh.util.triangle_strips_to_faces(s)                   
        assert (f == g.np.array([[0, 1, 2]])).all()              
        assert len(f) + 2  == len(s[0])   
        assert (f == strips_to_faces(s)).all()
        
        s = [g.np.arange(100)]
        f = g.trimesh.util.triangle_strips_to_faces(s)                     
        assert len(f) + 2  == len(s[0]) 
        assert (f == strips_to_faces(s)).all()
        
        
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
        a = g.get_mesh('ballA.off')
        r = a.export(file_type='dae')


class ContainsTest(unittest.TestCase):

    def test_inside(self):
        sphere = g.trimesh.primitives.Sphere(radius=1.0, subdivisions=4)
        g.log.info('Testing contains function with sphere')
        samples = (np.random.random((1000, 3)) - .5) * 5
        radius = np.linalg.norm(samples, axis=1)

        margin = .05
        truth_in = radius < (1.0 - margin)
        truth_out = radius > (1.0 + margin)

        contains = sphere.contains(samples)

        if not contains[truth_in].all():
            raise ValueError('contains test doesnt match truth!')

        if contains[truth_out].any():
            raise ValueError('contains test doesnt match truth!')


class MassTests(unittest.TestCase):

    def setUp(self):
        # inertia numbers pulled from solidworks
        self.truth = g.data['mass_properties']
        self.meshes = dict()
        for data in self.truth:
            filename = data['filename']
            self.meshes[filename] = g.get_mesh(filename)

    def test_mass(self):
        def check_parameter(a, b):
            check = np.all(
                np.less(np.abs(np.array(a) - np.array(b)), TOL_CHECK))
            return check

        for truth in self.truth:
            calculated = self.meshes[truth['filename']].mass_properties(density=truth[
                                                                        'density'])
            parameter_count = 0
            for parameter in calculated.keys():
                if not (parameter in truth):
                    continue
                parameter_ok = check_parameter(
                    calculated[parameter], truth[parameter])
                if not parameter_ok:
                    log.error('Parameter %s failed on file %s!',
                              parameter, truth['filename'])
                self.assertTrue(parameter_ok)
                parameter_count += 1
            log.info('%i mass parameters confirmed for %s',
                     parameter_count, truth['filename'])


class SphericalTests(unittest.TestCase):

    def test_spherical(self):
        v = g.trimesh.unitize(g.np.random.random((1000, 3)) - .5)
        spherical = g.trimesh.util.vector_to_spherical(v)
        v2 = g.trimesh.util.spherical_to_vector(spherical)
        self.assertTrue((np.abs(v - v2) < g.trimesh.constants.tol.merge).all())


class HemisphereTests(unittest.TestCase):

    def test_hemisphere(self):
        v = trimesh.unitize(np.random.random((10000, 3)) - .5)
        v[0] = [0, 1, 0]
        v[1] = [1, 0, 0]
        v[2] = [0, 0, 1]
        v = np.column_stack((v, -v)).reshape((-1, 3))

        resigned = trimesh.util.vector_hemisphere(v)

        check = (abs(np.diff(resigned.reshape((-1, 2, 3)),
                             axis=1).sum(axis=2)) < trimesh.constants.tol.zero).all()
        self.assertTrue(check)


class FileTests(unittest.TestCase):

    def test_io_wrap(self):
        test_b = g.np.random.random(1).tostring()
        test_s = 'this is a test yo'

        res_b = g.trimesh.util.wrap_as_stream(test_b).read()
        res_s = g.trimesh.util.wrap_as_stream(test_s).read()

        self.assertTrue(res_b == test_b)
        self.assertTrue(res_s == test_s)

    def test_file_hash(self):
        data = g.np.random.random(10).tostring()
        path = g.os.path.join(g.dir_data, 'nestable.json')

        for file_obj in [g.trimesh.util.wrap_as_stream(data),
                         open(path, 'rb')]:
            start = file_obj.tell()

            hashed = g.trimesh.util.hash_file(file_obj)

            self.assertTrue(file_obj.tell() == start)
            self.assertTrue(hashed is not None)
            self.assertTrue(len(hashed) > 5)

            file_obj.close()


if __name__ == '__main__':
    trimesh.util.attach_to_log()
    unittest.main()
