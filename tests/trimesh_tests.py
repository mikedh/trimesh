import trimesh
import unittest
import logging
import time
from collections import deque
import os
import sys
import inspect
import numpy as np
import json


SCRIPT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
MODELS_DIR = '../models'
TEST_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, MODELS_DIR))

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

        has_gt = trimesh.graph._has_gt
        trimesh.graph._has_gt = False

        if not has_gt:
            log.warning('No graph-tool to test!')

        log.info('Running tests on %d meshes', len(self.meshes))
        for mesh in self.meshes:
            log.info('Testing %s', mesh.metadata['filename'])
            self.assertTrue(len(mesh.faces) > 0)
            self.assertTrue(len(mesh.vertices) > 0)
            
            mesh.process()

            tic = [time.time()]

            if has_gt:
                trimesh.graph._has_gt = True 
                split     = trimesh.graph.split(mesh)
                tic.append(time.time())
                facets    = trimesh.graph.facets(mesh)
                tic.append(time.time())
                trimesh.graph._has_gt = False

            split     = trimesh.graph.split(mesh) 
            tic.append(time.time())
            facets    = trimesh.graph.facets(mesh)
            tic.append(time.time())

            facets, area = mesh.facets(1)
            faces = facets[np.argmax(area)]
            outline = mesh.outline(faces)
            smoothed = mesh.smoothed()

            if has_gt:
                times = np.diff(tic)
                log.info('Graph-tool sped up split by %f and facets by %f', 
                         (times[2] / times[0]), (times[3] / times[1]))

            section   = mesh.section(plane_normal=[0,0,1], plane_origin=mesh.centroid)
            hull      = mesh.convex_hull
            sample    = mesh.sample(1000)
            self.assertTrue(sample.shape == (1000,3))
            
    def test_hash(self):
        count = 10
        for mesh in self.meshes:
            if not mesh.is_watertight: 
                log.warning('Hashing non- watertight mesh (%s) produces garbage!',
                         mesh.metadata['filename'])
                continue
            log.info('Hashing %s', mesh.metadata['filename'])
            log.info('Trying hash at %d random transforms', count)
            result = deque()
            for i in range(count):
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
            if not mesh.is_watertight: continue
            mesh.faces = mesh.faces[1:-1]
            self.assertFalse(mesh.is_watertight)
            mesh.fill_holes()
            self.assertTrue(mesh.is_watertight)
            
    def test_fix_normals(self):
        for mesh in self.meshes[5:]:
            mesh.fix_normals()

class EqualTest(unittest.TestCase):
    def setUp(self):
        self.a = trimesh.load_mesh(os.path.abspath(os.path.join(TEST_DIR, 'ballA.off')))
        self.b = trimesh.load_mesh(os.path.abspath(os.path.join(TEST_DIR, 'ballB.off')))
    
    def test_equal(self):
        self.assertTrue(self.a == self.b)
        log.info('Mesh equality tested')

class ContainsTest(unittest.TestCase):
    def setUp(self):
        self.sphere = trimesh.load_mesh(os.path.abspath(os.path.join(TEST_DIR, 
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

class BooleanTest(unittest.TestCase):
    def setUp(self):
        self.a = trimesh.load_mesh(os.path.abspath(os.path.join(TEST_DIR, 'ballA.off')))
        self.b = trimesh.load_mesh(os.path.abspath(os.path.join(TEST_DIR, 'ballB.off')))
    
    def test_boolean(self):
        if _QUICK: return
        a, b = self.a, self.b
        d = a.difference(b)
        self.assertTrue(d.is_watertight)
        i = a.intersection(b)
        self.assertTrue(i.is_watertight)
        u = a.union(b)
        self.assertTrue(u.is_watertight)

   
class RayTests(unittest.TestCase):
    def setUp(self):
        ray_filename = os.path.join(SCRIPT_DIR, 'ray_data.json')
        with open(ray_filename, 'r') as f_obj: 
            data = json.load(f_obj)
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
        scale = 1+(trimesh.constants.tol.merge)

        test_on  = mesh.contains(mesh.vertices)
        test_in  = mesh.contains(mesh.vertices * (1.0/scale))
        test_out = mesh.contains(mesh.vertices * scale)
        
        #assert test_on.all()
        assert test_in.all()
        assert not test_out.any()

class MassTests(unittest.TestCase):
    def setUp(self):
        # inertia numbers pulled from solidworks

        mass_filename = os.path.join(SCRIPT_DIR, 'mass_properties.json')

        with open(mass_filename, 'r') as f_obj:
            self.truth  = json.load(f_obj)
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
   
def location(name):
    return os.path.abspath(os.path.join(TEST_DIR, name))
                
if __name__ == '__main__':
    trimesh.util.attach_to_log()
    unittest.main()
    
