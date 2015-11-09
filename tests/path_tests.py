import unittest
import numpy as np
import os
from collections import deque
from shapely.geometry import Polygon
import logging

import trimesh.path as vector

from trimesh.constants import log, time_function
from trimesh.constants import tol_path as tol
from trimesh.util      import euclidean, attach_to_log

TEST_DIR   = '../models/2D'

class VectorTests(unittest.TestCase):
    def setUp(self):
        self.drawings = deque()
        file_list     = os.listdir(TEST_DIR)
        tic           = time_function()
        for filename in file_list:
            file_path = os.path.join(TEST_DIR, filename)
            tic_load = time_function()
            drawing = vector.load_path(file_path)
            toc_load = time_function()
            log.info('loaded %s in %f', filename, toc_load-tic_load)
            drawing.filename = filename
            drawing.process()
            self.drawings.append(drawing)
        toc = time_function()
        log.info('Successfully loaded %i drawings from %i files in %f seconds',
                 len(self.drawings),
                 len(file_list),
                 toc-tic)
        self.drawings = list(self.drawings)

    def test_discrete(self):
        for d in self.drawings:
            self.assertTrue(len(d.polygons_closed) == len(d.paths))
            for path in d.paths:
                verts = d.discretize_path(path)
                dists = np.sum((np.diff(verts, axis=0))**2, axis=1)**.5
                self.assertTrue(np.all(dists > tol.zero))
                circuit_test = euclidean(verts[0], verts[-1]) < tol.merge
                if not circuit_test:
                    log.error('On file %s First and last vertex distance %f', 
                              d.filename,
                              euclidean(verts[0], verts[-1]))
                self.assertTrue(circuit_test)
                is_ccw = vector.polygons.is_ccw(verts)
                if not is_ccw:
                    log.error('%s not ccw: \n%s',
                              d.filename,
                              str(verts))
                #self.assertTrue(is_ccw)
                
    def test_paths(self):
        for d in self.drawings:
            self.assertTrue(len(d.paths) == len(d.polygons_closed))
            for i in range(len(d.paths)):
                if not d.polygons_closed[i].is_valid:
                    r = d.polygons_closed[i].buffer(0.0)
                    d.show()
                self.assertTrue(d.polygons_closed[i].is_valid)
                self.assertTrue(d.polygons_closed[i].area > tol.zero)
            d.export('dict')
            d.export('svg')
            d.simplify()
            split = d.split()
            log.info('Split %s into %d bodies, checking identifiers',
                     d.filename,
                     len(split))
            for body in split:
                body.identifier()

    def test_subset(self):
        for d in self.drawings[:5]:
            m = d.medial_axis()
                

class ArcTests(unittest.TestCase):
    def setUp(self):
        self.test_points  = [[[0,0], [1.0,1], [2,0]]]
        self.test_results = [[[1,0], 1.0]]
                
    def test_center(self):
        points                 = self.test_points[0]
        res_center, res_radius = self.test_results[0]
        C, R, N, angle = vector.arc.arc_center(points)
        self.assertTrue(abs(R-res_radius) < tol.zero)
        self.assertTrue(euclidean(C, res_center) < tol.zero)

if __name__ == '__main__':
    attach_to_log()
    unittest.main()
