import unittest
import numpy as np

import os
from collections import deque
from shapely.geometry import Polygon
import logging
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

import trimesh.path as vector

from trimesh.path.constants import *
from trimesh.path.util      import euclidean

TEST_DIR   = '../models/2D'

class VectorTests(unittest.TestCase):
    def setUp(self):
        self.drawings = deque()
        file_list     = os.listdir(TEST_DIR)
        tic           = time_function()
        for filename in file_list:
            log.info('Testing on %s', filename)
            file_path = os.path.join(TEST_DIR, filename)
            drawing = vector.load_path(file_path)
            drawing.filename = filename
            self.drawings.append(drawing)
        toc = time_function()
        log.info('Successfully loaded %i drawings from %i files in %f seconds',
                 len(self.drawings),
                 len(file_list),
                 toc-tic)
        self.drawings = list(self.drawings)

    def test_discrete(self):
        for d in self.drawings:
            self.assertTrue(len(d.polygons) == len(d.paths))
            for path in d.paths:
                verts = d.discretize_path(path)
                dists = np.sum((np.diff(verts, axis=0))**2, axis=1)**.5
                self.assertTrue(np.all(dists > TOL_ZERO))
                circuit_test = euclidean(verts[0], verts[-1]) < TOL_MERGE
                if not circuit_test:
                    log.error('On file %s First and last vertex distance %f', 
                              d.filename,
                              euclidean(verts[0], verts[-1]))
                self.assertTrue(circuit_test)
                self.assertTrue(vector.polygons.is_ccw(verts))
                
    def test_paths(self):
        for d in self.drawings:
            self.assertTrue(len(d.paths) == len(d.polygons))
            for i in range(len(d.paths)):
                #self.assertTrue(d.polygons[i].is_valid)
                self.assertTrue(d.polygons[i].area > TOL_ZERO) 

class ArcTests(unittest.TestCase):
    def setUp(self):
        self.test_points  = [[[0,0], [1.0,1], [2,0]]]
        self.test_results = [[[1,0], 1.0]]
                
    def test_center(self):
        points                 = self.test_points[0]
        res_center, res_radius = self.test_results[0]
        C, R, N, angle = vector.arc.arc_center(points)
        self.assertTrue(abs(R-res_radius) < TOL_ZERO)
        self.assertTrue(euclidean(C, res_center) < TOL_ZERO)

    def test_discrete(self):
        pass

if __name__ == '__main__':
    formatter = logging.Formatter("[%(asctime)s] %(levelname)-7s" +
                                   "(%(filename)s:%(lineno)3s) %(message)s",                               
                                   "%Y-%m-%d %H:%M:%S")
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    level = logging.INFO
    handler_stream.setLevel(level)
    for logger in logging.Logger.manager.loggerDict.values():
        if logger.__class__.__name__ != 'Logger': continue
        if logger.name in ['TerminalIPythonApp', 
                           'PYREADLINE']:
            continue
        logger.addHandler(handler_stream)
        logger.setLevel(level)

    np.set_printoptions(suppress=False)

    unittest.main()
