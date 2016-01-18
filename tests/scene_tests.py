import trimesh
import unittest
import logging
import numpy as np
import networkx as nx


from trimesh.scene.transforms import EnforcedForest

log = logging.getLogger('trimesh')
log.addHandler(logging.NullHandler())

def random_chr():
    return chr(ord('a')+int(round(np.random.random()*25)))

class GraphTests(unittest.TestCase):
    def test_forest(self):
        g = EnforcedForest(assert_forest=True)
        for i in range(5000):
            g.add_edge(random_chr(), random_chr())

if __name__ == '__main__':
    trimesh.util.attach_to_log()
    unittest.main()
    
