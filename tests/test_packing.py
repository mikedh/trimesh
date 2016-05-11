import generic as g
from shapely.geometry import Polygon

class PackingTest(g.unittest.TestCase):
    def setUp(self):
        self.nestable = [Polygon(i) for i in g.data['nestable']]
                      
    def test_obb(self):
        from trimesh.path import packing as packing
        inserted, transforms = packing.multipack(self.nestable)

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
    
