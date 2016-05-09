import generic as g

class DXFTest(g.unittest.TestCase):
    def setUp(self):
        self.drawings = [g.trimesh.load_path(i) for i in g.data['2D_files']]
        self.single = g.np.hstack([i.split() for i in self.drawings])

    def test_export(self):
        for p in self.single:
            p.vertices /= p.scale
            p.export(file_obj='res.dxf')
            r = g.trimesh.load('res.dxf')
            ratio = abs(p.area - r.area) / p.area
            assert ratio < .001

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
    
