import generic as g

class DXFTest(g.unittest.TestCase):
    def setUp(self):
        self.drawings = [g.trimesh.load_path(i) for i in g.data['2D_files']]
        self.single = g.np.hstack([i.split() for i in self.drawings])

    def test_dxf(self):
        for p in self.single:
            p.vertices /= p.scale
            p.export(file_obj='temp.dxf')
            r = g.trimesh.load('temp.dxf')
            ratio = abs(p.area - r.area) / p.area
            if ratio > .001:
                log.error('Area ratio on export wrong! %f %f',
                          p.area,
                          r.area)
                raise ValueError('Area ratio too large')
            
if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
    
