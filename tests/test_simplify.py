import generic as g

class SimplifyTest(g.unittest.TestCase):
    def test_export(self):
        single = g.get_mesh('2D/cycloidal.dxf')
        single.simplify()


        
if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
    
