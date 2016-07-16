import generic as g

class PermutateTest(g.unittest.TestCase):
    def test_permutate(self):
        def make_assertions(mesh, test):
            self.assertFalse(g.np.allclose(test.faces, 
                                           mesh.faces))
            self.assertFalse(g.np.allclose(test.face_adjacency, 
                                           mesh.face_adjacency))
            self.assertFalse(g.np.allclose(test.face_adjacency_edges, 
                                           mesh.face_adjacency_edges))
            self.assertFalse(g.np.allclose(test.extents,
                                           mesh.extents))
            self.assertFalse(test.md5() == mesh.md5())

        for mesh in g.get_meshes(5):
            original = g.deepcopy(mesh)
            noise     = g.trimesh.permutate.noise(mesh)
            noise_1   = g.trimesh.permutate.noise(mesh, magnitude=mesh.scale/10.0)
            transform = g.trimesh.permutate.transform(mesh)

            make_assertions(mesh, noise)
            make_assertions(mesh, noise_1)
            make_assertions(mesh, transform)
            self.assertTrue(original.md5() == mesh.md5())

    def test_tesselation(self):
        for mesh in g.get_meshes(5):
            tess = g.trimesh.permutate.tesselation(mesh)
            #print(tess.area-mesh.area)
            self.assertTrue(tess.area   - mesh.area   < g.tol.merge)
            self.assertTrue(tess.volume - mesh.volume < g.tol.merge)
            self.assertTrue(len(mesh.faces) < len(tess.faces))
            if mesh.is_winding_consistent:
                self.assertTrue(tess.is_winding_consistent)
            if mesh.is_watertight:
                self.assertTrue(tess.is_watertight)
           
    def test_base(self):
        for mesh in g.get_meshes(1):
            tess = mesh.permutated.tesselation()
            noise = mesh.permutated.noise()
            noise = mesh.permutated.noise(magnitude=mesh.scale/10)
            transform = mesh.permutated.transform()
            
            
if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
    
