import generic as g


class PermutateTest(g.unittest.TestCase):

    def test_permutate(self):
        def close(a,b):
            if len(a) == len(b) == 0:
                return False
            if a.shape != b.shape:
                return False
            return g.np.allclose(a,b)
        
        def make_assertions(mesh, test):
            if close(test.face_adjacency,
                     mesh.face_adjacency):
                raise ValueError('face adjacency of %s the same after permutation!',
                                 mesh.metadata['file_name'])

            self.assertFalse(close(test.faces,
                                   mesh.faces))
            self.assertFalse(close(test.face_adjacency_edges,
                                   mesh.face_adjacency_edges))
            self.assertFalse(close(test.vertices,
                                   mesh.vertices))
            self.assertFalse(test.md5() == mesh.md5())

        for mesh in g.get_meshes():
            original = mesh.copy()
            
            noise = g.trimesh.permutate.noise(mesh,
                                              magnitude=mesh.scale / 50.0)
            transform = g.trimesh.permutate.transform(mesh)

            make_assertions(mesh, noise)
            make_assertions(mesh, transform)

            # make sure permutate didn't alter the original mesh
            self.assertTrue(original.md5() == mesh.md5())

    def test_tesselation(self):
        for mesh in g.get_meshes(5):
            tess = g.trimesh.permutate.tesselation(mesh)
            # print(tess.area-mesh.area)
            self.assertTrue(abs(tess.area - mesh.area) < g.tol.merge)
            volume_check = abs(tess.volume - mesh.volume) / mesh.scale
            self.assertTrue(volume_check < g.tol.merge)
            self.assertTrue(len(mesh.faces) < len(tess.faces))
            if mesh.is_winding_consistent:
                self.assertTrue(tess.is_winding_consistent)
            if mesh.is_watertight:
                self.assertTrue(tess.is_watertight)

    def test_base(self):
        for mesh in g.get_meshes(1):
            tess = mesh.permutate.tesselation()
            noise = mesh.permutate.noise()
            noise = mesh.permutate.noise(magnitude=mesh.scale / 10)
            transform = mesh.permutate.transform()


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
