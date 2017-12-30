import generic as g

# minimum number of faces to test
# permutations on
MIN_FACES = 50


class PermutateTest(g.unittest.TestCase):

    def test_permutate(self):
        def close(a, b):
            if len(a) == len(b) == 0:
                return False
            if a.shape != b.shape:
                return False
            return g.np.allclose(a, b)

        def make_assertions(mesh, test, rigid=False):
            if (close(test.face_adjacency,
                      mesh.face_adjacency) and
                    len(mesh.faces) > MIN_FACES):
                raise ValueError('face adjacency of %s the same after permutation!',
                                 mesh.metadata['file_name'])

            if (close(test.face_adjacency_edges,
                      mesh.face_adjacency_edges) and
                    len(mesh.faces) > MIN_FACES):
                raise ValueError('face adjacency edges of %s the same after permutation!',
                                 mesh.metadata['file_name'])

            self.assertFalse(close(test.faces,
                                   mesh.faces))
            self.assertFalse(close(test.vertices,
                                   mesh.vertices))
            self.assertFalse(test.md5() == mesh.md5())

            # rigid transforms don't change area or volume
            if rigid:
                assert g.np.allclose(mesh.area, test.area)

                # volume is very dependent on meshes being watertight and sane
                if (mesh.is_watertight and
                    test.is_watertight and
                    mesh.is_winding_consistent and
                        test.is_winding_consistent):
                    assert g.np.allclose(mesh.volume, test.volume, rtol=.05)

        for mesh in g.get_meshes():
            if len(mesh.faces) < MIN_FACES:
                continue
            # warp the mesh to be a unit cube
            mesh.vertices /= mesh.extents
            original = mesh.copy()

            for i in range(5):
                mesh = original.copy()
                noise = g.trimesh.permutate.noise(mesh,
                                                  magnitude=mesh.scale / 50.0)
                # make sure that if we permutate vertices with no magnitude
                # area and volume remain the same
                no_noise = g.trimesh.permutate.noise(mesh, magnitude=0.0)

                transform = g.trimesh.permutate.transform(mesh)
                tesselate = g.trimesh.permutate.tesselation(mesh)

                make_assertions(mesh, noise, rigid=False)
                make_assertions(mesh, no_noise, rigid=True)
                make_assertions(mesh, transform, rigid=True)
                make_assertions(mesh, tesselate, rigid=True)

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
