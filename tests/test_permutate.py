try:
    from . import generic as g
except BaseException:
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
            if (
                close(test.face_adjacency, mesh.face_adjacency)
                and len(mesh.faces) > MIN_FACES
            ):
                g.log.error(f"face_adjacency unchanged: {str(test.face_adjacency)}")
                raise ValueError(
                    "face adjacency of %s the same after permutation!",
                    mesh.metadata["file_name"],
                )

            if (
                close(test.face_adjacency_edges, mesh.face_adjacency_edges)
                and len(mesh.faces) > MIN_FACES
            ):
                g.log.error(
                    f"face_adjacency_edges unchanged: {str(test.face_adjacency_edges)}"
                )
                raise ValueError(
                    "face adjacency edges of %s the same after permutation!",
                    mesh.metadata["file_name"],
                )

            assert not close(test.faces, mesh.faces)
            assert not close(test.vertices, mesh.vertices)
            assert not test.__hash__() == mesh.__hash__()

            # rigid transforms don't change area or volume
            if rigid:
                assert g.np.allclose(mesh.area, test.area)

                # volume is very dependent on meshes being watertight and sane
                if (
                    mesh.is_watertight
                    and test.is_watertight
                    and mesh.is_winding_consistent
                    and test.is_winding_consistent
                ):
                    assert g.np.allclose(mesh.volume, test.volume, rtol=0.05)

        for mesh in g.get_meshes(5):
            if len(mesh.faces) < MIN_FACES:
                continue
            # warp the mesh to be a unit cube
            mesh.vertices /= mesh.extents
            original = mesh.copy()

            for _i in range(5):
                mesh = original.copy()
                noise = g.trimesh.permutate.noise(mesh, magnitude=mesh.scale / 50.0)
                # make sure that if we permutate vertices with no magnitude
                # area and volume remain the same
                no_noise = g.trimesh.permutate.noise(mesh, magnitude=0.0)

                transform = g.trimesh.permutate.transform(mesh)
                tessellate = g.trimesh.permutate.tessellation(mesh)

                make_assertions(mesh, noise, rigid=False)
                make_assertions(mesh, no_noise, rigid=True)
                make_assertions(mesh, transform, rigid=True)
                make_assertions(mesh, tessellate, rigid=True)

            # make sure permutate didn't alter the original mesh
            assert original.__hash__() == mesh.__hash__()

    def test_tesselation(self):
        for mesh in g.get_meshes(5):
            tess = g.trimesh.permutate.tessellation(mesh)
            # g.log.debug(tess.area-mesh.area)
            assert abs(tess.area - mesh.area) < g.tol.merge
            volume_check = abs(tess.volume - mesh.volume) / mesh.scale
            assert volume_check < g.tol.merge
            assert len(mesh.faces) < len(tess.faces)
            if mesh.is_winding_consistent:
                assert tess.is_winding_consistent
            if mesh.is_watertight:
                assert tess.is_watertight

    def test_base(self):
        for mesh in g.get_meshes(1):
            tess = mesh.permutate.tessellation()  # NOQA
            noise = mesh.permutate.noise()
            noise = mesh.permutate.noise(magnitude=mesh.scale / 10)  # NOQA
            transform = mesh.permutate.transform()  # NOQA


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
