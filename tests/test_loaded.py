import generic as g


class LoaderTest(g.unittest.TestCase):

    def test_obj_groups(self):
        # a wavefront file with groups defined
        mesh = g.get_mesh('groups.obj')

        # make sure some data got loaded
        assert g.trimesh.util.is_shape(mesh.faces, (-1,3))
        assert g.trimesh.util.is_shape(mesh.vertices, (-1,3))

        # make sure groups are the right length
        assert len(mesh.metadata['face_groups']) == len(mesh.faces)

        # check to make sure there is signal not just zeros
        assert mesh.metadata['face_groups'].ptp() > 0

    def test_obj_quad(self):
        mesh = g.get_mesh('quadknot.obj')
        # make sure some data got loaded
        assert g.trimesh.util.is_shape(mesh.faces, (-1,3))
        assert g.trimesh.util.is_shape(mesh.vertices, (-1,3))

        assert mesh.is_watertight
        assert mesh.is_winding_consistent

    def test_obj_multiobj(self):
        # test a wavefront file with multiple objects in the same file
        meshes = g.get_mesh('two_objects.obj')
        self.assertTrue(len(meshes) == 2)

        for mesh in meshes:
            # make sure some data got loaded
            assert g.trimesh.util.is_shape(mesh.faces, (-1,3))
            assert g.trimesh.util.is_shape(mesh.vertices, (-1,3))

            assert mesh.is_watertight
            assert mesh.is_winding_consistent

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
