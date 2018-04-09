import generic as g


class LoaderTest(g.unittest.TestCase):

    def test_obj_groups(self):
        # a wavefront file with groups defined
        mesh = g.get_mesh('groups.obj')

        # make sure some data got loaded
        assert g.trimesh.util.is_shape(mesh.faces, (-1, 3))
        assert g.trimesh.util.is_shape(mesh.vertices, (-1, 3))

        # make sure groups are the right length
        assert len(mesh.metadata['face_groups']) == len(mesh.faces)

        # check to make sure there is signal not just zeros
        assert mesh.metadata['face_groups'].ptp() > 0

    def test_obj_quad(self):
        mesh = g.get_mesh('quadknot.obj')
        # make sure some data got loaded
        assert g.trimesh.util.is_shape(mesh.faces, (-1, 3))
        assert g.trimesh.util.is_shape(mesh.vertices, (-1, 3))

        assert mesh.is_watertight
        assert mesh.is_winding_consistent

    def test_obj_multiobj(self):
        # test a wavefront file with multiple objects in the same file
        meshes = g.get_mesh('two_objects.obj')
        self.assertTrue(len(meshes) == 2)

        for mesh in meshes:
            # make sure some data got loaded
            assert g.trimesh.util.is_shape(mesh.faces, (-1, 3))
            assert g.trimesh.util.is_shape(mesh.vertices, (-1, 3))

            assert mesh.is_watertight
            assert mesh.is_winding_consistent

    def test_obj_split_attributes(self):
        # test a wavefront file where pos/uv/norm have different indices
        # and where multiple objects share vertices
        # Note 'process=False' to avoid merging vertices
        meshes = g.get_mesh('joined_tetrahedra.obj', process=False)
        self.assertTrue(len(meshes) == 2)
        assert g.trimesh.util.is_shape(meshes[0].faces, (4, 3))
        assert g.trimesh.util.is_shape(meshes[0].vertices, (9, 3))
        assert g.trimesh.util.is_shape(meshes[1].faces, (4, 3))
        assert g.trimesh.util.is_shape(meshes[1].vertices, (9, 3))

    def test_obj_simple_order(self):
        # test a simple wavefront model without split indexes
        # and make sure we don't reorder vertices unneccessarily
        file_name = g.os.path.join(g.dir_models,
                                   'cube.OBJ')

        # load a simple OBJ file without merging vertices
        m = g.trimesh.load(file_name, process=False)

        # we're going to load faces in a basic text way
        # and compare the order from this method to the
        # trimesh loader, to see if we get the same thing
        faces = []
        with open(file_name, 'r') as f:
            for line in f:
                line = line.strip()
                if line[0] != 'f':
                    continue
                faces.append(line[1:].strip().split())
        # get faces as basic numpy array
        faces = g.np.array(faces, dtype=g.np.int64) - 1

        # trimesh loader should return the same face order
        assert g.np.allclose(faces, m.faces)

    def test_obj_compressed(self):
        mesh = g.get_mesh('cube_compressed.obj', process=False)

        assert g.np.allclose(g.np.abs(mesh.vertex_normals).sum(axis=1),
                             1.0)

    def test_stl(self):
        model = g.get_mesh('empty.stl')
        assert model.is_empty

    def test_3MF(self):
        # an assembly with instancing
        s = g.get_mesh('counterXP.3MF')
        # should be 2 unique meshes
        assert len(s.geometry) == 2
        # should be 6 instances around the scene
        assert len(s.graph.nodes_geometry) == 6

        # a single body 3MF assembly
        s = g.get_mesh('featuretype.3MF')
        # should be 2 unique meshes
        assert len(s.geometry) == 1
        # should be 6 instances around the scene
        assert len(s.graph.nodes_geometry) == 1


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
