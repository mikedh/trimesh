import generic as g


class RepairTests(g.unittest.TestCase):

    def test_fill_holes(self):
        for mesh_name in ['unit_cube.STL',
                          'machinist.XAML',
                          'round.stl',
                          'quadknot.obj']:
            mesh = g.get_mesh(mesh_name)
            if not mesh.is_watertight:
                continue
            mesh.faces = mesh.faces[1:-1]
            assert not mesh.is_watertight
            assert not mesh.is_volume

            # color some faces
            g.trimesh.repair.broken_faces(mesh,
                                          color=[255, 0, 0, 255])

            # run the fill holes operation
            mesh.fill_holes()
            # should be a superset of the last two
            assert mesh.is_volume
            assert mesh.is_watertight
            assert mesh.is_winding_consistent

    def test_fix_normals(self):
        for mesh in g.get_meshes(5):
            mesh.fix_normals()

    def test_winding(self):
        """
        Reverse some faces and make sure fix_face_winding flips
        them back.
        """

        meshes = [g.get_mesh(i) for i in
                  ['unit_cube.STL',
                   'machinist.XAML',
                   'round.stl',
                   'quadknot.obj',
                   'soup.stl']]

        for i, mesh in enumerate(meshes):
            # turn scenes into multibody meshes
            if g.trimesh.util.is_instance_named(mesh, 'Scene'):
                meta = mesh.metadata
                meshes[i] = mesh.dump().sum()
                meshes[i].metadata = meta

        timing = {}
        for mesh in meshes:
            # save the initial state
            is_volume = mesh.is_volume
            winding = mesh.is_winding_consistent

            tic = g.time.time()
            # flip faces to break winding
            mesh.faces[:4] = g.np.fliplr(mesh.faces[:4])

            # run the operation
            mesh.fix_normals()

            # make sure mesh is repaired to former glory
            assert mesh.is_volume == is_volume
            assert mesh.is_winding_consistent == winding

            # save timings
            timing[mesh.metadata['file_name']] = g.time.time() - tic
        # print timings as a warning
        g.log.warning(g.json.dumps(timing, indent=4))

    def test_multi(self):
        """
        Try repairing a multibody geometry
        """
        # create a multibody mesh with two cubes
        a = g.get_mesh('unit_cube.STL')
        b = a.copy()
        b.apply_translation([2, 0, 0])
        m = a + b

        # should be a volume: watertight, correct winding
        assert m.is_volume

        # flip one face
        m.faces[:1] = g.np.fliplr(m.faces[:1])

        # flip every face
        m.invert()

        # not a volume
        assert not m.is_volume

        m.fix_normals(multibody=False)

        # shouldn't fix inversion of one cube
        assert not m.is_volume

        # run fix normal with multibody mode
        m.fix_normals()

        # should be volume again
        assert m.is_volume

        # mesh should be volume of two boxes, and positive
        assert g.np.isclose(m.volume, g.np.abs(a.volume * 2.0))


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
