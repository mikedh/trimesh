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

            g.trimesh.repair.broken_faces(mesh, color=[255, 0, 0, 255])

            mesh.fill_holes()
            # should be a superset of the last two
            assert mesh.is_volume
            assert mesh.is_watertight
            assert mesh.is_winding_consistent

    def test_fix_normals(self):
        for mesh in g.get_meshes(5):
            mesh.fix_normals()


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
