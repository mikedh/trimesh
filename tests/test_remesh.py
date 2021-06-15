try:
    from . import generic as g
except BaseException:
    import generic as g


class SubDivideTest(g.unittest.TestCase):

    def test_subdivide(self):
        meshes = [
            g.get_mesh('soup.stl'),  # a soup of random triangles
            g.get_mesh('cycloidal.ply'),  # a mesh with multiple bodies
            g.get_mesh('featuretype.STL')]  # a mesh with a single body

        for m in meshes:
            sub = m.subdivide()
            assert g.np.allclose(m.area, sub.area)
            assert len(sub.faces) > len(m.faces)

            v, f = g.trimesh.remesh.subdivide(
                vertices=m.vertices,
                faces=m.faces)

            max_edge = m.scale / 50
            v, f, idx = g.trimesh.remesh.subdivide_to_size(
                vertices=m.vertices,
                faces=m.faces,
                max_edge=max_edge,
                return_index=True)
            ms = g.trimesh.Trimesh(vertices=v, faces=f)
            assert g.np.allclose(m.area, ms.area)
            edge_len = (g.np.diff(ms.vertices[ms.edges_unique],
                                  axis=1).reshape((-1, 3))**2).sum(axis=1)**.5
            assert (edge_len < max_edge).all()

            # should be one index per new face
            assert len(idx) == len(f)
            # every face should be subdivided
            assert idx.max() == (len(m.faces) - 1)

            # check the original face index using barycentric coordinates
            epsilon = 1e-3
            for vid in f.T:
                # find the barycentric coordinates
                bary = g.trimesh.triangles.points_to_barycentric(
                    m.triangles[idx], v[vid])
                # if face indexes are correct they will be on the triangle
                # which means all barycentric coordinates are between 0.0-1.0
                assert bary.max() < (1 + epsilon)
                assert bary.min() > -epsilon
                # make sure it's not all zeros
                assert bary.ptp() > epsilon

    def test_sub(self):
        # try on some primitives
        meshes = [g.trimesh.creation.box(),
                  g.trimesh.creation.icosphere()]

        for m in meshes:
            s = m.subdivide(face_index=[0, len(m.faces) - 1])
            # shouldn't have subdivided in-place
            assert len(s.faces) > len(m.faces)
            # area should be the same
            assert g.np.isclose(m.area, s.area)
            # volume should be the same
            assert g.np.isclose(m.volume, s.volume)

    def test_uv(self):
        # get a mesh with texture
        m = g.get_mesh('fuze.obj')
        #m.show()
        # get the shape of the initial mesh
        shape = m.vertices.shape
        # subdivide the mesh
        s = m.subdivide()

        # shouldn't have changed source mesh
        assert m.vertices.shape == shape
        # subdivided mesh should have more vertices
        assert s.vertices.shape[0] > shape[0]
        # should have UV coordinates matching vertices
        assert s.vertices.shape[0] == s.visual.uv.shape[0]
        #s.show()


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
