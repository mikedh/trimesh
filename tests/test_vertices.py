try:
    from . import generic as g
except BaseException:
    import generic as g


class VerticesTest(g.unittest.TestCase):

    def test_vertex_faces(self):

        # One watertight, one not; also various sizes
        meshes = [g.get_mesh('featuretype.STL'),
                  g.get_mesh('cycloidal.ply')]

        for m in meshes:
            # Choose random face indices and make sure that their vertices
            # contain those face indices
            rand_faces = g.np.random.randint(low=0, high=len(m.faces), size=100)
            for f in rand_faces:
                v_inds = m.faces[f]
                assert (f in m.vertex_faces[v_inds[0]])
                assert (f in m.vertex_faces[v_inds[1]])
                assert (f in m.vertex_faces[v_inds[2]])

            # choose some random vertices and make sure their
            # face indices are correct
            rand_vertices = g.np.random.randint(low=0, high=len(m.vertices), size=100)
            for v in rand_vertices:
                v_faces = g.np.where(m.faces == v)[0][::-1]
                assert (g.np.all(v_faces == m.vertex_faces[v][m.vertex_faces[v] >= 0]))

            # Intentionally cause fallback to looping over vertices and make sure we
            # get the same result
            loop_vertex_faces = g.trimesh.geometry.vertex_face_indices(len(m.vertices),
                                                                       m.faces,
                                                                       sparse=None)
            assert (loop_vertex_faces == m.vertex_faces).all()


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
