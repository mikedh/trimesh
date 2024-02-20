try:
    from . import generic as g
except BaseException:
    import generic as g


class VerticesTest(g.unittest.TestCase):
    def test_vertex_faces(self):
        # One watertight, one not; also various sizes
        meshes = [g.get_mesh("featuretype.STL"), g.get_mesh("cycloidal.ply")]

        for m in meshes:
            # make sure every
            vertex_faces = m.vertex_faces
            for i, v in enumerate(vertex_faces):
                # filter out negative indices
                faces = m.faces[v[v >= 0]]
                assert all(i in face for face in faces)

            # choose some random vertices and make sure their
            # face indices are correct
            rand_vertices = g.np.random.randint(low=0, high=len(m.vertices), size=100)
            for v in rand_vertices:
                v_faces = g.np.where(m.faces == v)[0][::-1]
                assert g.np.all(v_faces == m.vertex_faces[v][m.vertex_faces[v] >= 0])

            # make mesh degenerate
            m.faces[0] = [0, 0, 0]
            m.faces[1] = [1, 1, 0]
            # make sure every
            vertex_faces = m.vertex_faces
            for i, v in enumerate(vertex_faces):
                # filter out negative indices
                faces = m.faces[v[v >= 0]]
                assert all(i in face for face in faces)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
