try:
    from . import generic as g
except BaseException:
    import generic as g


class EdgeTest(g.unittest.TestCase):
    def test_face_unique(self):
        m = g.get_mesh("featuretype.STL")

        # our basic edges should have the same
        # unique values as our faces
        face_set = set(m.faces.ravel())
        assert set(m.edges_sorted.ravel()) == face_set
        assert set(m.edges_unique.ravel()) == face_set
        assert set(m.edges.ravel()) == face_set

        # check relation of edges_unique and faces_unique_edges
        e = m.edges_unique[m.faces_unique_edges].reshape((-1, 6))
        # should now be a row of 3 pairs of equal values
        e.sort(axis=1)
        # pairs should all be equal
        e = e.reshape((-1, 2))
        assert (e[:, 0] == e[:, 1]).all()

        # should be the same values as the original faces
        assert (e[:, 0].reshape((-1, 3)) == g.np.sort(m.faces, axis=1)).all()


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
