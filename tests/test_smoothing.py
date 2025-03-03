try:
    from . import generic as g
except BaseException:
    import generic as g


def explicit_laplacian_calculation(mesh, equal_weight=True, pinned_vertices=None):
    """
    Exact copy of non-sparse calculation of laplacian, for correctness testing.

    Calculate a sparse matrix for laplacian operations.
    Parameters
    -------------
    mesh : trimesh.Trimesh
      Input geometry
    equal_weight : bool
      If True, all neighbors will be considered equally
      If False, all neighbors will be weighted by inverse distance
    Returns
    ----------
    laplacian : scipy.sparse.coo.coo_matrix
      Laplacian operator
    """
    # get the vertex neighbors from the cache
    neighbors = mesh.vertex_neighbors

    # if a node is pinned, it will average his coordinates by himself
    # in practice it will not move
    if pinned_vertices is not None:
        for i in pinned_vertices:
            neighbors[i] = [i]

    # avoid hitting crc checks in loops
    vertices = mesh.vertices.view(g.np.ndarray)

    # stack neighbors to 1D arrays
    col = g.np.concatenate(neighbors)
    row = g.np.concatenate([[i] * len(n) for i, n in enumerate(neighbors)])

    if equal_weight:
        # equal weights for each neighbor
        data = g.np.concatenate([[1.0 / len(n)] * len(n) for n in neighbors])
    else:
        # umbrella weights, distance-weighted
        # use dot product of ones to replace array.sum(axis=1)
        ones = g.np.ones(3)
        # the distance from verticesex to neighbors
        norms = [
            1.0
            / g.np.maximum(
                1e-6, g.np.sqrt(g.np.dot((vertices[i] - vertices[n]) ** 2, ones))
            )
            for i, n in enumerate(neighbors)
        ]
        # normalize group and stack into single array
        data = g.np.concatenate([i / i.sum() for i in norms])

    # create the sparse matrix
    matrix = g.trimesh.graph.coo_matrix((data, (row, col)), shape=[len(vertices)] * 2)

    return matrix


class SmoothTest(g.unittest.TestCase):
    def test_laplacian_calculation(self):
        m = g.trimesh.creation.icosahedron()
        m.vertices, m.faces = g.trimesh.remesh.subdivide_to_size(m.vertices, m.faces, 0.1)

        explicit_laplacian = explicit_laplacian_calculation(m)
        laplacian = g.trimesh.smoothing.laplacian_calculation(m)

        assert g.np.allclose(explicit_laplacian.toarray(), laplacian.toarray())

        explicit_laplacian = explicit_laplacian_calculation(m, equal_weight=False)
        laplacian = g.trimesh.smoothing.laplacian_calculation(m, equal_weight=False)

        assert g.np.allclose(explicit_laplacian.toarray(), laplacian.toarray())

        explicit_laplacian = explicit_laplacian_calculation(m, pinned_vertices=[0, 1, 4])
        laplacian = g.trimesh.smoothing.laplacian_calculation(
            m, pinned_vertices=[0, 1, 4]
        )

        assert g.np.allclose(explicit_laplacian.toarray(), laplacian.toarray())

        explicit_laplacian = explicit_laplacian_calculation(
            m, equal_weight=False, pinned_vertices=[0, 1, 4]
        )
        laplacian = g.trimesh.smoothing.laplacian_calculation(
            m, equal_weight=False, pinned_vertices=[0, 1, 4]
        )

        assert g.np.allclose(explicit_laplacian.toarray(), laplacian.toarray())

    def test_smooth(self):
        """
        Load a collada scene with pycollada.
        """
        m = g.trimesh.creation.icosahedron()
        m.vertices, m.faces = g.trimesh.remesh.subdivide_to_size(m.vertices, m.faces, 0.1)

        s = m.copy()
        q = m.copy()
        f = m.copy()
        d = m.copy()
        b = m.copy()
        v = m.copy()

        assert m.is_volume

        # Equal Weights
        lap = g.trimesh.smoothing.laplacian_calculation(mesh=m, equal_weight=True)

        g.trimesh.smoothing.filter_laplacian(s, 0.5, 10, False, True, lap)
        g.trimesh.smoothing.filter_laplacian(q, 0.5, 10, True, True, lap)
        g.trimesh.smoothing.filter_humphrey(f, 0.1, 0.5, 10, lap)
        g.trimesh.smoothing.filter_taubin(d, 0.5, 0.53, 10, lap)
        g.trimesh.smoothing.filter_mut_dif_laplacian(b, 0.5, 10, False, lap)
        g.trimesh.smoothing.filter_mut_dif_laplacian(v, 0.5, 10, True, lap)

        assert s.is_volume
        assert q.is_volume
        assert f.is_volume
        assert d.is_volume
        assert b.is_volume
        assert v.is_volume

        assert g.np.isclose(s.volume, m.volume, rtol=0.1)
        assert g.np.isclose(q.volume, m.volume, rtol=0.1)
        assert g.np.isclose(f.volume, m.volume, rtol=0.1)
        assert g.np.isclose(d.volume, m.volume, rtol=0.1)
        assert g.np.isclose(b.volume, m.volume, rtol=0.1)
        assert g.np.isclose(v.volume, m.volume, rtol=0.1)

        s = m.copy()
        q = m.copy()
        f = m.copy()
        d = m.copy()
        b = m.copy()
        v = m.copy()

        # umbrella Weights
        lap = g.trimesh.smoothing.laplacian_calculation(m, equal_weight=False)

        g.trimesh.smoothing.filter_laplacian(s, 0.5, 10, False, True, lap)
        g.trimesh.smoothing.filter_laplacian(q, 0.5, 10, True, True, lap)
        g.trimesh.smoothing.filter_humphrey(f, 0.1, 0.5, 10, lap)
        g.trimesh.smoothing.filter_taubin(d, 0.5, 0.53, 10, lap)
        g.trimesh.smoothing.filter_mut_dif_laplacian(b, 0.5, 10, False, lap)
        g.trimesh.smoothing.filter_mut_dif_laplacian(v, 0.5, 10, True, lap)

        assert s.is_volume
        assert q.is_volume
        assert f.is_volume
        assert d.is_volume
        assert b.is_volume
        assert v.is_volume

        assert g.np.isclose(s.volume, m.volume, rtol=0.1)
        assert g.np.isclose(q.volume, m.volume, rtol=0.1)
        assert g.np.isclose(f.volume, m.volume, rtol=0.1)
        assert g.np.isclose(d.volume, m.volume, rtol=0.1)
        assert g.np.isclose(b.volume, m.volume, rtol=0.1)
        assert g.np.isclose(v.volume, m.volume, rtol=0.1)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
