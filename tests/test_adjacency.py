try:
    from . import generic as g
except BaseException:
    import generic as g


class AdjacencyTest(g.unittest.TestCase):
    def test_radius(self):
        for radius in [0.1, 1.0, 3.1459, 29.20]:
            m = g.trimesh.creation.cylinder(radius=radius, height=radius * 10)

            # remove the cylinder cap
            signs = (g.np.sign(m.vertices[:, 2]) < 0)[m.faces]
            not_cap = ~g.np.logical_or(signs.all(axis=1), ~signs.any(axis=1))
            m.update_faces(not_cap)

            # compare the calculated radius
            radii = m.face_adjacency_radius
            radii = radii[g.np.isfinite(radii)]

            assert g.np.allclose(radii, radius, atol=radius / 100)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
