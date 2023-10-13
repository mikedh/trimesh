try:
    from . import generic as g
except BaseException:
    import generic as g


class FacetTest(g.unittest.TestCase):
    def test_facet(self):
        m = g.get_mesh("featuretype.STL")

        assert len(m.facets) > 0
        assert len(m.facets) == len(m.facets_boundary)
        assert len(m.facets) == len(m.facets_normal)
        assert len(m.facets) == len(m.facets_area)
        assert len(m.facets) == len(m.facets_on_hull)

        # this mesh should have 8 facets on the convex hull
        assert m.facets_on_hull.astype(int).sum() == 8

    def test_empty(self):
        """
        An icosphere has no facets so make sure
        everything returns empty.
        """
        m = g.trimesh.creation.icosphere()
        assert len(m.facets) == 0
        assert len(m.facets) == len(m.facets_boundary)
        assert len(m.facets) == len(m.facets_normal)
        assert len(m.facets) == len(m.facets_area)
        assert len(m.facets) == len(m.facets_on_hull)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
