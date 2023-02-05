try:
    from . import generic as g
except BaseException:
    import generic as g


def assert_no_overlap(bounds, epsilon=1e-8):
    """
    Assert that a list of axis-aligned bounding boxes
    contains no overlaps using `rtree`.

    Parameters
    ------------
    bounds : (n, 2, dimension) float
      Axis aligned bounding boxes
    epsilon : float
      Amount to shrink AABB to avoid spurious floating
      point hits.

    Raises
    --------------
    AssertionError
      If any of `bounds` hit each other.
    """
    # pad AABB by epsilon for deterministic intersections
    padded = g.np.array(bounds) + g.np.reshape([epsilon, -epsilon], (1, 2, 1))
    tree = g.trimesh.util.bounds_tree(padded)
    # every returned AABB should not overlap with any other AABB
    assert all(set(tree.intersection(current.ravel())) ==
               {i} for i, current in enumerate(bounds))


class PackingTest(g.unittest.TestCase):

    def setUp(self):
        from shapely.geometry import Polygon
        self.nestable = [Polygon(i) for i in g.data['nestable']]

    def test_obb(self):
        from trimesh.path import packing
        inserted, transforms = packing.polygons(self.nestable)

    def test_paths(self):
        from trimesh.path import packing
        paths = [g.trimesh.load_path(i) for i in self.nestable]

        r, inserted = packing.paths(paths)

        # number of paths inserted
        count = len(g.np.unique(inserted))
        # should have inserted all our paths
        assert count == len(paths)
        # splitting should result in the right number of paths
        assert count == len(r.split())

    def test_3D(self):
        from trimesh.path import packing
        e = g.np.array([[14., 14., 0.125],
                        [13.84376457, 13.84376457, 0.25],
                        [14., 14., 0.125],
                        [12.00000057, 12.00000057, 0.25],
                        [14., 14., 0.125],
                        [12.83700787, 12.83700787, 0.375],
                        [12.83700787, 12.83700787, 0.125],
                        [14., 14., 0.625],
                        [1.9999977, 1.9999509, 0.25],
                        [0.87481696, 0.87463294, 0.05],
                        [0.99955503, 0.99911677, 0.1875]])

        # try packing these 3D boxes
        bounds, consume = packing.rectangles_single(e)
        assert consume.all()
        # assert all bounds are well constructed
        assert_no_overlap(bounds)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
