try:
    from . import generic as g
except BaseException:
    import generic as g


class PackingTest(g.unittest.TestCase):

    def setUp(self):
        from shapely.geometry import Polygon
        self.nestable = [Polygon(i) for i in g.data['nestable']]

    def test_obb(self):
        from trimesh.path import packing
        inserted, transforms = packing.multipack(self.nestable)

    def test_paths(self):
        from trimesh.path import packing
        paths = [g.trimesh.load_path(i) for i in self.nestable]

        r, inserted = packing.pack_paths(paths)

        # number of paths inserted
        count = len(g.np.unique(inserted))
        # should have inserted all our paths
        assert count == len(paths)
        # splitting should result in the right number of paths
        assert count == len(r.split())


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
