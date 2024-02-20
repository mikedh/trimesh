try:
    from . import generic as g
except BaseException:
    import generic as g

try:
    import gmsh
except ImportError:
    gmsh = None


class GMSHTest(g.unittest.TestCase):
    def test_generate(self):
        # don't test gmsh if not installed
        if gmsh is None:
            return
        # create volume mesh of simple icosahedron
        m = g.trimesh.creation.icosahedron()
        result = g.trimesh.interfaces.gmsh.to_volume(m)
        assert len(result) > 0


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
