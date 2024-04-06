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

    def test_load(self):
        if gmsh is None:
            return
        r = g.trimesh.interfaces.gmsh.load_gmsh(
            g.os.path.join(g.dir_models, "featuretype.STEP")
        )
        assert isinstance(r, dict)
        assert len(g.trimesh.Trimesh(**r).faces) > 0

    def test_to_volume(self):
        if gmsh is None:
            return
        m = g.trimesh.creation.box()

        r = g.trimesh.interfaces.gmsh.to_volume(mesh=m)

        assert isinstance(r, bytes)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
