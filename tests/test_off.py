"""
Load all the meshes we can get our hands on and check things, stuff.
"""
try:
    from . import generic as g
except BaseException:
    import generic as g


class OFFTests(g.unittest.TestCase):
    def test_comment(self):
        # see if we survive comments
        file_name = "comments.off"
        m = g.get_mesh(file_name, process=False)
        assert m.is_volume
        assert m.vertices.shape == (8, 3)
        assert m.faces.shape == (12, 3)

        with open(g.os.path.join(g.dir_models, file_name)) as f:
            lines = [line.split("#", 1)[0].strip() for line in str.splitlines(f.read())]
        lines = [line.split() for line in lines if "OFF" not in line and len(line) > 0]
        vertices = g.np.array(lines[1:9], dtype=g.np.float64)
        assert g.np.allclose(vertices, m.vertices)

    def test_whitespace(self):
        file_name = "whitespace.off"
        m = g.get_mesh(file_name, process=False)
        assert m.is_volume
        assert m.vertices.shape == (8, 3)
        assert m.faces.shape == (12, 3)

        with open(g.os.path.join(g.dir_models, file_name)) as f:
            lines = [line.split("#", 1)[0].strip() for line in str.splitlines(f.read())]
        lines = [line.split() for line in lines if "OFF" not in line and len(line) > 0]

    def test_corpus(self):
        g.get_mesh("off.zip")


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
