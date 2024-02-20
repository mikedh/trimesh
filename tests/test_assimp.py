try:
    from . import generic as g
except BaseException:
    import generic as g


class AssimpTest(g.unittest.TestCase):
    def test_duck(self):
        # load the duck using pyassimp
        try:
            import pyassimp  # NOQA
        except BaseException:
            return

        file_path = g.os.path.join(g.dir_models, "duck.dae")
        with open(file_path, "rb") as f:
            kwargs = g.trimesh.exchange.assimp.load_pyassimp(f, file_type="dae")

        scene = g.trimesh.exchange.load.load_kwargs(kwargs)

        assert len(scene.geometry) == 1
        assert len(scene.graph.nodes_geometry) == 1


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
