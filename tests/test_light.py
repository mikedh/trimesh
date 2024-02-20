try:
    from . import generic as g
except BaseException:
    import generic as g


class LightTests(g.unittest.TestCase):
    def test_basic(self):
        for light_class in [
            g.trimesh.scene.lighting.DirectionalLight,
            g.trimesh.scene.lighting.PointLight,
            g.trimesh.scene.lighting.SpotLight,
        ]:
            light = light_class()
            assert isinstance(light.intensity, float)
            assert light.color.shape == (4,)
            assert light.color.dtype == g.np.uint8

    def test_scene(self):
        s = g.get_mesh("duck.dae")
        assert len(s.lights) > 0
        assert isinstance(s.camera, g.trimesh.scene.cameras.Camera)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
