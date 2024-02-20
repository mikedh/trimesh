try:
    from . import generic as g
except BaseException:
    import generic as g


class UnwrapTest(g.unittest.TestCase):
    def test_image(self):
        try:
            import xatlas  # noqa
        except BaseException:
            g.log.info("not testing unwrap as no `xatlas`")
            return
        a = g.get_mesh("bunny.ply", force="mesh")

        u = a.unwrap()
        assert u.visual.uv.shape == (len(u.vertices), 2)

        checkerboard = g.np.kron([[1, 0] * 4, [0, 1] * 4] * 4, g.np.ones((10, 10)))
        try:
            from PIL import Image
        except BaseException:
            return

        image = Image.fromarray((checkerboard * 255).astype(g.np.uint8))
        u = a.unwrap(image=image)
        # make sure image was attached correctly
        assert u.visual.material.image.size == image.size


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
