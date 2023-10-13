try:
    from . import generic as g
except BaseException:
    import generic as g


class SampleTest(g.unittest.TestCase):
    def test_sample(self):
        m = g.get_mesh("featuretype.STL")

        samples = m.sample(1000)

        # check to make sure all samples are on the mesh surface
        distance = m.nearest.signed_distance(samples)
        assert g.np.abs(distance).max() < 1e-4

        even, index = g.trimesh.sample.sample_surface_even(m, 1000)
        # check to make sure all samples are on the mesh surface
        distance = m.nearest.signed_distance(even)
        assert g.np.abs(distance).max() < 1e-4

    def test_weights(self):
        m = g.trimesh.creation.box()

        # weigh all faces except first as zero
        weights = g.np.zeros(len(m.faces))
        weights[0] = 1.0

        # sample with passed weights
        points, fid = m.sample(count=100, return_index=True, face_weight=weights)
        # all faces should be on single face
        assert (fid == 0).all()

        # oversample box to make sure weights aren't screwing
        # up ability to get every face when weighted by area
        assert set(g.np.unique(m.sample(100000, return_index=True)[1])) == set(
            range(len(m.faces))
        )

    def test_color(self):
        # check to see if sampling by color works

        # sample a textured mesh
        m = g.get_mesh("fuze.obj")
        points, index, color = g.trimesh.sample.sample_surface(m, 100, sample_color=True)
        assert len(points) == len(color)

        # sample a color mesh
        m = g.get_mesh("machinist.XAML")
        assert m.visual.kind == "face"
        points, index, color = g.trimesh.sample.sample_surface(m, 100, sample_color=True)
        assert len(points) == len(color)

    def test_sample_volume(self):
        m = g.trimesh.creation.icosphere()
        samples = g.trimesh.sample.volume_mesh(mesh=m, count=100)

        # all samples should be approximately within the sphere
        radii = g.np.linalg.norm(samples, axis=1)
        assert (radii < 1.00000001).all()

    def sample_volume_rectangular(self):
        # check to see if our OBB volume sampling runs
        m = g.get_mesh("rabbit.obj")
        obb = m.bounding_box_oriented

        # should use a box-specific volume sampling method
        samples = obb.sample_volume(100)
        assert samples.shape == (100, 3)

    def test_deterministic_sample(self):
        m = g.get_mesh("featuretype.STL")

        # Without seed passed should return non-deterministic results
        even_first, index_first = g.trimesh.sample.sample_surface(m, 10000)
        even_last, index_last = g.trimesh.sample.sample_surface(m, 10000)
        assert not (even_first == even_last).all()
        assert not (index_first == index_last).all()

        # With seed passed should return identical results
        even_first, index_first = g.trimesh.sample.sample_surface(m, 10000, seed=10)
        even_last, index_last = g.trimesh.sample.sample_surface(m, 10000, seed=10)
        assert (even_first == even_last).all()
        assert (index_first == index_last).all()


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
