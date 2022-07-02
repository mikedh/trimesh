try:
    from . import generic as g
except BaseException:
    import generic as g


class SampleTest(g.unittest.TestCase):

    def test_sample(self):
        m = g.get_mesh('featuretype.STL')

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
        points, fid = m.sample(count=100,
                               return_index=True,
                               face_weight=weights)
        # all faces should be on single face
        assert (fid == 0).all()

        # oversample box to make sure weights aren't screwing
        # up ability to get every face when weighted by area
        assert set(g.np.unique(m.sample(
            100000, return_index=True)[1])) == set(range(len(m.faces)))


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
