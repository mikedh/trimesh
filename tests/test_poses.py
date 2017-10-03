import generic as g


class PosesTest(g.unittest.TestCase):

    def test_nonsampling_poses(self):
        mesh = g.trimesh.creation.icosahedron()

        # Compute the stable poses of the icosahedron
        trans, probs = mesh.compute_stable_poses()

        # Probabilities should all be 0.05 (20 faces)
        self.assertTrue(g.np.allclose(g.np.array(probs) - 0.05, 0.0))
        self.assertTrue(len(trans) == 20)
        self.assertTrue(len(probs) == 20)

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
