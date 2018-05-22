import generic as g


class ConvexTest(g.unittest.TestCase):

    def test_convex(self):
        for mesh in g.get_meshes(10):
            if not mesh.is_watertight:
                continue
            hulls = []
            for i in range(50):
                permutated = mesh.permutate.transform()
                if i % 10 == 0:
                    permutated = permutated.permutate.tesselation()
                hulls.append(permutated.convex_hull)

            volume = g.np.array([i.volume for i in hulls])

            if volume.ptp() > (mesh.scale / 1000):
                print(volume)
                raise ValueError('volume is inconsistent on {}'.format(
                    mesh.metadata['file_name']))
            self.assertTrue(volume.min() > 0.0)

            if not all(i.is_winding_consistent for i in hulls):
                raise ValueError(
                    'mesh %s reported bad winding on convex hull!',
                    mesh.metadata['file_name'])

            '''
            # to do: make this pass
            if not all(i.is_convex for i in hulls):
                raise ValueError('mesh %s reported non-convex convex hull!',
                                  mesh.metadata['file_name'])

            if not all(i.is_watertight for i in hulls):
                raise ValueError('mesh %s reported non-watertight hull!',
                                  mesh.metadata['file_name'])
            '''

    def test_primitives(self):
        for prim in [g.trimesh.primitives.Sphere(),
                     g.trimesh.primitives.Cylinder(),
                     g.trimesh.primitives.Box()]:
            self.assertTrue(prim.is_convex)

    def test_projections(self):
        for m in g.get_meshes(4):
            assert (len(m.face_adjacency_projections) ==
                    (len(m.face_adjacency)))


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
