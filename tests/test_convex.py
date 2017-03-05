import generic as g


class ConvexTest(g.unittest.TestCase):

    def test_convex(self):
        for mesh in g.get_meshes(10):
            hulls = []
            for i in range(50):
                permutated = mesh.permutate.transform()
                if i % 10 == 0:
                    permutated = permutated.permutate.tesselation()
                hulls.append(permutated.convex_hull)

            volume = g.np.array([i.volume for i in hulls])
            self.assertTrue(volume.ptp() < (mesh.scale / 10000))
            self.assertTrue(volume.min() > 0.0)

            if not all(i.is_winding_consistent for i in hulls):
                raise ValueError('mesh %s reported bad winding on convex hull!',
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


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
