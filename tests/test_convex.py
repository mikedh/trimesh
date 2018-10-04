try:
    from . import generic as g
except BaseException:
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
                    permutated = permutated.permutate.tessellation()
                hulls.append(permutated.convex_hull)

            volume = g.np.array([i.volume for i in hulls])

            # which of the volumes are close to the median volume
            close = g.np.isclose(volume,
                                 g.np.median(volume),
                                 atol=mesh.bounding_box.volume / 1000)

            if g.platform.system() == 'Linux':
                # on linux the convex hulls are pretty robust
                close_ok = close.all()
            else:
                # on windows sometimes there is flaky numerical weirdness
                # which causes spurious CI failures, so use softer metric
                # for success: here of 90% of values are close to the median
                # then declare everything hunky dory
                ratio = close.sum() / float(len(close))
                close_ok = ratio > .9

            if not close_ok:
                log.error('volume inconsistent: {}'.format(volume))
                raise ValueError('volume is inconsistent on {}'.format(
                    mesh.metadata['file_name']))
            assert volume.min() > 0.0

            if not all(i.is_winding_consistent for i in hulls):
                raise ValueError(
                    'mesh %s reported bad winding on convex hull!',
                    mesh.metadata['file_name'])

            if not all(i.is_convex for i in hulls):
                raise ValueError('mesh %s reported non-convex convex hull!',
                                 mesh.metadata['file_name'])

    def test_primitives(self):
        for prim in [g.trimesh.primitives.Sphere(),
                     g.trimesh.primitives.Cylinder(),
                     g.trimesh.primitives.Box()]:
            assert prim.is_convex
            # convex things should have hulls of the same volume
            # convert to mesh to get tesselated volume rather than
            # analytic primitive volume
            tess = prim.to_mesh()
            assert g.np.isclose(tess.convex_hull.volume,
                                tess.volume)

    def test_projections(self):
        # check the vertex projection onto adjacent face plane
        # this is used to calculate convexity
        for m in g.get_meshes(4):
            assert (len(m.face_adjacency_projections) ==
                    (len(m.face_adjacency)))


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
