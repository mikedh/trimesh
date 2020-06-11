try:
    from . import generic as g
except BaseException:
    import generic as g


class ConvexTest(g.unittest.TestCase):

    def test_convex(self):
        for name in ['featuretype.STL',
                     'quadknot.obj',
                     'unit_cube.STL',
                     '1002_tray_bottom.STL']:

            mesh = g.get_mesh(name)
            assert mesh.is_watertight

            hulls = []
            volume = []

            # transform the mesh around space and check volume
            # use repeatable transforms to avoid spurious failures
            for T in g.transforms[::2]:
                permutated = mesh.copy()
                permutated.apply_transform(T)
                hulls.append(permutated.convex_hull)
                volume.append(hulls[-1].volume)

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
                g.log.error('volume inconsistent: {}'.format(volume))
                raise ValueError('volume is inconsistent on {}'.format(
                    mesh.metadata['file_name']))
            assert min(volume) > 0.0

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
            # convert to mesh to get tessellated volume rather than
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

    def test_truth(self):
        # check a non-watertight mesh
        m = g.get_mesh('not_convex.obj')
        assert not m.is_convex


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
