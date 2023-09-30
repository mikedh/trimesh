try:
    from . import generic as g
except BaseException:
    import generic as g


class ConvexTest(g.unittest.TestCase):
    def test_convex(self):
        # store (true is_convex, mesh) tuples
        meshes = [
            (False, g.get_mesh("featuretype.STL")),
            (False, g.get_mesh("quadknot.obj")),
            (True, g.get_mesh("unit_cube.STL")),
            (False, g.get_mesh("1002_tray_bottom.STL")),
            (True, g.trimesh.creation.icosphere()),
            (True, g.trimesh.creation.uv_sphere()),
            (True, g.trimesh.creation.box()),
            (True, g.trimesh.creation.cylinder(radius=1, height=10)),
            (True, g.trimesh.creation.capsule()),
            (
                False,
                (
                    g.trimesh.creation.box(extents=(1, 1, 1))
                    + g.trimesh.creation.box(bounds=[[10, 10, 10], [12, 12, 12]])
                ),
            ),
        ]

        for is_convex, mesh in meshes:
            assert mesh.is_watertight

            # mesh should match it's true convexity
            assert mesh.is_convex == is_convex

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
            close = g.np.isclose(
                volume, g.np.median(volume), atol=mesh.bounding_box.volume / 1000
            )

            if g.platform.system() == "Linux":
                # on linux the convex hulls are pretty robust
                close_ok = close.all()
            else:
                # on windows sometimes there is flaky numerical weirdness
                # which causes spurious CI failures, so use softer metric
                # for success: here of 90% of values are close to the median
                # then declare everything hunky dory
                ratio = close.sum() / float(len(close))
                close_ok = ratio > 0.9

            if not close_ok:
                g.log.error(f"volume inconsistent: {volume}")
                raise ValueError(
                    "volume is inconsistent on {}".format(mesh.metadata["file_name"])
                )
            assert min(volume) > 0.0

            if not all(i.is_winding_consistent for i in hulls):
                raise ValueError(
                    "mesh %s reported bad winding on convex hull!",
                    mesh.metadata["file_name"],
                )

            if not all(i.is_convex for i in hulls):
                raise ValueError(
                    "mesh %s reported non-convex convex hull!", mesh.metadata["file_name"]
                )

    def test_primitives(self):
        for prim in [
            g.trimesh.primitives.Sphere(),
            g.trimesh.primitives.Cylinder(),
            g.trimesh.primitives.Box(),
        ]:
            assert prim.is_convex
            # convex things should have hulls of the same volume
            # convert to mesh to get tessellated volume rather than
            # analytic primitive volume
            tess = prim.to_mesh()
            assert g.np.isclose(tess.convex_hull.volume, tess.volume)

    def test_projections(self):
        # check the vertex projection onto adjacent face plane
        # this is used to calculate convexity
        for m in g.get_meshes(4):
            assert len(m.face_adjacency_projections) == (len(m.face_adjacency))

    def test_truth(self):
        # check a non-watertight mesh
        m = g.get_mesh("not_convex.obj")
        assert not m.is_convex


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
