try:
    from . import generic as g
except BaseException:
    import generic as g


class IdentifierTest(g.unittest.TestCase):
    def test_identifier(self, count=25):
        meshes = g.np.append(
            list(g.get_meshes(only_watertight=True, split=True, min_volume=0.001)),
            g.get_mesh("fixed_top.ply"),
        )
        for mesh in meshes:
            if not mesh.is_volume or mesh.body_count != 1:
                g.log.warning("Mesh %s is not watertight!", mesh.metadata["file_name"])
                continue

            g.log.info("Trying hash at %d random transforms", count)
            hashed = []
            identifier = []
            for transform in g.random_transforms(count):
                permutated = mesh.copy().apply_transform(transform)
                hashed.append(permutated.identifier_hash)
                identifier.append(permutated.identifier)

            result = g.np.array(hashed)
            ok = (result[0] == result[1:]).all()

            if not ok:
                ptp = g.np.ptp(identifier, axis=0)
                g.log.error(
                    "Hashes on %s differ after transform:\n %s\n",
                    mesh.metadata["file_name"],
                    str(ptp),
                )
                raise ValueError("values differ after transform!")

            # stretch the mesh by a small amount
            stretched = mesh.copy().apply_scale([0.99974507, 0.9995662, 1.0029832])
            if hashed[-1] == stretched.identifier_hash:
                raise ValueError(
                    "Hashes on %s didn't change after stretching",
                    mesh.metadata["file_name"],
                )

    def test_scene_id(self):
        """
        A scene has a nicely constructed transform tree, so
        make sure transforming meshes around it doesn't change
        the nuts of their identifier hash.
        """
        scenes = [g.get_mesh("cycloidal.3DXML")]

        for s in scenes:
            for geom_name, mesh in s.geometry.items():
                meshes = []
                if not mesh.is_volume:
                    continue

                for node in s.graph.nodes_geometry:
                    T, geo = s.graph[node]
                    if geom_name != geo:
                        continue

                    m = mesh.copy()
                    m.apply_transform(T)
                    meshes.append(m)
                if not all(
                    meshes[0].identifier_hash == i.identifier_hash for i in meshes
                ):
                    raise ValueError(f"{geom_name} differs after transform!")

        # check an example for a mirrored part
        assert (
            scenes[0].geometry["disc_cam_B"].identifier_hash
            != scenes[0].geometry["disc_cam_A"].identifier_hash
        )

    def test_reflection(self):
        # identifier should detect mirroring
        a = g.get_mesh("featuretype.STL")
        b = a.copy()
        b.vertices[:, 2] *= -1.0
        b.invert()
        assert g.np.isclose(a.volume, b.volume)
        # hash should differ
        assert a.identifier_hash != b.identifier_hash

        # a mesh which is quite sensitive to mirroring
        a = g.get_mesh("mirror.ply")
        b = a.copy()
        b.vertices[:, 2] *= -1.0
        b.invert()
        assert g.np.isclose(a.volume, b.volume)
        # hash should differ
        assert a.identifier_hash != b.identifier_hash

    def test_duplicates(self):
        def clean_name(name):
            return name.split("_", 1)[0].split("#", 1)[0]

        # a scene with instances
        s = g.get_mesh("cycloidal.3DXML")
        # a flat scene dump
        d = g.trimesh.Scene(s.dump())

        # should have the same number of geometry nodes
        assert len(s.graph.nodes_geometry) == len(d.graph.nodes_geometry)

        # uses the identifier to calculate
        a = s.duplicate_nodes
        b = d.duplicate_nodes

        # should be the same in both forms
        assert len(a) == len(b)

        a_set = {tuple(sorted([clean_name(i) for i in group])) for group in a}
        b_set = {tuple(sorted([clean_name(i) for i in group])) for group in b}
        assert a_set == b_set

        ptp = []
        for group in b:
            current = []
            for node in group:
                _, geom = d.graph[node]
                current.append(d.geometry[geom].identifier)
            ptp.append(g.np.ptp(current, axis=0))
            assert (ptp[-1] < 0.01).all()


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
