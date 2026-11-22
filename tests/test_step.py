import io
import os
import tempfile
from types import SimpleNamespace
from unittest import mock

try:
    import generic as g
except BaseException:
    from . import generic as g


class STEPTests(g.unittest.TestCase):
    def test_split_extension_cad_zst(self):
        split = g.trimesh.util.split_extension
        assert split("model.step.zst") == "step.zst"
        assert split("MODEL.STEP.ZST") == "step.zst"
        assert split("model.step.zstd") == "step.zstd"
        assert split("MODEL.STEP.ZSTD") == "step.zstd"
        assert split("part.stp.zst") == "stp.zst"
        assert split("part.stp.zstd") == "stp.zstd"
        assert split("part.iges.zst") == "iges.zst"
        assert split("part.iges.zstd") == "iges.zstd"
        assert split("part.IGS.ZST") == "igs.zst"
        assert split("part.IGS.ZSTD") == "igs.zstd"
        assert split("archive.tar.gz") == "tar.gz"
        assert split("archive.TAR.BZ2") == "tar.bz2"
        # must require a dot separator (or exact match)
        assert split("my_step.zst") == "zst"
        assert split("my_iges.zstd") == "zstd"
        assert split("step.zst") == "step.zst"
        assert split("iges.zstd") == "iges.zstd"

    def test_kwargs_routing_and_legacy(self):
        from trimesh.exchange import cascade

        seen_load = []
        seen_glb = []

        def fake_load(data, **kw):
            seen_load.append((data, kw))
            return b"FAKE_GLB"

        def fake_load_glb(file_obj, **kw):
            seen_glb.append((file_obj.read(), kw))
            return {"geometry": {}}

        fake_cascadio = SimpleNamespace(load=fake_load)

        with (
            mock.patch.object(cascade, "cascadio", fake_cascadio),
            mock.patch.object(cascade, "load_glb", fake_load_glb),
        ):
            # default call omits 0.2-only options so 0.1.1 stays compatible
            cascade.load_step(
                io.BytesIO(b"STEPDATA"),
                file_type="step.zst",
                skip_materials=True,
            )
            data, kw = seen_load.pop()
            assert data == b"STEPDATA"
            assert kw == {"file_type": "step", "merge_primitives": True}
            glb_bytes, glb_kw = seen_glb.pop()
            assert glb_bytes == b"FAKE_GLB"
            assert glb_kw == {"merge_primitives": True, "skip_materials": True}

            # explicit timeout=None / memory_limit=None must be forwarded,
            # while cascadio-only kwargs must not leak to load_glb
            cascade.load_step(
                io.StringIO("IGESTEXT"),
                file_type="iges.zstd",
                tol_linear=0.05,
                tol_angular=0.25,
                tol_relative=True,
                merge_primitives=False,
                use_parallel=False,
                include_brep=True,
                brep_types=["plane", "cylinder"],
                include_materials=True,
                node_name_format="PRODUCT",
                y_up=True,
                timeout=None,
                memory_limit=None,
                ignore_broken=True,
            )
            data, kw = seen_load.pop()
            assert data == b"IGESTEXT"
            assert kw == {
                "file_type": "iges",
                "merge_primitives": False,
                "tol_linear": 0.05,
                "tol_angular": 0.25,
                "tol_relative": True,
                "use_parallel": False,
                "include_brep": True,
                "brep_types": {"plane", "cylinder"},
                "include_materials": True,
                "node_name_format": "PRODUCT",
                "y_up": True,
                "timeout": None,
                "memory_limit": None,
            }
            _, glb_kw = seen_glb.pop()
            assert glb_kw == {"merge_primitives": False, "ignore_broken": True}

            # empty GLB return (cascadio 0.1.1 error convention) raises ValueError
            empty_cascadio = SimpleNamespace(load=lambda data, **kw: b"")
            with (
                mock.patch.object(cascade, "cascadio", empty_cascadio),
                self.assertRaises(ValueError),
            ):
                cascade.load_step(io.BytesIO(b"bad"), file_type="step")

            # legacy cascadio <= 0.0.17 fallback (only has step_to_glb)
            legacy_calls = []

            def fake_step_to_glb(in_path, out_path, **kw):
                with open(in_path, "rb") as f:
                    legacy_calls.append((f.read(), kw))
                with open(out_path, "wb") as f:
                    f.write(b"LEGACY_GLB")

            legacy_cascadio = SimpleNamespace(step_to_glb=fake_step_to_glb)
            with mock.patch.object(cascade, "cascadio", legacy_cascadio):
                cascade.load_step(
                    io.BytesIO(b"OLDSTEP"),
                    file_type="step",
                    tol_linear=0.02,
                    timeout=None,
                )
                assert legacy_calls == [
                    (b"OLDSTEP", {"tol_linear": 0.02, "merge_primitives": True})
                ]
                glb_bytes, _ = seen_glb.pop()
                assert glb_bytes == b"LEGACY_GLB"

                with self.assertRaises(ValueError):
                    cascade.load_step(io.BytesIO(b"IGES"), file_type="iges")
                with self.assertRaises(ValueError):
                    cascade.load_step(io.BytesIO(b"ZST"), file_type="step.zst")
                with self.assertRaises(ValueError):
                    cascade.load_step(io.BytesIO(b"ZSTD"), file_type="iges.zstd")

    def test_basic(self):
        try:
            import cascadio  # noqa
        except BaseException:
            g.log.error("failed to get cascadio!", exc_info=True)
            return

        s = g.get_mesh("box_sides.STEP")
        assert len(s.geometry) == 10

        b = g.get_mesh("featuretype.STEP")
        assert len(b.geometry) == 1
        m = next(iter(b.geometry.values()))
        m.merge_vertices(merge_tex=True, merge_norm=True)
        assert m.is_watertight

    def test_merge_primitives(self):
        try:
            import cascadio  # noqa
        except BaseException:
            g.log.error("failed to get cascadio!", exc_info=True)
            return

        # a cube with faces in two colors is exported by cascadio
        # as one glTF mesh with two primitives that get merged
        for skip_materials in [False, True]:
            m = g.trimesh.load_mesh(
                os.path.join(g.dir_models, "two_color_cube.STEP"),
                skip_materials=skip_materials,
            )
            assert len(m.faces) == 12
            m.merge_vertices()
            assert m.is_watertight

        # two polylines in two colors are exported as two LINES
        # primitives, which have no faces to merge
        for skip_materials in [False, True]:
            s = g.trimesh.load_scene(
                os.path.join(g.dir_models, "two_color_lines.STEP"),
                skip_materials=skip_materials,
            )
            assert len(s.geometry) == 2
            assert all(isinstance(m, g.trimesh.path.Path3D) for m in s.geometry.values())

    def test_in_memory_and_options(self):
        try:
            import cascadio
        except BaseException:
            g.log.error("failed to get cascadio!", exc_info=True)
            return

        if not hasattr(cascadio, "load"):
            return

        step_path = os.path.join(g.dir_models, "featuretype.STEP")
        with open(step_path, "rb") as f:
            raw = f.read()

        # prove no TemporaryDirectory is touched during modern in-memory loading
        with mock.patch.object(
            tempfile,
            "TemporaryDirectory",
            side_effect=AssertionError("should not use temp files"),
        ):
            scene = g.trimesh.load_scene(io.BytesIO(raw), file_type="step")
            assert len(scene.geometry) == 1
            assert g.np.allclose(scene.extents, [0.127, 0.0635, 0.034925], atol=1e-4)

            # include_brep attaches TM_brep_faces via cascadio.extension
            brep_scene = g.trimesh.load_scene(
                io.BytesIO(raw), file_type="step", include_brep=True
            )
            mesh = next(iter(brep_scene.geometry.values()))
            assert len(mesh.face_attributes["brep_index"]) == len(mesh.faces)
            assert len(mesh.metadata["cascadio"]["brep_faces"]) == 96

            # invalid STEP bytes raise ValueError
            with self.assertRaises(ValueError):
                g.trimesh.load_scene(io.BytesIO(b"not a step file"), file_type="step")

            # cascadio >= 0.2: IGES + Zstandard (.iges.zstd), y_up, timeout/memory_limit=None
            if hasattr(cascadio, "DEFAULT_TIMEOUT"):
                iges_scene = g.get_mesh("minimal.iges.zstd")
                assert len(iges_scene.geometry) == 1
                iges_mesh = iges_scene.to_mesh()
                iges_mesh.merge_vertices()
                assert iges_mesh.is_watertight
                assert g.np.allclose(iges_mesh.extents, [1.0, 1.0, 1.0], atol=1e-4)
                assert g.np.isclose(iges_mesh.area, 6.0, atol=1e-4)
                assert g.np.isclose(iges_mesh.volume, 1.0, atol=1e-4)

                yup_scene = g.trimesh.load_scene(
                    io.BytesIO(raw),
                    file_type="step",
                    y_up=True,
                    timeout=None,
                    memory_limit=None,
                )
                # (x, y, z) -> (x, z, -y) swaps Y and Z extents
                assert g.np.allclose(
                    yup_scene.extents, [0.127, 0.034925, 0.0635], atol=1e-4
                )


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
