import io
import os
import tempfile
from types import SimpleNamespace
from unittest import mock

try:
    import generic as g
except BaseException:
    from . import generic as g


def test_split_extension_cad_zst():
    # multipart CAD extensions in any case, which need a dot separator
    cases = {
        "model.step.zst": "step.zst",
        "MODEL.STEP.ZST": "step.zst",
        "model.step.zstd": "step.zstd",
        "MODEL.STEP.ZSTD": "step.zstd",
        "part.stp.zst": "stp.zst",
        "part.stp.zstd": "stp.zstd",
        "part.iges.zst": "iges.zst",
        "part.iges.zstd": "iges.zstd",
        "part.IGS.ZST": "igs.zst",
        "part.IGS.ZSTD": "igs.zstd",
        "archive.tar.gz": "tar.gz",
        "archive.TAR.BZ2": "tar.bz2",
        "my_step.zst": "zst",
        "my_iges.zstd": "zstd",
        "step.zst": "step.zst",
        "iges.zstd": "iges.zstd",
    }
    for name, truth in cases.items():
        assert g.trimesh.util.split_extension(name) == truth


def test_legacy_step_to_glb():
    # cascadio<0.2 fallback: delete with its deprecation
    # a cascadio with only `step_to_glb` converts through temporary files
    from trimesh.exchange import cascade

    calls = []

    def step_to_glb(in_path, out_path, **kwargs):
        with open(in_path, "rb") as f:
            calls.append((f.read(), kwargs))
        with open(out_path, "wb") as f:
            f.write(b"LEGACY_GLB")

    def load_glb(file_obj, **kwargs):
        return {"glb": file_obj.read(), "kwargs": kwargs}

    with (
        mock.patch.object(cascade, "cascadio", SimpleNamespace(step_to_glb=step_to_glb)),
        mock.patch.object(cascade, "load_glb", load_glb),
    ):
        loaded = cascade.load_step(
            io.BytesIO(b"OLDSTEP"), file_type="step", tol_linear=0.02, timeout=None
        )
    # only the options it takes, and nothing it doesn't leaks to `load_glb`
    assert calls == [(b"OLDSTEP", {"tol_linear": 0.02, "merge_primitives": True})]
    assert loaded == {"glb": b"LEGACY_GLB", "kwargs": {"merge_primitives": True}}


def test_basic():
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


def test_merge_primitives():
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


def test_in_memory_and_options():
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
        with g.pytest.raises(ValueError):
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
            assert g.np.allclose(yup_scene.extents, [0.127, 0.034925, 0.0635], atol=1e-4)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_basic()
