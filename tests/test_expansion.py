"""
Tests that we handle "expanding inputs," i.e XML or ZIP
files that expand far beyond their size.
"""

import io
import warnings
import zipfile

import pytest

import trimesh
from trimesh import resolvers, util

EXPANSION_CHECK = b"""<?xml version="1.0"?>
<!DOCTYPE root [
 <!ENTITY ent "ent">
 <!ENTITY ent2 "&ent;&ent;&ent;&ent;&ent;&ent;&ent;&ent;&ent;&ent;">
 <!ENTITY ent3 "&ent2;&ent2;&ent2;&ent2;&ent2;&ent2;&ent2;&ent2;&ent2;&ent2;">
 <!ENTITY ent4 "&ent3;&ent3;&ent3;&ent3;&ent3;&ent3;&ent3;&ent3;&ent3;&ent3;">
]>
<root>&ent4;</root>
"""


def test_xml_parse_blocks_entity_expansion():
    # entities must not expand — if they did the serialized tree would
    # balloon by orders of magnitude over the input size
    from lxml import etree

    from trimesh.exchange.common import XML_PARSER_OPTIONS

    # mirror the parser the XML loaders build from the shared options
    root = etree.fromstring(EXPANSION_CHECK, parser=etree.XMLParser(**XML_PARSER_OPTIONS))
    # input is ~360 bytes — expanded output would be many MB
    assert len(etree.tostring(root)) < 4096


THREEMF_EXPANSION_CHECK = b"""<?xml version="1.0"?>
<!DOCTYPE model [
 <!ENTITY ent "ent">
 <!ENTITY ent2 "&ent;&ent;&ent;&ent;&ent;&ent;&ent;&ent;&ent;&ent;">
 <!ENTITY ent3 "&ent2;&ent2;&ent2;&ent2;&ent2;&ent2;&ent2;&ent2;&ent2;&ent2;">
 <!ENTITY ent4 "&ent3;&ent3;&ent3;&ent3;&ent3;&ent3;&ent3;&ent3;&ent3;&ent3;">
]>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <resources/>
  <build><name>&ent4;</name></build>
</model>
"""


def test_threemf_iterparse_blocks_entity_expansion():
    # exercise the same iterparse path the 3mf loader uses on a 3mf-wrapped
    # entity-expansion payload — entities must remain unexpanded
    from lxml import etree

    from trimesh.exchange.common import XML_PARSER_OPTIONS

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("3D/3dmodel.model", THREEMF_EXPANSION_CHECK)
    buf.seek(0)
    with zipfile.ZipFile(buf) as z:
        model_bytes = io.BytesIO(z.read("3D/3dmodel.model"))

    # mirror the hardened iterparse the 3mf loader uses
    name_el = next(
        el
        for _, el in etree.iterparse(model_bytes, events=("end",), **XML_PARSER_OPTIONS)
        if el.tag.endswith("name")
    )
    # if entities had expanded, serialized form would be ~10^4 `ent` chars
    # (many KB); preserved-as-Entity form stays roughly the input element size
    serialized = etree.tostring(name_el)
    assert len(serialized) < 256
    # and `name_el.text` is None — the entity ref is a child Entity node,
    # not flattened to text content
    assert name_el.text is None


def test_filepath_resolver_traversal_rejected(tmp_path):
    # a mesh referencing `../../../etc/passwd` must not escape root
    root = tmp_path / "assets"
    root.mkdir()
    (root / "real.txt").write_bytes(b"ok")
    resolver = resolvers.FilePathResolver(str(root / "real.txt"))
    with pytest.raises(FileNotFoundError):
        resolver.get("../../../etc/passwd")
    # legitimate name in-root still works — the hardening did not break the
    # happy path
    assert resolver.get("real.txt") == b"ok"


def test_filepath_resolver_write_traversal_rejected(tmp_path):
    # same check for .write() — must not write outside the resolver root
    root = tmp_path / "assets"
    root.mkdir()
    resolver = resolvers.FilePathResolver(str(root / "anchor.txt"))
    with pytest.raises(ValueError, match="escapes resolver root"):
        resolver.write("../escaped.txt", b"data")
    assert not (tmp_path / "escaped.txt").exists()
    # legitimate in-root write still works
    resolver.write("ok.txt", b"safe")
    assert (root / "ok.txt").read_bytes() == b"safe"


def test_decompress_rejects_oversize_archive(monkeypatch):
    # patch the cap down so we don't have to craft a real 512 MB archive
    monkeypatch.setattr(util, "MAX_ARCHIVE_SIZE", 1024)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("big.bin", b"\x00" * (2 * 1024 * 1024))
    buf.seek(0)
    with pytest.raises(ValueError, match="size cap"):
        util.decompress(buf, file_type="zip")


def test_decompress_zip_cap_is_cumulative(monkeypatch):
    # F2 — running byte total spans members. each individual member fits
    # under the cap, but the sum of several does not. the cap must fire
    # mid-archive rather than per-member.
    monkeypatch.setattr(util, "MAX_ARCHIVE_SIZE", 2048)
    buf = io.BytesIO()
    payload = b"\x00" * 1024
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for i in range(4):
            z.writestr(f"chunk_{i}.bin", payload)
    buf.seek(0)
    with pytest.raises(ValueError, match="size cap"):
        util.decompress(buf, file_type="zip")


def test_gltf_base64_oversize_rejected(monkeypatch):
    # F1 — `_uri_to_bytes` decodes `data:…;base64,…` URIs; the encoded
    # payload size must be capped before allocation. patch the cap low so
    # we don't have to allocate hundreds of MB to exercise the guard.
    from trimesh.exchange.gltf import _uri_to_bytes

    monkeypatch.setattr(util, "MAX_ARCHIVE_SIZE", 1024)
    # build a base64 payload longer than the encoded-length threshold
    # (MAX_ARCHIVE_SIZE * 4 // 3 + 4)
    oversize = "A" * (1024 * 4 // 3 + 64)
    uri = "data:application/octet-stream;base64," + oversize
    with pytest.raises(ValueError, match="exceeds size cap"):
        _uri_to_bytes(uri, resolver=None)


def test_obj_image_metadata_records_reference(tmp_path):
    # F6 — the resolver root check is the actual traversal control, so the
    # `Image.info` metadata just records the texture reference exactly as the
    # MTL wrote it. the texture is referenced as `../../../tex.png` but a real
    # `tex.png` lives in-root, so the resolver's basename fallback still loads
    # it — and the recorded reference is the raw `../../../tex.png`.
    PIL = pytest.importorskip("PIL.Image")

    root = tmp_path / "obj"
    root.mkdir()
    # 1x1 white PNG — smallest texture Pillow decodes reliably
    PIL.new("RGB", (1, 1), (255, 255, 255)).save(root / "tex.png")
    (root / "m.mtl").write_text(
        "newmtl mat\nKa 1 1 1\nKd 1 1 1\nmap_Kd ../../../tex.png\n"
    )
    (root / "m.obj").write_text(
        "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl mat\nf 1 2 3\n"
    )
    mesh = trimesh.load(str(root / "m.obj"), process=False)
    image = mesh.visual.material.image
    # the material was actually loaded — not a vacuous pass
    assert image is not None
    # the reference is recorded verbatim; the resolver guarded the actual read
    assert image.info["file_path"] == "../../../tex.png"


def test_load_rejects_non_http_url():
    # A2 — `file://` and friends must not slip through the URL gate;
    # urlparse().scheme replaces the substring check, so file:// falls
    # through to the "string is not a file" path rather than being fetched
    with pytest.raises(ValueError, match="not a file"):
        trimesh.load("file:///etc/passwd", allow_remote=True)


def test_webresolver_rejects_disallowed_scheme():
    # A1 — explicit SSRF guard at construction time; scheme check fires
    # before the session-deprecation warning so no warning leaks
    with pytest.raises(ValueError, match=r"'file' not in \('http', 'https'\)"):
        resolvers.WebResolver("file:///etc/passwd")


def test_webresolver_warns_when_no_session():
    # callers should pass their own session — the implicit httpx.get path
    # is deprecated. uses an http URL since file:// would be rejected before
    # the deprecation warning has a chance to fire.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolver = resolvers.WebResolver("http://example.com/foo.glb")
    deprecations = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning) and "session" in str(w.message)
    ]
    assert len(deprecations) == 1

    # passing a session silences the warning
    class DummySession:
        def get(self, *a, **kw):
            raise RuntimeError("not called")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolvers.WebResolver("http://example.com/foo.glb", session=DummySession())
    assert not any(
        issubclass(w.category, DeprecationWarning) and "session" in str(w.message)
        for w in caught
    )
    # construction succeeded both ways
    assert resolver.url == "http://example.com/foo.glb"


def test_eval_cached_emits_deprecation():
    # D1 — flagged but kept around for one more release; behavior preserved
    mesh = trimesh.creation.box()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mesh.eval_cached("1 + 1")
    assert result == 2
    deprecations = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning) and "eval_cached" in str(w.message)
    ]
    assert len(deprecations) == 1


def test_meshscript_argv_no_split_injection(monkeypatch, tmp_path):
    # H4 — a substituted value containing whitespace/flags must NOT yield extra
    # argv tokens. drive the real MeshScript.run tokenization and capture the
    # argv it hands to the subprocess instead of re-implementing the pattern.
    from trimesh.interfaces import generic

    script = generic.MeshScript(meshes=[], script="", exchange="stl")
    bad_path = str(tmp_path / "a b --evil-flag.stl")
    script.replacement = {"MESH_PRE": bad_path, "SCRIPT": "x"}

    captured = {}

    class StopRun(Exception):
        pass

    def fake_check_output(argv, *a, **kw):
        # capture argv and bail before the real binary / mesh load runs
        captured["argv"] = argv
        raise StopRun

    monkeypatch.setattr(generic, "check_output", fake_check_output)
    with pytest.raises(StopRun):
        script.run("$MESH_PRE $SCRIPT")
    # whitespace and flag-like text in the substituted path stayed one token
    assert captured["argv"] == [bad_path, "x"]
