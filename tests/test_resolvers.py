import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

try:
    from . import generic as g
except BaseException:
    import generic as g

# canned response body for the local web server
CANNED = b"hello resolver"


@pytest.fixture(scope="module")
def local_url():
    # a local server so session handling is tested end-to-end
    # against the real clients without touching the network
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if "missing" in self.path:
                self.send_error(404)
            elif "redirect" in self.path:
                # exercises the redirect kwarg each library names differently
                self.send_response(302)
                self.send_header("Location", "/models/thing.obj")
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(CANNED)))
                self.end_headers()
                self.wfile.write(CANNED)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/models/thing.obj"
    server.shutdown()


def test_filepath_allow_anywhere(tmp_path):
    # a model in `root/sub` referencing `../outside.txt`
    outside = b"outside the root"
    (tmp_path / "outside.txt").write_bytes(outside)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "inside.txt").write_bytes(CANNED)

    resolver = g.trimesh.resolvers.FilePathResolver(str(sub))
    assert resolver.get("inside.txt") == CANNED
    # escaping paths are blocked by default with an actionable message
    with pytest.raises(ValueError, match="allow_anywhere"):
        resolver.get("../outside.txt")
    with pytest.raises(ValueError, match="allow_anywhere"):
        resolver.write("../evil.txt", b"nope")

    # the flag restores the pre-5.0 behavior
    anywhere = g.trimesh.resolvers.FilePathResolver(str(sub), allow_anywhere=True)
    assert anywhere.get("../outside.txt") == outside
    anywhere.write("../written.txt", b"fine")
    assert (tmp_path / "written.txt").read_bytes() == b"fine"

    # namespaced resolvers keep the flag
    (sub / "deeper").mkdir()
    assert anywhere.namespaced("deeper").allow_anywhere
    assert not resolver.namespaced("deeper").allow_anywhere

    # a genuinely missing file must not mention the policy
    with pytest.raises(FileNotFoundError) as excinfo:
        resolver.get("nope.txt")
    assert "allow_anywhere" not in str(excinfo.value)


@pytest.mark.parametrize("library", ["httpx", "requests"])
def test_web_sessions(local_url, library):
    # every supported session flavor must fetch through the same path
    lib = pytest.importorskip(library)
    session = lib.Client() if library == "httpx" else lib.Session()

    resolver = g.trimesh.resolvers.WebResolver(local_url, session=session)
    # fetch twice — catches clients that break on connection reuse
    assert resolver.get("thing.mtl") == CANNED
    assert resolver.get("thing.mtl") == CANNED
    assert resolver.get_base() == CANNED
    # both libraries must follow redirects
    assert resolver.get("redirect.mtl") == CANNED
    # a missing asset raises the session's own http error
    with pytest.raises(Exception, match="404"):
        resolver.get("missing.mtl")


def test_web_session_duck(local_url):
    # a duck-typed session must be called with no assumed kwargs
    class Response:
        status_code = 200
        content = CANNED

        def raise_for_status(self):
            pass

    class Duck:
        def get(self, url):
            return Response()

    resolver = g.trimesh.resolvers.WebResolver(local_url, session=Duck())
    assert resolver.request_kwargs == {}
    assert resolver.get("thing.mtl") == CANNED


def test_web_session_aiohttp(local_url):
    # aiohttp sessions only construct inside a running event loop and
    # bind to it, so the resolver must reject them loud and early
    aiohttp = pytest.importorskip("aiohttp")

    async def check():
        session = aiohttp.ClientSession()
        with pytest.raises(ValueError, match="aiohttp"):
            g.trimesh.resolvers.WebResolver(local_url, session=session)
        await session.close()

    asyncio.run(check())


class ResolverTest(g.unittest.TestCase):
    def test_filepath_namespace(self):
        # check the namespaced method
        models = g.dir_models
        subdir = "2D"

        # create a resolver for the models directory
        resolver = g.trimesh.resolvers.FilePathResolver(models)

        # should be able to get an asset
        assert len(resolver.get("rabbit.obj")) > 0

        # check a few file path keys
        check = {"ballA.off", "featuretype.STL"}
        assert set(resolver.keys()).issuperset(check)

        # try a namespaced resolver
        ns = resolver.namespaced(subdir)
        assert not set(ns.keys()).issuperset(check)
        assert set(ns.keys()).issuperset(["tray-easy1.dxf", "single_arc.dxf"])

    def test_web_namespace(self):
        base = "https://example.com"
        name = "stuff"
        target = "hi.gltf"

        # check with a trailing slash
        a = g.trimesh.resolvers.WebResolver(base + "/")
        b = g.trimesh.resolvers.WebResolver(base + "//")
        c = g.trimesh.resolvers.WebResolver(base)
        d = a.namespaced(name)

        # base URL's should always be the same with one trailing slash
        assert a.base_url == b.base_url
        assert b.base_url == c.base_url
        assert c.base_url == base + "/"
        # check namespaced
        assert d.base_url == base + "/" + name + "/"

        # should have correct slashes
        truth = "/".join([base, name, target])

        assert a.base_url + name + "/" + target == truth
        assert d.base_url + target == truth

    def test_items(self):
        # check __getitem__ and __setitem__
        archive = {}
        resolver = g.trimesh.resolvers.ZipResolver(archive)
        assert len(set(resolver.keys())) == 0
        resolver["hi"] = b"what"
        # should have one item
        assert set(resolver.keys()) == {"hi"}
        # should have the right value
        assert resolver["hi"] == b"what"
        # original archive should have been modified
        assert set(archive.keys()) == {"hi"}

        # add a subdirectory key
        resolver["stuff/nah"] = b"sup"
        assert set(archive.keys()) == {"hi", "stuff/nah"}
        assert set(resolver.keys()) == {"hi", "stuff/nah"}

        # try namespacing
        ns = resolver.namespaced("stuff")
        assert ns["nah"] == b"sup"
        g.log.debug(ns.keys())
        assert set(ns.keys()) == {"nah"}


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
