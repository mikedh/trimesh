try:
    from . import generic as g
except BaseException:
    import generic as g


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
