try:
    from . import generic as g
except BaseException:
    import generic as g


class ResolverTest(g.unittest.TestCase):

    def test_filepath_namespace(self):
        # check the namespaced method
        models = g.dir_models
        subdir = '2D'

        # create a resolver for the models diretory
        resolver = g.trimesh.resolvers.FilePathResolver(models)

        # should be able to get an asset
        assert len(resolver.get('rabbit.obj')) > 0

        # check a few file path keys
        check = set(['ballA.off', 'featuretype.STL'])
        assert set(resolver.keys()).issuperset(check)

        # try a namespaced resolver
        ns = resolver.namespaced(subdir)
        assert not set(ns.keys()).issuperset(check)
        assert set(ns.keys()).issuperset(['tray-easy1.dxf',
                                          'single_arc.dxf'])

    def test_web_namespace(self):
        base = 'https://example.com'
        ns = 'stuff'
        # check with a trailing slash
        a = g.trimesh.resolvers.WebResolver(base + '/')
        b = a.namespaced(ns)

        # should have correct slashes
        truth = base + '/' + ns
        assert b.base_url == truth
        # should not have altered original
        assert a.base_url == base + '/'

    def test_items(self):
        # check __getitem__ and __setitem__
        archive = {}
        resolver = g.trimesh.resolvers.ZipResolver(archive)
        assert len(set(resolver.keys())) == 0
        resolver['hi'] = b'what'
        # should have one item
        assert set(resolver.keys()) == set(['hi'])
        # should have the right value
        assert resolver['hi'] == b'what'
        # original archive should have been modified
        assert set(archive.keys()) == set(['hi'])

        # add a subdirectory key
        resolver['stuff/nah'] = b'sup'
        assert set(archive.keys()) == set(['hi', 'stuff/nah'])
        assert set(resolver.keys()) == set(['hi', 'stuff/nah'])

        # try namespacing
        ns = resolver.namespaced('stuff')
        assert ns['nah'] == b'sup'
        print(ns.keys())
        assert set(ns.keys()) == set(['nah'])


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
