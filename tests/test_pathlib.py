"""
Test loading from pathlib objects.
"""
try:
    from . import generic as g
except BaseException:
    import generic as g


class PathTest(g.unittest.TestCase):

    def test_pathlib(self):
        """
        Test loading with paths passed as pathlib
        """

        try:
            import pathlib
        except ImportError:
            g.log.warning('no pathlib')
            return

        # create a pathlib object for a model that exists
        path = pathlib.Path(g.dir_models) / 'featuretype.STL'
        # load the mesh
        m = g.trimesh.load(path)
        # should be a mesh
        assert isinstance(m, g.trimesh.Trimesh)

        # will generate writeable file namey
        with g.tempfile.NamedTemporaryFile(suffix='.ply') as f:
            name = pathlib.Path(f.name)

        # should export to file from pathlib object
        m.export(file_obj=name)
        # should reload
        r = g.trimesh.load(file_obj=name)

        # mesh should be the same after exporting
        assert g.np.isclose(m.volume, r.volume)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
