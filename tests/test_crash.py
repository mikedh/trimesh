try:
    from . import generic as g
except BaseException:
    import generic as g


def not_open(file_name, proc):
    """
    Assert that a file name is not open
    """
    # expand input path
    file_name = g.os.path.abspath(g.os.path.expanduser(file_name))
    # assert none of the open files are the one specified
    assert all(i.path != file_name for i in proc.open_files())


class FileTests(g.unittest.TestCase):
    def test_close(self):
        """
        Even when loaders crash, we should close files
        """
        try:
            import psutil
        except ImportError:
            g.log.warning("no psutil, exiting")
            return

        # a reference to current process
        proc = psutil.Process()

        # create a blank empty PLY file
        f = g.tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
        # close file
        f.close()

        # file shouldn't be open
        not_open(f.name, proc)
        try:
            # should crash
            g.trimesh.load(f.name)
            # shouldn't make it to here
            raise AssertionError()
        except ValueError:
            # should be raised
            pass
        # file shouldn't be open
        not_open(f.name, proc)
        try:
            # should crash
            g.trimesh.load_mesh(f.name)
            # shouldn't make it to here
            raise AssertionError()
        except ValueError:
            # should be raised
            pass
        not_open(f.name, proc)
        # clean up after test
        g.os.remove(f.name)

        # create a blank empty unsupported file
        f = g.tempfile.NamedTemporaryFile(suffix=".blorb", delete=False)
        # close file
        f.close()
        # file shouldn't be open
        not_open(f.name, proc)
        try:
            # should crash
            g.trimesh.load(f.name)
            # shouldn't make it to here
            raise AssertionError()
        except ValueError:
            # should be raised
            pass
        # file shouldn't be open
        not_open(f.name, proc)
        try:
            # should crash
            g.trimesh.load_mesh(f.name)
            # shouldn't make it to here
            raise AssertionError()
        except KeyError:
            # should be raised
            pass
        not_open(f.name, proc)
        # clean up after test
        g.os.remove(f.name)

        # create a blank empty DXF file
        f = g.tempfile.NamedTemporaryFile(suffix=".dxf", delete=False)
        # close file
        f.close()
        # file shouldn't be open
        not_open(f.name, proc)
        try:
            # should crash
            g.trimesh.load(f.name)
            # shouldn't make it to here
            raise AssertionError()
        except IndexError:
            # should be raised
            pass
        # file shouldn't be open
        not_open(f.name, proc)
        try:
            # should crash
            g.trimesh.load_path(f.name)
            # shouldn't make it to here
            raise AssertionError()
        except IndexError:
            # should be raised
            pass
        not_open(f.name, proc)
        # clean up after test
        g.os.remove(f.name)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
