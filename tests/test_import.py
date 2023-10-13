"""
test_import.py
-------------------

Make sure trimesh is importing right.
"""
import unittest


class ImportTests(unittest.TestCase):
    def test_path(self):
        import os

        # make sure trimesh imports without any environment variables
        # this was failing on `PATH` at some point
        keys = list(os.environ.keys())
        path = {k: os.environ.pop(k) for k in keys}

        # this should succeed with nothing in PATH
        import trimesh

        # not sure if this is necessary but put back removed values
        for key, value in path.items():
            os.environ[key] = value

        # do a very basic operation
        assert trimesh.creation.icosphere().is_volume


if __name__ == "__main__":
    unittest.main()
