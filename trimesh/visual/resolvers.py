"""
resolvers.py
---------------

Tools to load assets referenced in meshes,
like MTL files or texture images.
"""
import os
import trimesh


class Resolver(object):
    def get(self, name):
        raise NotImplementedError('you need to implement a get method!')


class FileSystemResolver(Resolver):

    def __init__(self, source):
        """
        Resolve files based on a source path.
        """

        # remove everything other than absolute path
        clean = os.path.expanduser(os.path.abspath(str(source)))

        if os.path.isdir(clean):
            self.parent = clean
        # get the parent directory of the
        elif os.path.isfile(clean):
            split = os.path.split(clean)
            self.parent = split[0]
        else:
            raise ValueError('path not a file or directory!')

    def get(self, name):
        with open(os.path.join(self.parent, name), 'rb') as f:
            data = f.read()
        return data


class ZipResolver(Resolver):
    """
    Resolve file from a ZIP archive.
    """

    def __init__(self, archive):
        self.archive = archive

    def get(self, name):
        return self.archive[name].read()
