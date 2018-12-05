"""
resolvers.py
---------------

Tools to load assets referenced in meshes, like MTL files
or texture images.
"""
import os
import trimesh


class Resolver(object):
    def get(self, name):
        raise NotImplementedError('you need to implement a get method!')


class FilePathResolver(Resolver):

    def __init__(self, source):
        """
        Resolve files based on a source path.

        Parameters
        ------------
        source : str
          File path where mesh was loaded from
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
        """
        Get an asset.

        Parameters
        -------------
        name : str
          Name of the asset

        Returns
        ------------
        data : bytes
          Loaded data from asset
        """
        # load the file by path name
        with open(os.path.join(self.parent, name), 'rb') as f:
            data = f.read()
        return data


class ZipResolver(Resolver):
    """
    Resolve file from a ZIP archive.
    """

    def __init__(self, archive):
        """
        Resolve files inside a ZIP archive as loaded by
        trimesh.util.decompress

        Parameters
        -------------
        archive : dict
          Contains resources as file object
        """
        self.archive = archive

    def get(self, name):
        """
        Get an asset from the ZIP archive.

        Parameters
        -------------
        name : str
          Name of the asset

        Returns
        -------------
        data : bytes
          Loaded data from asset
        """
        # if name isn't in archive try some similar values
        if name not in self.archive:
            if hasattr(name, 'decode'):
                name = name.decode('utf-8')
            for option in [name, name.strip(), name.split('/')[-1]]:
                if option in self.archive:
                    name = option
                    break

        # data is stored as a file object
        data = self.archive[name].read()
        # set the file object back to beginning
        self.archive[name].seek(0)

        return data
