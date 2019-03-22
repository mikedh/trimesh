"""
resolvers.py
---------------

Tools to load assets referenced in meshes, like MTL files
or texture images.
"""
import os

# URL parsing for remote resources via WebResolver
try:
    # python 3
    from urllib.parse import urlparse, urljoin
except ImportError:
    # python 2
    from urlparse import urlparse, urljoin


class Resolver(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Use a resolver subclass!')

    def __getitem__(self, key):
        return self.get(key)


class FilePathResolver(Resolver):
    """
    Resolve files from a source path on the file system.
    """

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
        with open(os.path.join(self.parent,
                               name.strip()), 'rb') as f:
            data = f.read()
        return data


class ZipResolver(Resolver):
    """
    Resolve files inside a ZIP archive.
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
        # not much we can do with that
        if name is None:
            return

        # if name isn't in archive try some similar values
        if name not in self.archive:
            if hasattr(name, 'decode'):
                name = name.decode('utf-8')
            # try with cleared whitespace, split paths
            for option in [name,
                           name.lstrip('./'),
                           name.strip(),
                           name.split('/')[-1]]:
                if option in self.archive:
                    name = option
                    break
        # read file object from beginning
        self.archive[name].seek(0)
        # data is stored as a file object
        data = self.archive[name].read()

        return data


class WebResolver(Resolver):
    """
    Resolve assets from a remote URL.
    """

    def __init__(self, url):
        """
        Resolve assets from a base URL.

        Parameters
        --------------
        url : str
          Location where a mesh was stored or
          directory where mesh was stored
        """
        if hasattr(url, 'decode'):
            url = url.decode('utf-8')

        # parse string into namedtuple
        parsed = urlparse(url)

        # we want a base url where the mesh was located
        path = parsed.path
        if path[-1] != '/':
            # clip off last item
            path = '/'.join(path.split('/')[:-1]) + '/'

        # store the base url
        self.base_url = '{scheme}://{netloc}/{path}'.format(
            scheme=parsed.scheme,
            netloc=parsed.netloc,
            path=path)

    def get(self, name):
        """
        Get a resource from the remote site.

        Parameters
        -------------
        name : str
          Asset name, i.e. 'quadknot.obj.mtl'
        """
        # do import here to keep soft dependency
        import requests

        # append base url to requested name
        url = urljoin(self.base_url, name)
        # fetch the data from the remote url
        response = requests.get(url)
        # return the bytes of the response
        return response.content
