"""
resolvers.py
---------------

Provides a common interface to load assets referenced by name
like MTL files, texture images, etc. Assets can be from ZIP
archives, web assets, or a local file path.
"""

import os
import itertools

from . import util

# URL parsing for remote resources via WebResolver
try:
    # Python 3
    from urllib.parse import urlparse, urljoin
except ImportError:
    # Python 2
    from urlparse import urlparse, urljoin


class Resolver(object):
    """
    The base class for resolvers.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Use a resolver subclass!')

    def write(self, name, data):
        raise NotImplementedError('write not implemented for {}'.format(
            self.__class__.__name__))

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
            # if we were passed a directory use it
            self.parent = clean
        else:
            # otherwise get the parent directory we've been passed
            split = os.path.split(clean)
            self.parent = split[0]

        # exit if directory doesn't exist
        if not os.path.isdir(self.parent):
            raise ValueError('path `{} `not a directory!'.format(
                self.parent))

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
        with open(os.path.join(self.parent, name.strip()), 'rb') as f:
            data = f.read()
        return data

    def write(self, name, data):
        """
        Write an asset to a file path.

        Parameters
        -----------
        name : str
          Name of the file to write
        data : str or bytes
          Data to write to the file
        """
        # write files to path name
        with open(os.path.join(self.parent, name.strip()), 'wb') as f:
            # handle encodings correctly for str/bytes
            util.write_encoded(file_obj=f, stuff=data)


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
        # not much we can do with None
        if name is None:
            return
        # make sure name is a string
        if hasattr(name, 'decode'):
            name = name.decode('utf-8')
        # store reference to archive inside this function
        archive = self.archive

        # requested name not identical in storage so attempt to recover
        if name not in archive:
            # loop through unique results
            for option in nearby_names(name):
                if option in archive:
                    # cleaned option is in archive so store value and exit
                    name = option
                    break

        # get the stored data
        obj = archive[name]
        # if the dict is storing data as bytes just return
        if isinstance(obj, (bytes, str)):
            return obj

        # otherwise get it as a file object
        # read file object from beginning
        obj.seek(0)
        # data is stored as a file object
        data = obj.read()
        obj.seek(0)
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

        # remove leading and trailing whitespace
        name = name.strip()
        # fetch the data from the remote url
        response = requests.get(urljoin(
            self.base_url, name))

        if response.status_code != 200:
            # try to strip off filesystem crap
            response = requests.get(urljoin(
                self.base_url,
                name.lstrip('./')))

        if response.status_code == '404':
            raise ValueError(response.content)

        # return the bytes of the response
        return response.content


def nearby_names(name):
    """
    Try to find nearby variants of a specified name.

    Parameters
    ------------
    name : str
      Initial name.

    Yields
    -----------
    nearby : str
      Name that is a lightly permutated version of initial name.
    """
    # the various operations that *might* result in a correct key
    cleaners = [lambda x: x,
                lambda x: x.strip(),
                lambda x: x.lstrip('./'),
                lambda x: x.split('/')[-1],
                lambda x: x.replace('%20', ' ')]

    # make sure we don't return repeat values
    hit = set()
    for f in cleaners:
        # try just one cleaning function
        current = f(name)
        if current in hit:
            continue
        hit.add(current)
        yield current

    for a, b in itertools.combinations(cleaners, 2):
        # apply both clean functions
        current = a(b(name))
        if current in hit:
            continue
        hit.add(current)
        yield current

        # try applying in reverse order
        current = b(a(name))
        if current in hit:
            continue
        hit.add(current)
        yield current
