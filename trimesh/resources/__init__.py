\
import os

# find the current absolute path to this directory
_pwd = os.path.expanduser(os.path.abspath(os.path.dirname(__file__)))

# once resources are loaded cache them
_cache = {}


def get(name, decode=True):
    """
    Get a resource from the trimesh/resources folder.

    Parameters
    -------------
    name : str
      File path relative to `trimesh/resources`
    decode : bool
      Whether or not to decode result as UTF-8

    Returns
    -------------
    resource : str or bytes
      File data
    """
    # key by name and decode
    cache_key = (name, bool(decode))
    if cache_key in _cache:
        # return cached resource
        return _cache[cache_key]

    # get the resource using relative names
    with open(os.path.join(_pwd, name), 'rb') as f:
        resource = f.read()

    # make sure we return it as a string if asked
    if decode and hasattr(resource, 'decode'):
        resource = resource.decode('utf-8')
    # store for later access
    _cache[cache_key] = resource

    return resource
