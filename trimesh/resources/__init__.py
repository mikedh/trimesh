import os
import json

from .. import util
# find the current absolute path to this directory
_pwd = os.path.expanduser(
    os.path.abspath(os.path.dirname(__file__)))

# once resources are loaded cache them
_cache = {}


def get(name, decode=True, decode_json=False, unzip=False):
    """
    Get a resource from the `trimesh/resources` folder.

    Parameters
    -------------
    name : str
      File path relative to `trimesh/resources`
    decode : bool
      Whether or not to decode result as UTF-8
    decode_json : bool
      Run `json.loads` on resource if True.
    unzip : bool
      Return result as decompressed zip dict

    Returns
    -------------
    resource : str, bytes, or decoded JSON
      File data
    """
    # key by name and decode
    cache_key = (name, bool(decode), bool(decode_json))
    if cache_key in _cache:
        # return cached resource
        return _cache[cache_key]

    # get the resource using relative names
    with open(os.path.join(_pwd, name), 'rb') as f:
        resource = f.read()

    if unzip:
        return util.decompress(resource, file_type='zip')

    # make sure we return it as a string if asked
    if decode and hasattr(resource, 'decode'):
        resource = resource.decode('utf-8')

    if decode_json:
        resource = json.loads(resource)

    # store for later access
    _cache[cache_key] = resource

    return resource
