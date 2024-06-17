import json
import os
import warnings

from ..typed import Dict, Stream
from ..util import decode_text, wrap_as_stream

# find the current absolute path to this directory
_pwd = os.path.expanduser(os.path.abspath(os.path.dirname(__file__)))

# once resources are loaded cache them
_cache = {}


def get(
    name: str, decode: bool = True, decode_json: bool = False, as_stream: bool = False
):
    """
    DERECATED JANUARY 2025 REPLACE WITH TYPED `get_json`, `get_string`, etc.
    """
    warnings.warn(
        "`trimesh.resources.get` is deprecated "
        + "and will be removed in January 2025: "
        + "replace with typed `trimesh.resources.get_*type*`",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return _get(name=name, decode=decode, decode_json=decode_json, as_stream=as_stream)


def _get(name: str, decode: bool, decode_json: bool, as_stream: bool):
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
    as_stream : bool
      Return as a file-like object

    Returns
    -------------
    resource : str, bytes, or decoded JSON
      File data
    """
    # key by name and decode
    cache_key = (name, bool(decode), bool(decode_json), bool(as_stream))
    cached = _cache.get(cache_key)
    if hasattr(cached, "seek"):
        cached.seek(0)
    if cached is not None:
        return cached

    # get the resource using relative names
    with open(os.path.join(_pwd, name), "rb") as f:
        resource = f.read()

    # make sure we return it as a string if asked
    if decode:
        # will decode into text if possibly
        resource = decode_text(resource)

    if decode_json:
        resource = json.loads(resource)
    elif as_stream:
        resource = wrap_as_stream(resource)

    # store for later access
    _cache[cache_key] = resource

    return resource


def get_schema(name: str) -> Dict:
    """
    Load a schema and evaluate the referenced files.

    Parameters
    ------------
    name : str
      Filename of schema.

    Returns
    ----------
    schema
      Loaded and resolved schema.
    """
    from ..resolvers import FilePathResolver
    from ..schemas import resolve

    # get a resolver for our base path
    resolver = FilePathResolver(os.path.join(_pwd, "schema", name))
    # recursively load $ref keys
    return resolve(json.loads(decode_text(resolver.get(name))), resolver=resolver)


def get_json(name: str) -> Dict:
    """
    Get a resource from the `trimesh/resources` folder as a decoded string.

    Parameters
    -------------
    name : str
      File path relative to `trimesh/resources`

    Returns
    -------------
    resource
      File data decoded from JSON.
    """
    return _get(name, decode=True, decode_json=True, as_stream=False)


def get_string(name: str) -> str:
    """
    Get a resource from the `trimesh/resources` folder as a decoded string.

    Parameters
    -------------
    name : str
      File path relative to `trimesh/resources`

    Returns
    -------------
    resource
      File data as a string.
    """
    return _get(name, decode=True, decode_json=False, as_stream=False)


def get_bytes(name: str) -> bytes:
    """
    Get a resource from the `trimesh/resources` folder as binary data.

    Parameters
    -------------
    name : str
      File path relative to `trimesh/resources`

    Returns
    -------------
    resource
      File data as raw bytes.
    """
    return _get(name, decode=False, decode_json=False, as_stream=False)


def get_stream(name: str) -> Stream:
    """
    Get a resource from the `trimesh/resources` folder as a binary stream.

    Parameters
    -------------
    name : str
      File path relative to `trimesh/resources`

    Returns
    -------------
    resource
      File data as a binary stream.
    """

    return _get(name, decode=False, decode_json=False, as_stream=True)
