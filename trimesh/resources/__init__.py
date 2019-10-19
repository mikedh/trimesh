import os
import inspect

# find the current absolute path using inspect
_pwd = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))


def get_resource(name, decode=True):
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
    # get the resource using relative names
    with open(os.path.join(_pwd, name), 'rb') as f:
        resource = f.read()

    # make sure we return it as a string if asked
    if decode and hasattr(resource, 'decode'):
        return resource.decode('utf-8')

    return resource
