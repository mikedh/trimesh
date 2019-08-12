import os

from pkg_resources import resource_string


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
    # get the resource
    resource = resource_string(
        'trimesh', os.path.join('resources', name))
    # make sure we return it as a string
    if decode and hasattr(resource, 'decode'):
        return resource.decode('utf-8')
    return resource
