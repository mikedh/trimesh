import os

from pkg_resources import resource_string


def get_resource(name):
    """
    Get a resource from the trimesh/resources folder.

    Parameters
    -------------
    name: str, location relative to trimesh/resources

    Returns
    -------------
    resource: str, string of file data
    """
    # get the resource
    resource = resource_string('trimesh',
                               os.path.join('resources', name))
    # make sure we return it as a string
    if hasattr(resource, 'decode'):
        return resource.decode('utf-8')
    return resource
