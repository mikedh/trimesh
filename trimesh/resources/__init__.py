from pkg_resources import resource_string


def get_resource(name):
    result = resource_string('trimesh',
                             'resources/' + name)
    if hasattr(result, 'decode'):
        return result.decode('utf-8')
    return result
