from pkg_resources import resource_string

def get_template(name):
    result = resource_string('trimesh',
                             'templates/' + name)
    if hasattr(result, 'decode'):
        return result.decode('utf-8')
    return result
