from pkg_resources import resource_string

def get_template(name):
    return resource_string('trimesh',
                           'templates/' + name)
