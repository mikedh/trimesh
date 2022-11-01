import trimesh
from trimesh.util import wrap_as_stream
from pyinstrument import Profiler


# get a set with available extension
available = trimesh.available_formats()

# remove loaders that are thin wrappers
available.difference_update(
    [k for k, v in
     trimesh.exchange.load.mesh_loaders.items()
     if v in (trimesh.exchange.misc.load_meshio,
              trimesh.exchange.dae.load_collada)])
# remove loaders we don't care about
available.difference_update({'json'})
available.update(trimesh.exchange.load.path_formats())


def on_repo(repo, commit):
    """
    Try loading all supported files in a Github repo.
    """

    # get a copy of a recent commit from assimp
    repo = trimesh.resolvers.GithubResolver(
        repo=repo, commit=commit,
        save='~/.trimesh-cache')
    # list file names in the repo we can load
    paths = [i for i in repo.keys()
             if i.lower().split('.')[-1] in available]

    report = {}
    for i, path in enumerate(paths):
        namespace, name = path.rsplit('/', 1)
        # get a subresolver that has a root at
        # the file we are trying to load
        resolver = repo.namespaced(namespace)

        check = path.lower()
        broke = ('malformed empty outofmemory ' +
                 'bad incorrect missing failures pond.0.ply').split()
        should_raise = any(b in check for b in broke)
        raised = False

        # clip off the front of the report
        saveas = path[path.find('models') + 7:]

        try:
            m = trimesh.load(
                file_obj=wrap_as_stream(resolver.get(name)),
                file_type=name,
                resolver=resolver)
            report[saveas] = str(m)
        except NotImplementedError as E:
            # this is what unsupported formats
            # like GLTF 1.0 should raise
            print(E)
            report[saveas] = str(E)
        except BaseException as E:
            raised = True
            if not should_raise:
                print(path, E)
                raise E
            report[saveas] = str(E)

        # if it worked and it shouldn't have, raise
        if should_raise and not raised:
            # raise ValueError(name)
            report[saveas] += ' SHOULD HAVE RAISED'

    return report


if __name__ == '__main__':

    trimesh.util.attach_to_log()

    with Profiler() as P:
        report = on_repo(
            repo='assimp/assimp',
            commit='c2967cf79acdc4cd48ecb0729e2733bf45b38a6f')
        report.update(on_repo(
            repo='KhronosGroup/glTF-Sample-Models',
            commit='8e9a5a6ad1a2790e2333e3eb48a1ee39f9e0e31b'))
    P.print()

    print('\n'.join(f'# {k}\n{v}\n' for k, v in report.items()))
