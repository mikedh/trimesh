import trimesh
import numpy as np
from trimesh.util import wrap_as_stream

if __name__ == '__main__':

    trimesh.util.attach_to_log()

    from pyinstrument import Profiler

    # get a copy of a recent commit from assimp
    repo = trimesh.resolvers.GithubResolver(
        repo='assimp/assimp',
        commit='c2967cf79acdc4cd48ecb0729e2733bf45b38a6f',
        save='~/.trimesh-cache')

    # get a set with available extension
    available = trimesh.available_formats()

    # remove loaders that are thin wrappers
    available.difference_update(
        [k for k, v in trimesh.exchange.load.mesh_loaders.items()
         if v in (trimesh.exchange.misc.load_meshio,
                  trimesh.exchange.dae.load_collada)])

    # list file names in the repo we can load
    file_paths = [
        i for i in repo.keys()
        if i.lower().split('.')[-1] in available]

    results = {}

    with Profiler() as P:
        for i, path in enumerate(file_paths):

            # print(path)

            namespace, name = path.rsplit('/', 1)
            # get a subresolver that has a root at
            # the file we are trying to load
            resolver = repo.namespaced(namespace)

            check = path.lower()
            broke = 'incorrect missing failures'.split()
            should_raise = any(b in check for b in broke)
            raised = False

            # clip off the front of the report
            saveas = path[path.find('models') + 7:]

            try:
                m = trimesh.load(
                    file_obj=wrap_as_stream(resolver.get(name)),
                    file_type=name,
                    resolver=resolver)
                results[saveas] = str(m)
            except NotImplementedError as E:
                # this is what unsupported formats
                # like GLTF 1.0 should raise
                print(E)
                results[saveas] = str(E)
            except BaseException as E:
                raised = True
                if not should_raise:
                    pass  # raise E
                results[saveas] = str(E)

            # if it worked and it shouldn't have, raise
            if should_raise and not raised:
                # raise ValueError(name)
                results[saveas] += ' SHOULD HAVE RAISED'
    P.print()

    print('\n'.join(f'{k}: {v}' for k, v in results.items()))
    
    
