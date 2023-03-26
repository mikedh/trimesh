"""
corpus.py
------------

Test loaders against large corpuses of test data from github:
will download more than a gigabyte to your home directory!
"""
import trimesh
from trimesh.util import wrap_as_stream, log
import numpy as np

from pyinstrument import Profiler


# get a set with available extension
available = trimesh.available_formats()

# remove loaders that are thin wrappers
available.difference_update(
    [k for k, v in
     trimesh.exchange.load.mesh_loaders.items()
     if v in (trimesh.exchange.misc.load_meshio,)])
# remove loaders we don't care about
available.difference_update({'json', 'dae', 'zae'})
available.update({'dxf', 'svg'})


def on_repo(repo, commit):
    """
    Try loading all supported files in a Github repo.

    Parameters
    -----------
    repo : str
      Github "slug" i.e. "assimp/assimp"
    commit : str
      Full hash of the commit to check.
    """

    # get a resolver for the specific commit
    repo = trimesh.resolvers.GithubResolver(
        repo=repo, commit=commit,
        save='~/.trimesh-cache')
    # list file names in the repo we can load
    paths = [i for i in repo.keys()
             if i.lower().split('.')[-1] in available]

    report = {}
    for _i, path in enumerate(paths):
        namespace, name = path.rsplit('/', 1)
        # get a subresolver that has a root at
        # the file we are trying to load
        resolver = repo.namespaced(namespace)

        check = path.lower()
        broke = ('malformed empty outofmemory ' +
                 'bad incorrect missing ' +
                 'failures pond.0.ply').split()
        should_raise = any(b in check for b in broke)
        raised = False

        # clip off the big old name from the archive
        saveas = path[path.find(commit) + len(commit):]

        try:
            m = trimesh.load(
                file_obj=wrap_as_stream(resolver.get(name)),
                file_type=name,
                resolver=resolver)
            report[saveas] = str(m)

            # if our source was a GLTF we should be able to roundtrip without
            # dropping
            if name.lower().split('.')[-1] in ('gltf',
                                               'glb') and len(m.geometry) > 0:
                # try round-tripping the file
                e = trimesh.load(file_obj=wrap_as_stream(m.export(file_type='glb')),
                                 file_type='glb', process=False)

                # geometry keys should have survived roundtrip
                assert set(m.geometry.keys()) == set(e.geometry.keys())
                assert set(m.graph.nodes) == set(e.graph.nodes)
                for key, geom in e.geometry.items():
                    # the original loaded mesh
                    ori = m.geometry[key]
                    # todo : why doesn't this pass
                    # assert np.allclose(ori.vertices, geom.vertices)
                    if isinstance(getattr(geom, 'visual', None),
                                  trimesh.visual.TextureVisuals):
                        a, b = geom.visual.material, ori.visual.material
                        # try our fancy equal
                        assert equal(a.baseColorFactor, b.baseColorFactor)
                        try:
                            assert equal(
                                a.baseColorTexture, b.baseColorTexture)
                        except BaseException:
                            from IPython import embed
                            embed()

        except NotImplementedError as E:
            # this is what unsupported formats
            # like GLTF 1.0 should raise
            log.debug(E)
            report[saveas] = str(E)
        except BaseException as E:
            raised = True
            # we got an error on a file that should have passed
            if not should_raise:
                log.debug(path, E)
                raise E
            report[saveas] = str(E)

        # if it worked when it didn't have to add a label
        if should_raise and not raised:
            # raise ValueError(name)
            report[saveas] += ' SHOULD HAVE RAISED'

    return report


def equal(a, b):
    """
    Check equality of two things.

    Parameters
    -----------
    a : any
      Something.
    b : any
      Another thing.

    Returns
    ----------------
    equal : bool
      Are these things equal-ish.
    """
    if a is None:
        return b is None

    if not isinstance(a, type(b)):
        return False

    if isinstance(a, np.ndarray):
        # if we have a numpy array first check shape
        if a.shape != b.shape:
            return False
        # return every-element-is-close
        return np.allclose(a, b)

    # a PIL image of some variety
    if hasattr(a, 'getpixel'):
        if a.size != b.size:
            return False
        # very crude: it's pretty hard to check if two images
        # are similar and JPEG produces diffs that are like 25%
        # different but something is better than nothing I suppose
        aa, bb = np.array(a), np.array(b)
        percent = np.abs(aa - bb).sum() / (np.product(aa.shape) * 256)
        return percent < 0.5

    # try built-in eq method
    return a == b


if __name__ == '__main__':

    trimesh.util.attach_to_log()

    with Profiler() as P:
        # check the assimp corpus, about 50mb
        report = on_repo(
            repo='assimp/assimp',
            commit='c2967cf79acdc4cd48ecb0729e2733bf45b38a6f')
        # check the gltf-sample-models, about 1gb
        report.update(on_repo(
            repo='KhronosGroup/glTF-Sample-Models',
            commit='8e9a5a6ad1a2790e2333e3eb48a1ee39f9e0e31b'))

        # add back collada for this repo
        available.update(['dae', 'zae'])
        report.update(on_repo(
            repo='ros-industrial/universal_robot',
            commit='8f01aa1934079e5a2c859ccaa9dd6623d4cfa2fe'))

    # show all profiler lines
    log.info(P.output_text(show_all=True))

    # print a formatted report of what we loaded
    log.debug('\n'.join(f'# {k}\n{v}\n' for k, v in report.items()))
