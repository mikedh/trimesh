"""
corpus.py
------------

Test loaders against large corpuses of test data from github:
will download more than a gigabyte to your home directory!
"""

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
from pyinstrument import Profiler
from pyinstrument.renderers.jsonrenderer import JSONRenderer

import trimesh
from trimesh.typed import List, Optional, Tuple
from trimesh.util import log, wrap_as_stream


@dataclass
class LoadReport:
    # i.e. 'hi.glb'
    file_name: str

    # i.e 'glb'
    file_type: str

    # i.e. 'Scene'
    type_load: Optional[str] = None

    # what type was every geometry
    type_geometry: Optional[Tuple[str]] = None

    # what is the printed repr of the object, i.e. `<Trimesh ...>`
    repr_load: Optional[str] = None

    # if there was an exception save it here
    exception: Optional[str] = None


@dataclass
class Report:
    # what did we load
    load: list[LoadReport]

    # what version of trimesh was this produced on
    version: str

    # what was the profiler output for this run
    # a pyinstrument.renderers.JSONRenderer output
    profile: str


def on_repo(
    repo: str, commit: str, available: set, root: Optional[str] = None
) -> List[LoadReport]:
    """
    Try loading all supported files in a Github repo.

    Parameters
    -----------
    repo : str
      Github "slug" i.e. "assimp/assimp"
    commit : str
      Full hash of the commit to check.
    available
      Which `file_type` to check
    root
      If passed only consider files under this root directory.
    """

    # get a resolver for the specific commit
    repo = trimesh.resolvers.GithubResolver(
        repo=repo, commit=commit, save="~/.trimesh-cache"
    )
    # list file names in the repo we can load
    paths = [i for i in repo.keys() if i.lower().split(".")[-1] in available]

    if root is not None:
        # clip off any file not under the root path
        paths = [p for p in paths if p.startswith(root)]

    report = []
    for _i, path in enumerate(paths):
        namespace, name = path.rsplit("/", 1)
        # get a subresolver that has a root at
        # the file we are trying to load
        resolver = repo.namespaced(namespace)

        check = path.lower()
        broke = "malformed outofmemory bad incorrect missing invalid failures".split()
        should_raise = any(b in check for b in broke)
        raised = False

        # start collecting data about the current load attempt
        current = LoadReport(file_name=name, file_type=trimesh.util.split_extension(name))

        print(f"Attempting: {name}")

        try:
            m = trimesh.load(
                file_obj=wrap_as_stream(resolver.get(name)),
                file_type=name,
                resolver=resolver,
            )

            # save the load types
            current.type_load = m.__class__.__name__
            if isinstance(m, trimesh.Scene):
                # save geometry types
                current.type_geometry = tuple(
                    [g.__class__.__name__ for g in m.geometry.values()]
                )
            # save the <Trimesh ...> repr
            current.repr_load = str(m)

            # if our source was a GLTF we should be able to roundtrip without
            # dropping
            if name.lower().split(".")[-1] in ("gltf", "glb") and len(m.geometry) > 0:
                # try round-tripping the file
                e = trimesh.load(
                    file_obj=wrap_as_stream(m.export(file_type="glb")),
                    file_type="glb",
                    process=False,
                )

                # geometry keys should have survived roundtrip
                assert set(m.geometry.keys()) == set(e.geometry.keys())
                assert set(m.graph.nodes) == set(e.graph.nodes)
                for key, geom in e.geometry.items():
                    # the original loaded mesh
                    ori = m.geometry[key]
                    # todo : why doesn't this pass
                    # assert np.allclose(ori.vertices, geom.vertices)
                    if isinstance(
                        getattr(geom, "visual", None), trimesh.visual.TextureVisuals
                    ):
                        a, b = geom.visual.material, ori.visual.material
                        # try our fancy equal
                        assert equal(a.baseColorFactor, b.baseColorFactor)
                        assert equal(a.baseColorTexture, b.baseColorTexture)

        except NotImplementedError as E:
            # this is what unsupported formats
            # like GLTF 1.0 should raise
            log.debug(E)
            current.exception = str(E)
        except BaseException as E:
            raised = True
            # we got an error on a file that should have passed
            if not should_raise:
                log.debug(path, E)
                raise E
            current.exception = str(E)

        # if it worked when it didn't have to add a label
        if should_raise and not raised:
            current.exception = "PROBABLY SHOULD HAVE RAISED BUT DIDN'T!"
        report.append(current)

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
    if hasattr(a, "getpixel"):
        if a.size != b.size:
            return False
        # very crude: it's pretty hard to check if two images
        # are similar and JPEG produces diffs that are like 25%
        # different but something is better than nothing I suppose
        aa, bb = np.array(a), np.array(b)
        percent = np.abs(aa - bb).sum() / (np.prod(aa.shape) * 256)
        return percent < 0.5

    # try built-in eq method
    return a == b


if __name__ == "__main__":
    trimesh.util.attach_to_log()

    # get a set with available extension
    available = trimesh.available_formats()

    # remove meshio loaders because we're not testing meshio
    available.difference_update(
        [
            k
            for k, v in trimesh.exchange.load.mesh_loaders.items()
            if v in (trimesh.exchange.misc.load_meshio,)
        ]
    )
    """
    # remove loaders we don't care about
    available.difference_update({"json", "dae", "zae"})
    available.update({"dxf", "svg"})
    """

    with Profiler() as P:
        # check against the small trimesh corpus
        loads = on_repo(
            repo="mikedh/trimesh",
            commit="2fcb2b2ea8085d253e692ecd4f71b8f450890d51",
            available=available,
            root="models",
        )

        # check the assimp corpus, about 50mb
        loads.extend(
            on_repo(
                repo="assimp/assimp",
                commit="1e44036c363f64d57e9f799beb9f06d4d3389a87",
                available=available,
                root="test",
            )
        )
        # check the gltf-sample-models, about 1gb
        loads.extend(
            on_repo(
                repo="KhronosGroup/glTF-Sample-Models",
                commit="8e9a5a6ad1a2790e2333e3eb48a1ee39f9e0e31b",
                available=available,
            )
        )
        # try on the universal robot models
        loads.extend(
            on_repo(
                repo="ros-industrial/universal_robot",
                commit="8f01aa1934079e5a2c859ccaa9dd6623d4cfa2fe",
                available=available,
            )
        )

    # show all profiler lines
    log.info(P.output_text(show_all=True))

    # save the profile for comparison loader
    profile = P.output(JSONRenderer())

    # compose the overall report
    report = Report(load=loads, version=trimesh.__version__, profile=profile)

    with open(f"trimesh.{trimesh.__version__}.{int(time.time())}.json", "w") as F:
        json.dump(asdict(report), F)
