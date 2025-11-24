"""
corpus.py
------------

Test loaders against large corpuses of test data from github:
will download more than a gigabyte to your home directory!
"""

import argparse
import json
import sys
import time
from collections import defaultdict
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

    # how long did this take
    duration: float

    # how many bytes was this file?
    file_size: float

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

    def summary(self) -> str:
        """
        Prints a nice markdown table of load results including both overall
        and per-format statistics.
        """
        # Group loads by file type and split success/failure
        by_type = defaultdict(list)
        for load in self.load:
            by_type[load.file_type].append(load)

        # Count exceptions per type
        exc_counts = defaultdict(lambda: defaultdict(int))
        for load in self.load:
            if load.exception:
                exc_counts[load.file_type][load.exception] += 1

        # Extract successful load metrics for overall stats
        successful = [load for load in self.load if load.exception is None]
        duration = np.array([load.duration for load in successful])
        size = np.array([load.file_size for load in successful])

        lines = []

        # Build exception table if there are any exceptions
        if exc_counts:
            lines.append("Exceptions\n=================\n")
            exc_rows = [
                (ftype.upper(), count, str(exc)[:70])
                for ftype in sorted(exc_counts.keys())
                for exc, count in exc_counts[ftype].items()
            ]
            lines.append(markdown_table(("Format", "Count", "Exception"), exc_rows))
            lines.append("")

        # Build main results table
        rows = []
        # Add overall row
        success = float(len(successful)) / len(self.load) if len(self.load) > 0 else 0.0
        rows.append(
            (
                "Overall",
                f"{len(successful)}/{len(self.load)} ({success * 100.0:.2f}%)",
                f"{duration.mean():.3f} ± {duration.std():.3f}",
                f"{size.mean() / 1e6:.2f} ± {size.std() / 1e6:.2f}",
            )
        )

        # Add per-format rows
        for ftype in sorted(by_type.keys()):
            loads = by_type[ftype]
            ok = [load for load in loads if load.exception is None]
            if ok:
                dur = np.array([load.duration for load in ok])
                sz = np.array([load.file_size for load in ok])
                success = float(len(ok)) / len(loads) if len(loads) > 0 else 0.0
                rows.append(
                    (
                        ftype.upper(),
                        f"{len(ok)}/{len(loads)} ({success * 100.0:.2f}%)",
                        f"{dur.mean():.3f} ± {dur.std():.3f}",
                        f"{sz.mean() / 1e6:.2f} ± {sz.std() / 1e6:.2f}",
                    )
                )

        lines.append("\nLoad Results\n=================\n")
        lines.append(markdown_table(("Format", "Loaded", "Time (s)", "Size (MB)"), rows))

        return "\n".join(lines)


def markdown_table(headers: tuple[str, ...], rows: list[tuple]) -> str:
    """
    Print a markdown-formatted table.

    Parameters
    ----------
    headers
        Column headers as a tuple of strings.
    rows
        List of tuples, where each tuple represents a row of data.

    Returns
    -------
    table
        A string containing the markdown-formatted table.
    """
    # set column widths based on the longest item in each column
    col_widths = [
        max(len(h), max(len(str(row[i])) for row in rows)) for i, h in enumerate(headers)
    ]
    # start with header row and separator row
    lines = [
        "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |",
        "| " + " | ".join("-" * w for w in col_widths) + " |",
    ]

    # extend with data rows
    lines.extend(
        "| "
        + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        + " |"
        for row in rows
    )

    return "\n".join(lines)


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

        blob = resolver.get(name)

        # start collecting data about the current load attempt
        current = LoadReport(
            file_name=name,
            duration=0.0,
            file_size=len(blob),
            file_type=trimesh.util.split_extension(name),
        )

        print(f"Attempting: {name}")

        try:
            tic = time.time()
            m = trimesh.load_scene(
                file_obj=wrap_as_stream(resolver.get(name)),
                file_type=name,
                resolver=resolver,
            )
            toc = time.time()
            # save geometry types
            current.type_geometry = tuple(
                {g.__class__.__name__ for g in m.geometry.values()}
            )
            current.duration = toc - tic
            # save the <Trimesh ...> repr
            current.repr_load = str(m)

            # if our source was a GLTF we should be able to roundtrip without
            # dropping
            if name.lower().split(".")[-1] in ("gltf", "glb") and len(m.geometry) > 0:
                # try round-tripping the file
                e = trimesh.load_scene(
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


def run(save: bool = False) -> Report:
    """
    Try to load and export every mesh we can get our hands on.

    Parameters
    -----------
    save
      If passed, save a JSON dump of the load report.
    """
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

    with Profiler() as P:
        # check against the small trimesh corpus
        loads = on_repo(
            repo="mikedh/trimesh",
            commit="76b6dd1a2f552673b3b38ffd44ce4342d4e95273",
            available=available,
            root="models",
        )

        # check the assimp corpus, about 50mb
        loads.extend(
            on_repo(
                repo="assimp/assimp",
                commit="ab28db52f022a7268ffff499cd85bbabf84c4271",
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

    if save:
        with open(f"trimesh.{trimesh.__version__}.{int(time.time())}.json", "w") as F:
            json.dump(asdict(report), F)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test trimesh loaders against large corpuses (downloads >1GB to ~/.trimesh-cache)"
    )
    parser.add_argument("-run", action="store_true", help="Run the corpus test")
    parser.add_argument("-save", action="store_true", help="Save JSON report")

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    trimesh.util.attach_to_log()

    if args.run:
        report = run(save=args.save)
        print(report.summary())
