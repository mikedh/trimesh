"""
fuzz.py
-------

Atheris harness for every loader including archives, used by
OSS-Fuzz and `make fuzz`. The first input byte selects `file_type`
and the rest is the file, which is how `-dump` writes seeds.
"""

import argparse
import hashlib
import io
import os
import random
import sys

import atheris
import corpus
import numpy as np

import trimesh
from trimesh.exchange.load import mesh_loaders
from trimesh.exchange.misc import load_meshio
from trimesh.util import log

# every loader in this environment except meshio
FORMATS = sorted(
    trimesh.available_formats() - {k for k, v in mesh_loaders.items() if v is load_meshio}
)


def TestOneInput(data: bytes) -> None:
    if len(data) < 2:
        return
    file_type = FORMATS[data[0] % len(FORMATS)]
    try:
        trimesh.load_scene(io.BytesIO(data[1:]), file_type=file_type)
    except (RecursionError, MemoryError, SystemError):
        # unbounded nesting, huge allocations or a broken C extension
        raise
    except Exception:
        # any loader raising on malformed input is fine
        pass


def dump(
    path: str,
    count: int | None = None,
    offset: float | None = None,
    max_size: int = 1000000,
    local: str | None = None,
) -> None:
    """
    Write a seed corpus for `TestOneInput` into a directory.

    Parameters
    -----------
    path
      Directory to write seeds into.
    count
      If passed take a systematic sample of this many files: `count`
      evenly spaced points in `[0, 1)` shifted by `offset` index into
      every corpus' sorted file names laid end to end.
    offset
      Where in `[0, 1)` the evenly spaced sample points start,
      if not passed a random one is picked and logged.
    max_size
      Skip any file larger than this many bytes.
    local
      If passed read every file under this directory rather than `corpus.REPOS`.
    """
    if local is None:
        sources = [
            (
                trimesh.resolvers.GithubResolver(
                    repo=repo, commit=commit, save="~/.trimesh-cache"
                ),
                root,
            )
            for repo, commit, root in corpus.REPOS
        ]
    else:
        sources = [(trimesh.resolvers.FilePathResolver(local), None)]

    # sorted loadable file names under the root of each corpus, end to end
    files = [
        (resolver, name)
        for resolver, root in sources
        for name in sorted(
            n
            for n in resolver.keys()
            if n.startswith(root or "")
            and trimesh.util.split_extension(n.lower()) in FORMATS
        )
    ]
    count = len(files) if count is None else min(count, len(files))
    if offset is None:
        offset = random.random()
        log.info(f"sampling {count}/{len(files)} files at `offset={offset}`")
    # shift by `offset` in whole files then space `count` picks evenly using
    # integer math so picks are at least 1 apart and stay under `len(files)`
    shift = min(int(offset * len(files)), len(files) - 1)
    index = (np.arange(count) * len(files) + shift) // count

    os.makedirs(path, exist_ok=True)
    for i in index:
        resolver, name = files[i]
        file_type = trimesh.util.split_extension(name.lower())
        data = bytes([FORMATS.index(file_type)]) + resolver.get(name)
        if len(data) > max_size:
            continue
        # name seeds by content hash like libFuzzer does
        with open(os.path.join(path, hashlib.sha1(data).hexdigest()), "wb") as f:
            f.write(data)


def main():
    # our flags, with everything else passed through to libFuzzer
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("-dump", type=str, help="Write a seed corpus to this directory")
    parser.add_argument("-local", type=str, help="Seed from this directory not Github")
    parser.add_argument("-count", type=int, help="Seed an evenly spaced sample")
    args, rest = parser.parse_known_args()

    if args.dump:
        trimesh.util.attach_to_log()
        dump(path=args.dump, count=args.count, local=args.local)
        return

    atheris.instrument_all()
    atheris.Setup([sys.argv[0], *rest], TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
