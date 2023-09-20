"""
# version.py

Get the current version from package metadata or pyproject.toml
if everything else fails.
"""


def _get_version():
    """
    Try all our methods to get the version.
    """
    for method in [_importlib, _pkgresources, _pyproject]:
        try:
            return method()
        except BaseException:
            pass
    return None


def _importlib() -> str:
    """
    Get the version string using package metadata on Python >= 3.8
    """

    from importlib.metadata import version

    return version("trimesh")


def _pkgresources() -> str:
    """
    Get the version string using package metadata on Python < 3.8
    """
    from pkg_resources import get_distribution

    return get_distribution("trimesh").version


def _pyproject() -> str:
    """
    Get the version string from the pyproject.toml file.
    """
    import json
    import os

    # use a path relative to this file
    pyproject = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(os.path.expanduser(__file__))),
            "..",
            "pyproject.toml",
        )
    )
    with open(pyproject) as f:
        # json.loads cleans up the string and removes the quotes
        return next(json.loads(L.split("=")[1]) for L in f if "version" in L)


# try all our tricks
__version__ = _get_version()

if __name__ == "__main__":
    # print version if run directly i.e. in a CI script
    print(__version__)
