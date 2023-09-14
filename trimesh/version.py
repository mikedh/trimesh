# get the version trimesh was installed with from metadata
try:
    # Python >= 3.8
    from importlib.metadata import version
    __version__ = version('trimesh')
except BaseException:
    # Python < 3.8
    from pkg_resources import get_distribution
    __version__ = get_distribution('trimesh').version

if __name__ == '__main__':
    # print version if run directly i.e. in a CI script
    print(__version__)
