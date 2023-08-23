from importlib.metadata import version

# will get the version the package was installed with
__version__ = version('trimesh')

if __name__ == '__main__':
    # print version if run directly i.e. in a CI script
    print(__version__)
