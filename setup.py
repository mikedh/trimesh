#!/usr/bin/env python

import os
import sys
from setuptools import setup

# load __version__ without importing anything
_version_file = os.path.join(
    os.path.dirname(__file__),
    'trimesh', 'version.py')

if os.path.isfile(_version_file):
    with open(_version_file, 'r') as f:
        _version_raw = f.read()
    # use eval to get a clean string of version from file
    __version__ = eval(next(
        line.strip().split('=')[-1]
        for line in str.splitlines(_version_raw)
        if '_version_' in line))
else:
    __version__ = None

# load README.md as long_description
long_description = ''
if os.path.exists('README.md'):
    with open('README.md', 'r') as f:
        long_description = f.read()

# minimal requirements for installing trimesh
# note that `pip` requires setuptools itself
requirements_default = set(['numpy'])

# "easy" requirements should install without compiling
# anything on Windows, Linux, and Mac, for Python 2.7-3.4+
requirements_easy = set([
    'scipy',     # provide convex hulls, fast graph ops, etc
    'networkx',  # provide slow graph ops with a nice API
    'lxml',      # handle XML better and faster than built- in XML
    'pyglet<2',  # render preview windows nicely : note pyglet 2.0 is basically a re-write
    'shapely',   # handle 2D polygons robustly
    'rtree',     # create N-dimension trees for broad-phase queries
    'svg.path',  # handle SVG format path strings
    'sympy',     # do analytical math
    'msgpack',   # serialize into msgpack
    'pillow',    # load images
    'requests',  # do network requests
    'xxhash',    # hash ndarrays faster than built-in MD5/CRC
    'setuptools',  # do setuptools stuff
    'jsonschema',  # validate JSON schemas like GLTF
    'pycollada',   # parse collada/dae/zae files
    'chardet',     # figure out if someone used UTF-16
    'mapbox-earcut',  # fast 2D triangulations of polygons
    'colorlog'])   # log in pretty colors

# "all" requirements only need to be installable
# through some mechanism on Linux with Python 3.5+
# and are allowed to compile code
requirements_all = requirements_easy.union([
    'python-fcl',    # do fast 3D collision queries
    'psutil',        # figure out how much memory we have
    'glooey',        # make GUI applications with 3D stuff
    'meshio',        # load a number of additional mesh formats; Python 3.5+
    'scikit-image',  # marching cubes and other nice stuff
    'xatlas',        # texture unwrapping
])
# requirements for running unit tests
requirements_test = set(['pytest',       # run all unit tests
                         'pytest-cov',   # coverage plugin
                         'pyinstrument',  # profile code
                         'coveralls',    # report coverage stats
                         'ezdxf'])       # use as a validator for exports

# Python 2.7 and 3.4 support has been dropped from packages
# version lock those packages here so install succeeds
current = (sys.version_info.major, sys.version_info.minor)
# packages that no longer support old Python
# setuptools-scm is required by sympy-mpmath chain
# and will hopefully be removed in future versions
lock = [((3, 4), 'lxml', '4.3.5'),
        ((3, 4), 'shapely', '1.6.4'),
        ((3, 4), 'pyglet', '1.4.10'),
        ((3, 5), 'sympy', None),
        ((3, 0), 'pyglet<2', None),
        ((3, 6), 'svg.path', '4.1')]
for max_python, name, version in lock:
    if current <= max_python:
        # remove version-free requirements
        requirements_easy.discard(name)
        # if version is None drop that package
        if version is not None:
            # add working version locked requirements
            requirements_easy.add('{}=={}'.format(name, version))


def format_all():
    """
    A shortcut to run automatic formatting and complaining
    on all of the trimesh subdirectories.
    """
    import subprocess

    def run_on(target):
        # words that codespell hates
        # note that it always checks against the lower case
        word_skip = "datas,coo,nd,files',filetests,ba,childs,whats"
        # files to skip spelling on
        file_skip = "*.pyc,*.zip,.DS_Store,*.js,./trimesh/resources"
        spell = ['codespell', '-i', '3',
                 '--skip=' + file_skip,
                 '-L', word_skip, '-w', target]
        print("Running: \n {} \n\n\n".format(' '.join(spell)))
        subprocess.check_call(spell)

        formatter = ["autopep8", "--recursive", "--verbose",
                     "--in-place", "--aggressive", target]
        print("Running: \n {} \n\n\n".format(
            ' '.join(formatter)))
        subprocess.check_call(formatter)

        flake = ['flake8', target]
        print("Running: \n {} \n\n\n".format(' '.join(flake)))
        subprocess.check_call(flake)

    # run on our target locations
    for t in ['trimesh', 'tests', 'examples']:
        run_on(t)


# if someone wants to output a requirements file
# `python setup.py --list-all > requirements.txt`
if '--list-all' in sys.argv:
    # will not include default requirements (numpy)
    print('\n'.join(requirements_all))
    exit()
elif '--list-easy' in sys.argv:
    # again will not include numpy+setuptools
    print('\n'.join(requirements_easy))
    exit()
elif '--list-test' in sys.argv:
    # again will not include numpy+setuptools
    print('\n'.join(requirements_test))
    exit()
elif '--format' in sys.argv:
    format_all()
    exit()
elif '--bump' in sys.argv:
    # bump the version number
    # convert current version to integers
    bumped = [int(i) for i in __version__.split('.')]
    # increment the last field by one
    bumped[-1] += 1
    # re-combine into a version string
    version_new = '.'.join(str(i) for i in bumped)
    print('version bump `{}` => `{}`'.format(
        __version__, version_new))
    # write back the original version file with
    # just the value replaced with the new one
    raw_new = _version_raw.replace(__version__, version_new)
    with open(_version_file, 'w') as f:
        f.write(raw_new)
    exit()


# call the magical setuptools setup
setup(name='trimesh',
      version=__version__,
      description='Import, export, process, analyze and view triangular meshes.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Michael Dawson-Haggerty',
      author_email='mikedh@kerfed.com',
      license='MIT',
      url='https://github.com/mikedh/trimesh',
      keywords='graphics mesh geometry 3D',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Natural Language :: English',
          'Topic :: Scientific/Engineering'],
      packages=[
          'trimesh',
          'trimesh.ray',
          'trimesh.path',
          'trimesh.path.exchange',
          'trimesh.scene',
          'trimesh.voxel',
          'trimesh.visual',
          'trimesh.viewer',
          'trimesh.exchange',
          'trimesh.resources',
          'trimesh.interfaces'],
      package_data={'trimesh': ['resources/templates/*',
                                'resources/*.json',
                                'resources/schema/*',
                                'resources/schema/primitive/*.json',
                                'resources/*.zip']},
      install_requires=list(requirements_default),
      extras_require={'test': list(requirements_test),
                      'easy': list(requirements_easy),
                      'all': list(requirements_all)})
