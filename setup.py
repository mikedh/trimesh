#!/usr/bin/env python

import os
from setuptools import setup

# load __version__ without importing anything
version_file = 'trimesh/version.py'
with open(version_file, 'r') as f:
    # use eval to get a clean string of version from file
    __version__ = eval(f.read().strip().split('=')[-1])

# load README.md as long_description
long_description = ''
if os.path.exists('README.md'):
    with open('README.md', 'r') as f:
        long_description = f.read()

# "easy" requirements should install without compiling
# anything on Windows, Linux, and Mac, for Python 2.7-3.4+
requirements_easy = set([
    'lxml',      # handle XML better and faster than built- in XML
    'pyglet',    # render preview windows nicely
    'shapely',   # handle 2D polygons robustly
    'rtree',     # create N- dimension trees for broad- phase queries
    'svg.path',  # handle SVG format path strings
    'sympy',     # do analytical math
    'msgpack',   # serialize into msgpack
    'pillow',    # load images
    'requests',  # do network requests
    'xxhash',    # hash ndarrays faster than built-in MD5/CRC
    'setuptools',  # do setuptools stuff
    'pycollada',   # parse collada/dae/zae files
    'colorlog'])   # log in pretty colors

# "all" requirements only need to be installable
# through some mechanism on Linux with Python 3.5+
# and are allowed to compile code
requirements_all = requirements_easy.union([
    'triangle',    # 2D triangulations of polygons
    'python-fcl',  # do fast 3D collision queries
    'psutil',      # figure out how much memory we have
    'glooey'])     # make GUI applications with 3D stuff

# call the magical setuptools setup
setup(name='trimesh',
      version=__version__,
      description='Import, export, process, analyze and view triangular meshes.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Michael Dawson-Haggerty',
      author_email='mikedh@kerfed.com',
      license='MIT',
      url='http://github.com/mikedh/trimesh',
      keywords='graphics mesh geometry 3D',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Natural Language :: English',
          'Topic :: Scientific/Engineering'],
      packages=[
          'trimesh',
          'trimesh.ray',
          'trimesh.path',
          'trimesh.path.exchange',
          'trimesh.scene',
          'trimesh.visual',
          'trimesh.viewer',
          'trimesh.exchange',
          'trimesh.resources',
          'trimesh.interfaces'],
      package_data={'trimesh': ['resources/*.template',
                                'resources/*.json']},
      install_requires=['numpy',
                        'scipy',
                        'networkx'],
      extras_require={'test': ['pytest',
                               'pytest-cov',
                               'pyinstrument',
                               'coveralls'],
                      'easy': list(requirements_easy),
                      'all': list(requirements_all)}
      )
