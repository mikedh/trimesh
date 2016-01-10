#!/usr/bin/env python

from setuptools import setup

# load __version__
exec(open('trimesh/version.py').read())

import os
if os.path.exists('README.md'):
    try:
        import pypandoc
        long_description = pypandoc.convert('README.md', 'rst')
    except ImportError:
        long_description = open('README.md', 'r').read()
else:
    long_description = ''

setup(name = 'trimesh',
      version = __version__,
      description='Import, export, process, analyze and view triangular meshes.',
      long_description=long_description,
      author='Mike Dawson-Haggerty',
      author_email='mik3dh@gmail.com',
      license = 'MIT',
      url='github.com/mikedh/trimesh',
      packages         = ['trimesh',
                          'trimesh.io',
                          'trimesh.ray',
                          'trimesh.path',
                          'trimesh.path.io',
                          'trimesh.scene'],
      install_requires = ['numpy', 
                          'scipy', 
                          'networkx'],
      extras_require    = {'path' : ['svg.path', 
                                     'Shapely', 
                                     'rtree'],
                           'viewer' : ['pyglet']}
     )
