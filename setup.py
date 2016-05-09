#!/usr/bin/env python

from setuptools import setup

# load __version__
exec(open('trimesh/version.py').read())

import os
long_description = ''
if os.path.exists('README.md'):
    try:
        import pypandoc
        long_description = pypandoc.convert('README.md', 'rst')
    except ImportError:
        long_description = open('README.md', 'r').read()

setup(name = 'trimesh',
      version = __version__,
      description='Import, export, process, analyze and view triangular meshes.',
      long_description=long_description,
      author='Mike Dawson-Haggerty',
      author_email='mik3dh@gmail.com',
      license = 'MIT',
      url='http://github.com/mikedh/trimesh',
      keywords = 'graphics mesh geometry',
      classifiers = [
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Natural Language :: English',
          'Operating System :: POSIX :: Linux',
          'Topic :: Scientific/Engineering' ],
      packages = [
          'trimesh',
          'trimesh.io',
          'trimesh.ray',
          'trimesh.path',
          'trimesh.path.io',
          'trimesh.scene',
          'trimesh.templates',
          'trimesh.interfaces'],
      package_data={'trimesh' : ['templates/*.template']},
      install_requires = [
          'numpy', 
          'scipy', 
          'networkx' ],
      extras_require    = {'path' : ['svg.path', 
                                     'Shapely', 
                                     'rtree'],
                           'viewer' : ['pyglet']}
     )
