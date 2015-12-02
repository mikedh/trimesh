#!/usr/bin/env python

from setuptools import setup

# load __version__ cleanly
exec(open('trimesh/version.py').read())

with open('README.md', 'r') as file_obj:
    readme = file_obj.read()


setup(name='trimesh',
      version=__version__,
      description='Import, export, process and view triangular meshes.',
      long_description=readme,
      author='Mike Dawson-Haggerty',
      author_email='mik3dh@gmail.com',
      license='MIT',
      url='github.com/mikedh/trimesh',
      packages         = ['trimesh',
                          'trimesh.io',
                          'trimesh.ray',
                          'trimesh.path',
                          'trimesh.scene'],
      install_requires = ['numpy', 
                          'scipy', 
                          'networkx'],
      extras_require    = {'path' : ['svg.path', 
                                     'Shapely', 
                                     'rtree'],
                           'viewer' : ['pyglet']}
     )
