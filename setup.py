#!/usr/bin/env python

from distutils.core import setup

setup(name='trimesh',
      version='1.0.4',
      description='Load, process, and view triangular meshes.',
      author='Mike Dawson-Haggerty',
      author_email='mik3dh@gmail.com',
      url='github.com/mikedh/trimesh.py',
      packages=['trimesh',
                'trimesh.io',
                'trimesh.ray'],
      install_requires=['numpy', 'scipy', 'networkx', 'pyglet']
     )
