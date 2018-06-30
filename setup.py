#!/usr/bin/env python

import os
from setuptools import setup

# load __version__
exec(open('trimesh/version.py').read())


long_description = ''
if os.path.exists('README.md'):
    with open('README.md', 'r') as f:
        long_description = f.read()

setup(name='trimesh',
      version=__version__,
      description='Import, export, process, analyze and view triangular meshes.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Michael Dawson-Haggerty',
      author_email='mik3dh@gmail.com',
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
          'Natural Language :: English',
          'Topic :: Scientific/Engineering'],
      packages=[
          'trimesh',
          'trimesh.io',
          'trimesh.ray',
          'trimesh.path',
          'trimesh.path.io',
          'trimesh.scene',
          'trimesh.resources',
          'trimesh.interfaces'],
      package_data={'trimesh': ['resources/*.template']},
      install_requires=['numpy',
                        'scipy',
                        'networkx'],
      extras_require={'easy': ['lxml',
                               'pyglet',
                               'Shapely',
                               'rtree',
                               'svg.path',
                               'sympy',
                               'msgpack',
                               'pillow',
                               'colorlog'],
                      'all': ['lxml',
                              'pyglet',
                              'Shapely',
                              'rtree',
                              'svg.path',
                              'triangle',
                              'sympy',
                              'msgpack',
                              'python-fcl',
                              'colorlog',
                              'xxhash',
                              'pillow',
                              'setuptools']}
      )
