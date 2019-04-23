#!/usr/bin/env python

import os
from setuptools import setup

# load __version__
version_file = 'trimesh/version.py'
exec(open(version_file).read())

# load README.md as long_description
long_description = ''
if os.path.exists('README.md'):
    with open('README.md', 'r') as f:
        long_description = f.read()

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
                      'easy': ['lxml',
                               'pyglet',
                               'shapely',
                               'rtree',
                               'svg.path',
                               'sympy',
                               'msgpack',
                               'pillow',
                               'requests',
                               'xxhash',
                               'pycollada',
                               'colorlog'],
                      'all': ['lxml',
                              'pyglet',
                              'shapely',
                              'rtree',
                              'svg.path',
                              'triangle',
                              'sympy',
                              'msgpack',
                              'pycollada',
                              'python-fcl',
                              'colorlog',
                              'xxhash',
                              'requests',
                              'pycollada',
                              'psutil',
                              'pillow',
                              'setuptools',
                              'glooey']}
      )
