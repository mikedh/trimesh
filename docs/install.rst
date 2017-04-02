Advanced Installation
=====================

The minimum set of packages required to import ``trimesh`` are
`numpy <http://www.numpy.org/>`__, `scipy <http://www.scipy.org/>`__ and
`networkx <https://networkx.github.io/>`__.

Ubuntu Pre-install
''''''''''''''''''

Blender and openSCAD are backends used for boolean operations,
libspatialindex and libgeos are the libraries used by RTree and Shapely
respectivly, and cmake is included to build assimp if you want the
latest version.

.. code:: bash

    sudo apt-get install cmake openscad blender libspatialindex-dev libgeos-dev

Windows Pre-Install:
''''''''''''''''''''

The easiest way to get going on Windows is to install the `Anaconda
Python distribution <https://www.continuum.io/downloads>`__, followed by
``shapely``, ``rtree``, and ``meshpy`` from the `Unofficial Windows
Binaries from Christoph
Gohlke <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`__

Optional Dependencies
'''''''''''''''''''''

To install the latest assimp for `additional import
formats <http://www.assimp.org/main_features_formats.html>`__
(python-pyassimp in Ubuntu 14.04 is very old):

.. code:: bash

    sudo pip install git+https://github.com/robotics/assimp_latest.git

If you are using a lot of graph operations (specifically mesh.split)
trimesh will automatically use
`graph-tool <https://graph-tool.skewed.de/download>`__ if it is
installed, for a roughly 10x speedup over networkx on certain
operations.
