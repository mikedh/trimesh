Advanced Installation
=====================

The minimal dependancies for ``trimesh`` are
`numpy <http://www.numpy.org/>`__, `scipy <http://www.scipy.org/>`__ and
`networkx <https://networkx.github.io/>`__.

All other dependancies are 'soft', or trimesh will only fail if a function is called that requires something not installed. If you do the most basic pip install, it will only require those three packages:

.. code:: bash

   pip install trimesh

	  
Conda Install
'''''''''''''

The easiest way to get going on the most platforms is through Conda.

.. code:: bash

   # install Miniconda if you have no conda:
   # https://conda.io/docs/install/quick.html
   # if you are on Linux and feeling lazy, just run:
   wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  --no-check-certificate -O miniconda.sh
   bash miniconda.sh -b -p /opt/conda; rm miniconda.sh
   export PATH="/opt/conda/bin:$PATH"
   conda update -q conda
   conda create -q -n conda_env python=3.6
   source activate conda_env
   
   # cyassimp is a much faster binding for the assimp importers
   # they use non- standard labels, master vs main
   # note that it installs cleanly in Linux and Windows
   # but generally fails on OSX
   conda config --add channels menpo 
   conda install -c menpo/label/master cyassimp

   # install most trimesh requirements with built components from conda-forge
   conda config --add channels conda-forge  # rtree, shapely, pyembree
   conda install shapely rtree graph-tool pyembree numpy scipy

   # graph-tool is only tested on Ubuntu and very unlikely to work elsewhere
   # it is used in connected components calculations, and is slighly faster in
   # most cases than scipy.sparse and networkx (although if graph-tool is not
   # installed trimesh will automatically use next the fastest option)
   conda config --add channels ostrokach
   conda install graph-tool

   # requires compilation, and fails a lot
   # used by trimesh.primitives.Extrusion objects
   pip install meshpy

   # install trimesh et al (everything except for meshpy, which is separated
   # above because it fails all the time and is easier to debug individually)
   pip install trimesh[easy]


Ubuntu Notes
''''''''''''''''''

Blender and openSCAD are soft dependancies used for boolean operations with subprocess, you can get them with apt:

.. code:: bash

   sudo apt-get install openscad blender

Windows Notes
''''''''''''''''''''

The easiest way to get going on Windows is to install the `Anaconda
Python distribution <https://www.continuum.io/downloads>`__.

Most requirements are available as above, but to get ``meshpy`` the easiest way is from the `Unofficial Windows Binaries from Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`__


