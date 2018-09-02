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

   # install most trimesh requirements with built components from conda-forge
   conda config --add channels conda-forge  # rtree, shapely, pyembree
   conda install shapely rtree graph-tool pyembree numpy scipy
   conda install -c conda-forge scikit-image

   # install trimesh and all possible dependancies
   # if this fails try: pip install trimesh[easy]
   pip install trimesh[all]


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


