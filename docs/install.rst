Advanced Installation
=====================

The minimal requirement to install ``trimesh`` is just
`numpy <http://www.numpy.org/>`__.

All other dependancies are 'soft', or trimesh will only fail if a function is called that requires something not installed. If you do the most basic pip install, it will only require those three packages:

.. code:: bash

   pip install trimesh

If you'd like most soft dependancies which should install cleanly:

.. code:: bash

   pip install trimesh[easy]


	  
Conda Install
'''''''''''''

The easiest way to get going on the most platforms is through Conda.

.. code:: bash

   # install Miniconda if you have no conda:
   # https://conda.io/docs/install/quick.html

   conda install -c conda-forge scikit-image shapely rtree pyembree

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

