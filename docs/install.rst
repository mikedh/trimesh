Advanced Installation
=====================

The only thing required to install ``trimesh`` is `numpy <http://www.numpy.org/>`__.

All other dependencies are 'soft,' or trimesh will raise the ``ImportError`` at runtime if a function is called that requires a package that isn't installed. If you do the most basic install of ``trimesh`` it will only install ``numpy``:

.. code:: bash

   pip install trimesh

If you'd like most soft dependencies which should install cleanly, you can use the ``easy`` pip extra:

.. code:: bash

   pip install trimesh[easy]


	  
Conda Install
'''''''''''''

The easiest way to get going on the most platforms is through a Python provided by conda. You can install `Miniconda <https://conda.io/docs/install/quick.html>`__ easily on all major platforms. Then, to install ``trimesh``:

.. code:: bash

   conda install -c conda-forge scikit-image shapely rtree pyembree

   # install trimesh and all possible dependencies
   # if this fails try: pip install trimesh[easy]
   pip install trimesh[all]


Ubuntu Notes
''''''''''''''''''

Blender and openSCAD are soft dependencies used for boolean operations with subprocess, you can get them with apt:

.. code:: bash

   sudo apt-get install openscad blender

Windows Notes
''''''''''''''''''''

The easiest way to get going on Windows is to install the `Anaconda
Python distribution <https://www.continuum.io/downloads>`__.

