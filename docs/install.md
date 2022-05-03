Advanced Installation
=====================

The only thing required to install `trimesh` is
[numpy](http://www.numpy.org/).

All other dependencies are \'soft,\' or trimesh will raise the exceptions (usually but not always an `ImportError`) at runtime if a function is called that requires a package that isn\'t installed. If you do the most basic install of `trimesh` it will only install `numpy`:

```
pip install trimesh
```

If you\'d like most soft dependencies which should install cleanly on Mac, Windows, and Linux, you can use the `easy` pip extra:

```
pip install trimesh[easy]
```

Or if you want the full experience, you can try the `all` extra, where packages may only be available for Linux:
```
pip install trimesh[all]
```


Conda Packages
--------------

If you prefer a `conda` environment, `trimesh` is available on `conda-forge` ([trimesh-feedstock repo](https://github.com/conda-forge/trimesh-feedstock))


If you install [Miniconda](https://conda.io/docs/install/quick.html) you can then run:

```
conda install -c conda-forge trimesh
```
      
Ubuntu-Debian Notes
------------

Blender and openSCAD are soft dependencies used for boolean operations with subprocess, you can get them with apt:

```
sudo apt-get install openscad blender
```
