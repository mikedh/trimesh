# a size unconstrained Ubuntu based image with all trimesh dependancies
# includes X as graph-tool, blender, and openscad require it for install

from ubuntu:16.04

maintainer Dockerfiles

# install APT packages
copy apt.bash /tmp/
run bash /tmp/apt.bash

# install conda environment
copy conda.bash /tmp/
run bash /tmp/conda.bash

# add conda to path so we use their python
env PATH /opt/conda/bin:$PATH

# install trimesh from pypi
run pip install trimesh[all]
