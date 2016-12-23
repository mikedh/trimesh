# a size unconstrained Ubuntu based image with all trimesh dependancies
# includes X as graph-tool, blender, and openscad require it for install

from ubuntu:16.04

maintainer Dockerfiles

# copy this container directory to the filesystem
add . /home/docker/config

run bash /home/docker/config/apt.bash
run bash /home/docker/config/conda.bash

run pip install git+https://github.com/robotics/assimp_latest.git
run pip install trimesh[all]
