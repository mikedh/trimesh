## A 'minimal' image with Debian Jessie
# Doesn't install blender, openscad or graph-tool, so no X dependancy
# but also no boolean operations and slightly slower split operations

from debian:jessie

maintainer Dockerfiles

add . /home/docker/config

run bash /home/docker/config/apt.bash
run bash /home/docker/config/conda.bash

run pip3 install git+https://github.com/robotics/assimp_latest.git
run pip3 install trimesh[all]
