## A 'minimal' image with Debian Jessie
# Doesn't install blender, openscad or graph-tool, so no X dependancy
# but also no boolean operations and slightly slower split operations

from debian:jessie

maintainer Dockerfiles

add . /home/docker/config

run apt-get update
run apt-get install -y `cat /home/docker/config/apt.list`

run pip3 install git+https://github.com/robotics/assimp_latest.git
run pip3 install trimesh[all] --upgrade
