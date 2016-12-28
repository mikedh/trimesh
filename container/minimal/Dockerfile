from debian:jessie

maintainer Dockerfiles

# copy this container directory to the filesystem
add . /home/docker/config

run bash /home/docker/config/apt.bash
run bash /home/docker/config/conda.bash

env PATH /opt/conda/bin:$PATH

run pip install trimesh[all]
