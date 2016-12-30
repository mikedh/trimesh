from debian:jessie

maintainer Dockerfiles

# install APT packages
copy apt.bash /tmp/
run bash /tmp/apt.bash

# install conda environment
copy conda.bash /tmp/
run bash /tmp/conda.bash

# add conda to path so we use their python
env PATH /opt/conda/bin:$PATH

# make sure pip is up to date
run pip install --upgrade pip

# install trimesh from pypi
run pip install trimesh[all]
