from debian:jessie-slim

# add conda and our env to path
ENV PATH /opt/conda/envs/docker-environment/bin:$PATH

# install apt stuff thats staying around
COPY apt.bash /tmp/
RUN bash /tmp/apt.bash

# install conda environment
COPY conda.bash /tmp/
RUN bash /tmp/conda.bash

# install packages are going to need to build
COPY draco.bash /tmp/
COPY built.bash /tmp/
RUN bash /tmp/built.bash

# install straightforward pip packages including trimesh
COPY pip.bash /tmp/
RUN bash /tmp/pip.bash


