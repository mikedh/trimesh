from debian:jessie

maintainer Dockerfiles

# install APT packages
COPY apt.bash /tmp/
RUN bash /tmp/apt.bash

# install conda environment
COPY conda.bash /tmp/
RUN bash /tmp/conda.bash

COPY pip.bash /tmp/
RUN bash /tmp/pip.bash

# add conda and our env to path
ENV PATH /opt/conda/envs/docker-environment/bin:/opt/conda/bin:$PATH
