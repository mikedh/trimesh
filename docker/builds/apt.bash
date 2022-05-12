#!/bin/bash
set -xe
DEBIAN_FRONTEND=noninteractive

# Install binary dependencies directly from APT.
apt-get update -qq
apt-get install -qq --no-install-recommends wget sudo
#  bzip2 zstd git 
#	libgl1-mesa-glx libgl1-mesa-dri xvfb xauth \
#	ca-certificates zstd freeglut3-dev \
#	build-essential

# Remove cache and build files.
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
