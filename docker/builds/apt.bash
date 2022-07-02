#!/bin/bash
set -xe
DEBIAN_FRONTEND=noninteractive

PACKAGES="wget curl sudo g++ make"
if [ $1 = true ]
then
    # add the X11 options if requested
    PACKAGES="${PACKAGES} bzip2 zstd git \
        libgl1-mesa-glx libgl1-mesa-dri xvfb xauth \
	ca-certificates freeglut3-dev \
	build-essential"
fi

echo $PACKAGES

# Install binary dependencies directly from APT.
apt-get update -qq
apt-get install -qq --no-install-recommends $PACKAGES

# Remove cache and build files.
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 
