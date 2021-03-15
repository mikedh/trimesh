#!/bin/bash
set -xe
DEBIAN_FRONTEND=noninteractive

# Install binary dependencies directly from APT.
apt-get update -qq
apt-get install -qq --no-install-recommends \
	wget bzip2 supervisor \
	libgl1-mesa-glx libgl1-mesa-dri xvfb xauth \
	ca-certificates zstd freeglut3-dev git sudo \
	build-essential g++ gcc cmake libfcl-dev

# Install a newer version of pandoc.
wget https://github.com/jgm/pandoc/releases/download/2.9.2/pandoc-2.9.2-1-amd64.deb -nv
echo "78525735ac6181f639c5c8776572d0ca10f0314c0052f5af2f369b5d0e1980b3  pandoc-2.9.2-1-amd64.deb" | sha256sum --check
sudo dpkg --install pandoc-2.9.2-1-amd64.deb
rm -f pandoc-2.9.2-1-amd64.deb

# Remove cache and build files.
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
