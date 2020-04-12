#!/bin/bash

# Fetch the archive from GitHub releases.
wget https://github.com/embree/embree/releases/download/v2.17.7/embree-2.17.7.x86_64.linux.tar.gz -O /tmp/embree.tar.gz -nv
sha256sum /tmp/embree.tar.gz
echo "a3ac692397574166ad99493ff1efcd7b5c69b580e7eb4500d5d181b2f676aa6e  /tmp/embree.tar.gz" | sha256sum --check
tar -xzf /tmp/embree.tar.gz --strip-components=1 -C /usr/local

# Install python bindings for embree (and upstream requirements).
pip install --no-cache-dir numpy cython rtree==0.9.4
pip install --no-cache-dir https://github.com/scopatz/pyembree/releases/download/0.1.6/pyembree-0.1.6.tar.gz
