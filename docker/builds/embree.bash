#!/bin/bash
set -xe

# Fetch the archive from GitHub releases.
wget https://github.com/embree/embree/releases/download/v2.17.7/embree-2.17.7.x86_64.linux.tar.gz -O /tmp/embree.tar.gz -nv
echo "2c4bdacd8f3c3480991b99e85b8f584975ac181373a75f3e9675bf7efae501fe  /tmp/embree.tar.gz" | sha256sum --check
tar -xzf /tmp/embree.tar.gz --strip-components=1 -C /usr/local

# Install python bindings for embree (and upstream requirements).
pip install --no-cache-dir numpy cython
pip install --no-cache-dir https://github.com/scopatz/pyembree/releases/download/0.1.6/pyembree-0.1.6.tar.gz
