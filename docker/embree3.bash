#!/bin/bash
set -xe

# Fetch the archive from GitHub releases.
wget https://github.com/embree/embree/releases/download/v3.12.1/embree-3.12.1.x86_64.linux.tar.gz -O /tmp/embree.tar.gz

echo "5e218dd4e95c035c04aa893f7b1169ade11ecc57580e0027eae2ec84cdc9baff  /tmp/embree.tar.gz" | sha256sum --check
sudo tar -xzf /tmp/embree.tar.gz --strip-components=1 -C /usr/local
# remove archive
rm -rf /tmp/embree.tar.gz

# Install python bindings for embree3
pip install --no-cache-dir git+https://github.com/sampotter/python-embree@5d604f8ce30c752c42308266cd049737aa7124b0

