#!/bin/bash
# Install draco, google's mesh compression utility.
set -xe

# Remove temporary build directories.
rm -rf /tmp/draco_build /tmp/draco_source /tmp/draco.tar.gz

# Fetch the archive from github releases.
wget https://github.com/google/draco/archive/1.3.5.tar.gz -O /tmp/draco.tar.gz -nv
echo "a3ac692397574166ad99493ff1efcd7b5c69b580e7eb4500d5d181b2f676aa6e  /tmp/draco.tar.gz" | sha256sum --check
tar -xzf /tmp/draco.tar.gz --strip-components=1 --one-top-level=/tmp/draco_source 

# Perform the build.
mkdir /tmp/draco_build
cd /tmp/draco_build
cmake /tmp/draco_source
make -j$(nproc)

# Move executables to local path.
mv /tmp/draco_build/draco_encoder /usr/local/bin
mv /tmp/draco_build/draco_decoder /usr/local/bin

# Remove temporary build directories.
rm -rf /tmp/draco_build /tmp/draco_source /tmp/draco.tar.gz
