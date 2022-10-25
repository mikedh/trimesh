#!/bin/bash
set -xe

# Set the installation path.
BINVOX_PATH=/usr/local/bin/binvox

# Grab the binvox binary from the trimesh S3 bucket.
# This to avoid CI hammering the original address:
# http://www.patrickmin.com/binvox/linux64/binvox
wget https://trimesh.s3-us-west-1.amazonaws.com/binvox -O "${BINVOX_PATH}" -nv

# Check the hash of the file before using it.
echo "cc05b3ceec0b3f7061f629448c3764e87f035ec34bba46ec4dcc21e089dd40c5  ${BINVOX_PATH}" | sha256sum --check

# Make it executable.
chmod +x ${BINVOX_PATH}
