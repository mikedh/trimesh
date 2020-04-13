#!/bin/bash
set -xe

# Set the installation path.
VHACD_PATH=/usr/local/bin/testVHACD

# Grab the VHACD (convex segmenter) binary.
wget https://github.com/mikedh/v-hacd-1/raw/master/bin/linux/testVHACD -O "${VHACD_PATH}" -nv

# Check the hash of the downloaded file
echo "e1e79b2c1b274a39950ffc48807ecb0c81a2192e7d0993c686da90bd33985130  ${VHACD_PATH}" | sha256sum --check

# Make it executable.
chmod +x ${VHACD_PATH}
