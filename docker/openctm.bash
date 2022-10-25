#!/bin/bash
set -xe

# Remove any temporary build directories.
rm -rf /tmp/ctm

# Clone the most active github fork of openCTM.
# Lock to a manually verified commit.
CTM_COMMIT=243a343bd23bbeef8731f06ed91e3996604e1af4
wget https://github.com/Danny02/OpenCTM/archive/${CTM_COMMIT}.tar.gz -O /tmp/ctm -nv

# Build the library.
make -j$(nproc) -C /tmp/ctm -f Makefile.linux openctm

# Copy the build ourselves, as make install tries
# to copy things we haven't built and don't need.
cp /tmp/ctm/lib/libopenctm.so /usr/local/lib/
cp /tmp/ctm/lib/openctm.h /usr/local/include/
cp /tmp/ctm/lib/openctmpp.h /usr/local/include/

# Remove any temporary build directories.
rm -rf /tmp/ctm
