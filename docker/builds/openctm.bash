set -xe

# clone the most active github fork of openCTM
git clone https://github.com/Danny02/OpenCTM.git /tmp/ctm

cd /tmp/ctm
# just build the library
make -f Makefile.linux openctm
# copy the build ourself as make install tries
# to copy even things we haven't built
cp /tmp/ctm/lib/libopenctm.so /usr/lib
cp /tmp/ctm/lib/openctm.h /usr/local/include/
cp /tmp/ctm/lib/openctmpp.h /usr/local/include/

cd
rm -rf /tmp/ctm
