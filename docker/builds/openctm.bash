set -xe

# clone the most active github fork of openCTM
git clone https://github.com/Danny02/OpenCTM.git /tmp/ctm

cd /tmp/ctm
# lock to a manually verified commit 
git checkout 243a343bd23bbeef8731f06ed91e3996604e1af4
# build the library 
make -f Makefile.linux openctm
# copy the build ourselves as make install tries
# to copy things we haven't built and don't need
cp /tmp/ctm/lib/libopenctm.so /usr/lib
cp /tmp/ctm/lib/openctm.h /usr/local/include/
cp /tmp/ctm/lib/openctmpp.h /usr/local/include/

cd
rm -rf /tmp/ctm
