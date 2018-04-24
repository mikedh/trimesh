set -xe

# install draco, google's mesh compression utility
# requires cmake
mkdir /tmp/draco_build
git clone http://github.com/google/draco.git /tmp/draco
cd /tmp/draco
git checkout 5bbf04c298856b096ceba77924183d041d1e7dd5

cd /tmp/draco_build
cmake /tmp/draco
make
mv draco_encoder /usr/bin
mv draco_decoder /usr/bin
