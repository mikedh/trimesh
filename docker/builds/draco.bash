set -xe

# install draco, google's mesh compression utility
# requires cmake
mkdir /tmp/draco_build
git clone http://github.com/google/draco.git /tmp/draco
cd /tmp/draco
# lock to a commit to avoid breakage
git checkout e3a9d6ce5241f2b7dc668e11db0a4308a65dd3fd

cd /tmp/draco_build
cmake /tmp/draco
make
mv draco_encoder /usr/bin
mv draco_decoder /usr/bin
