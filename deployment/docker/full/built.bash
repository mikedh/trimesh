# add jessie backports
echo "deb http://ftp.debian.org/debian jessie-backports main" > /etc/apt/sources.list.d/backports.list

apt-get update
PACKAGES_BUILD="git ca-certificates build-essential g++ gcc cmake"
#apt-get install -y --no-install-recommends git ca-certificates
# debian jessie cmake is very old, so use backport
apt-get -y --no-install-recommends -t jessie-backports install $PACKAGES_BUILD

# install draco, google's mesh compression utility
mkdir ~/draco_build
git clone http://github.com/google/draco.git /tmp/draco
cd /tmp/draco
git checkout 5bbf04c298856b096ceba77924183d041d1e7dd5

cd ~/draco_build
cmake /tmp/draco
make
mv draco_encoder /usr/bin
mv draco_decoder /usr/bin
cd
rm -rf draco_build

# install and build meshpy
source activate docker-environment
pip install --upgrade pip
pip install meshpy

# remove garbage
apt-get remove --auto-remove --purge -y $PACKAGES_BUILD
apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
