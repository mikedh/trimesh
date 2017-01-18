# add jessie backports
echo "deb http://ftp.debian.org/debian jessie-backports main" > /etc/apt/sources.list.d/backports.list

apt-get update
PACKAGES_BUILD="git ca-certificates build-essential g++ gcc cmake"
#apt-get install -y --no-install-recommends git ca-certificates
# debian jessie cmake is very old, so use backport
apt-get -y --no-install-recommends -t jessie-backports install $PACKAGES_BUILD

# install draco, google's mesh compression utility
git clone http://github.com/google/draco.git draco
cd draco
cmake .
make
mv draco_encoder /usr/bin
mv draco_decoder /usr/bin
cd ..
rm -rf draco

# install and build meshpy
source activate docker-environment
pip install --upgrade pip
pip install meshpy

# remove garbage
apt-get remove --auto-remove --purge -y $PACKAGES_BUILD
apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
