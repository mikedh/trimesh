set -xe


# install build requirements
apt-get update -qq
# packages needed to build stuff
PACKAGES_BUILD="build-essential g++ gcc cmake sudo"
apt-get -y --no-install-recommends -qq install $PACKAGES_BUILD
# install draco, google's mesh compression utility
# bash "$(dirname $0)/draco.bash"
# install VHACD, a mesh decomposition utility
#bash "$(dirname $0)/vhacd.bash"

# run the middle bit as `user`
exec sudo -i -u user /bin/bash - << eof

# get versioned miniconda installer via HTTPS
wget https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh --quiet -O miniconda.sh

# check hash of file
echo "bfe34e1fa28d6d75a7ad05fd02fa5472275673d5f5621b77380898dee1be15d2 miniconda.sh" | sha256sum --check
# run miniconda install
bash miniconda.sh -b -p /home/user/conda
# delete installer
rm miniconda.sh

# make sure conda base
export PATH="/home/user/conda/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
# add conda-forge as remote channel
conda config --add channels conda-forge

# pyembree is used for fast ray tests
# this will also install numpy from conda
# conda/numpy is compiled with intel's MKL
conda install pyembree rtree

# include all soft dependencies
pip install --no-cache-dir /tmp/trimesh[all,test] pyassimp==4.1.3

# remove archives
conda clean --all -f -y
# remove pip cache and temp files
rm -rf ~/.cache || true

eof

# remove build packages from image
apt-get remove -y --purge $PACKAGES_BUILD
# remove any orphaned packages
apt-get autoremove -y
# remove garbage
apt-get clean -y
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

