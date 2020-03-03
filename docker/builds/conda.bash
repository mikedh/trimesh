set -xe

cd
# get versioned miniconda installer via HTTPS
wget https://repo.anaconda.com/miniconda/Miniconda2-4.7.12.1-Linux-x86_64.sh --quiet -O miniconda.sh

# check hash of file
echo "383fe7b6c2574e425eee3c65533a5101e68a2d525e66356844a80aa02a556695 miniconda.sh" | sha256sum --check
# run miniconda install
bash miniconda.sh -b -p ~/conda
# delete installer
rm miniconda.sh

# make sure conda base
export PATH="~/conda/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
# add conda-forge as remote channel
conda config --add channels conda-forge

# pyembree is used for fast ray tests
# this will also install numpy from conda
# conda/numpy is compiled with intel's MKL
conda install pyembree rtree

# install trimesh from the repo
cd /tmp/trimesh
# include all soft dependencies
pip install .[all,test] pyassimp==4.1.3

# remove archives
conda clean --all
# remove pip cache and temp files
rm -rf ~/.cache/pip || true
