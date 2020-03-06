set -xe

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
