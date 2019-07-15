set -xe

cd
# get versioned miniconda installer via HTTPS
wget https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh --quiet -O miniconda.sh
# check hash of file
echo "0d6b23895a91294a4924bd685a3a1f48e35a17970a073cd2f684ffe2c31fc4be  miniconda.sh" | sha256sum --check
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
conda install pyembree 

# install trimesh from the repo
cd /tmp/trimesh
# include all soft dependencies
pip install .[all,test] pyassimp==4.1.3

# remove archives
conda clean --all
# remove pip cache and temp files
rm -rf ~/.cache/pip || true
