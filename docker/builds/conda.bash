set -xe

cd
# get versioned miniconda installer via HTTPS
wget https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh --quiet -O miniconda.sh
# check hash of file
echo "80ecc86f8c2f131c5170e43df489514f80e3971dd105c075935470bbf2476dea  miniconda.sh" | sha256sum --check
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

# install trimesh from repo version
cd /tmp/trimesh
# include all soft dependencies
pip install .[all,test] pyassimp==4.1.3

# remove archives
# conda clean --all
