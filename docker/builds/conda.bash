set -xe

cd
# get versioned miniconda installer via HTTPS
wget https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh --quiet -O miniconda.sh
# check hash of file
echo "80ecc86f8c2f131c5170e43df489514f80e3971dd105c075935470bbf2476dea  miniconda.sh" | sha256sum --check
# run miniconda install
bash miniconda.sh -b -p ~/conda
rm miniconda.sh

# make sure conda bin is in front of PATH so we get correct pip
export PATH="~/conda/bin:$PATH"

# create a conda env
conda config --set always_yes yes --set changeps1 no
conda create -q -n denv python=3.6

# make sure pip/conda is the latest
pip install --upgrade pip
conda update -n base conda

# add conda-forge as remote channel
conda config --add channels conda-forge

# scikit-image is used for marching cubes
# pyembree is used for fast ray tests
conda install scikit-image pyembree 

# actually install trimesh and pytest
pip install trimesh[all] pytest pyassimp==4.1.3

# remove archives
conda clean --all -y
