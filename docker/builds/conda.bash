set -xe

cd
wget https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh --no-check-certificate --quiet -O miniconda.sh
bash miniconda.sh -b -p ~/conda
rm miniconda.sh

export PATH="~/conda/bin:$PATH"
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

# install most trimesh requirements with built components 
#conda install -c conda-forge shapely rtree numpy scipy

# remove archives
conda clean --all -y
