cd
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  --no-check-certificate --quiet -O miniconda.sh
bash miniconda.sh -b -p ~/conda
rm miniconda.sh

export PATH="~/conda/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
conda create -q -n denv python=3.6

# make sure pip/conda is the latest
pip install --upgrade pip
conda update -n base conda

conda config --add channels conda-forge  # rtree, shapely, pyembree

# scikit-image is used for marching cubes
conda install -c conda-forge scikit-image

# pyembree is used for fast ray tests
conda install -c conda-forge pyembree

# install most trimesh requirements with built components 
#conda install -c conda-forge shapely rtree numpy scipy

# remove archives
conda clean --all -y
