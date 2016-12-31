apt-get update
apt-get install -y --no-install-recommends wget bzip2 #conda install needs wget and bzip2

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  --no-check-certificate -O miniconda.sh
bash miniconda.sh -b -p /opt/conda
rm miniconda.sh

export PATH="/opt/conda/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
conda update -q conda

conda create -q -n docker-environment python=3.5
source activate docker-environment

# add channels
# graph-tool installs 1.8gb worth of stuff
# however it is really useful if you are calling
# mesh.split() on a lot of large meshes
#conda config --add channels ostrokach   # graph-tool
conda config --add channels conda-forge  # rtree, shapely, pyembree
conda config --add channels menpo        # cyassimp
conda config --add channels defaults     # stuff, things

# use openBLAS instead of MKL (saves 400mb and is in some cases faster)
conda install nomkl

# cyassimp is a much faster binding for the assimp importers
# they use non- standard labels, master vs main
conda install -c menpo/label/master cyassimp

# install most trimesh requirements with built components 
conda install shapely rtree  pyembree numpy scipy #graph-tool

# remove archives
conda clean --all -y

apt-get remove --purge --auto-remove -y bzip2 wget
apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
