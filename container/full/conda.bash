wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /opt/conda
rm miniconda.sh

export PATH="/opt/conda/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
conda update -q conda

conda create -q -n docker-environment python=3.5
source activate docker-environment

conda config --add channels ostrokach    # graph-tool
conda config --add channels conda-forge  # rtree, shapely, pyembree
conda config --add channels menpo        # cyassimp
conda config --add channels defaults     # stuff, things

# cyassimp is a much faster binding for the assimp importers
# they use non- standard labels, master vs main
conda install -c menpo/label/master cyassimp

conda install shapely rtree graph-tool pyembree pip numpy scipy

