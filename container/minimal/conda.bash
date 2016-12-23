wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda

export PATH="$HOME/miniconda/bin:$PATH"

conda config --set always_yes yes --set changeps1 no

conda config --add channels ostrokach
conda config --add channels conda-forge
conda config --add channels scitools
conda config --add channels ioos
conda config --add channels defaults
conda update -q conda
conda install shapely rtree graph-tool numpy scipy
