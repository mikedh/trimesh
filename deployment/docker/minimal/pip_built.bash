apt-get update
apt-get install -y --no-install-recommends g++

# install and build meshpy
source activate docker-environment
pip install --upgrade pip
pip install 'meshpy==2016.1.2'

apt-get remove --auto-remove --purge -y g++
apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
