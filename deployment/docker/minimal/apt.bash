apt-get update
apt-get install -y --no-install-recommends libspatialindex-c3 wget bzip2

apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
