set -xe
apt-get update
apt-get install -y --no-install-recommends wget bzip2 supervisor libgl1-mesa-glx libgl1-mesa-dri xvfb xauth libgeos-dev libspatialindex-c5 libassimp-dev ca-certificates zstd unzip

# remove garbage
apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
