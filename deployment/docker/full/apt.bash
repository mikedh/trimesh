apt-get update
apt-get install -y --no-install-recommends blender openscad libspatialindex-dev

apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
