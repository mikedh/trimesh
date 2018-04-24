set -xe
apt-get update
apt-get install -y --no-install-recommends blender openscad wget bzip2 supervisor libgl1-mesa-glx libgl1-mesa-dri xvfb libgeos-dev libspatialindex-c4v5

# grab the VHACD (convex segmenter) binary
wget --no-check-certificate https://github.com/mikedh/v-hacd-1/raw/master/bin/linux/testVHACD
chmod +x testVHACD
mv testVHACD /usr/bin/

# remove garbage
apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
