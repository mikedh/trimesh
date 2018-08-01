set -xe
apt-get update
apt-get install -y --no-install-recommends blender openscad wget bzip2 supervisor libgl1-mesa-glx libgl1-mesa-dri xvfb libgeos-dev libspatialindex-c4v5

# get teigha converter for DWG to DXF conversion
wget https://download.opendesign.com/guestfiles/TeighaFileConverter/TeighaFileConverter_QT5_lnxX64_4.7dll.deb --no-check-certificate --quiet -O teigha.deb
dpkg -i teigha.deb
rm teigha.deb

# remove garbage
apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
