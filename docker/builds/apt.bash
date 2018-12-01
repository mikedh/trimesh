set -xe
apt-get update
apt-get install -y --no-install-recommends blender openscad wget bzip2 supervisor libgl1-mesa-glx libgl1-mesa-dri xvfb xauth libgeos-dev libspatialindex-c4v5

# get teigha converter for DWG to DXF conversion
wget https://download.opendesign.com/guestfiles/ODAFileConverter/ODAFileConverter_QT5_lnxX64_4.7dll.deb --no-check-certificate --quiet -O teigha.deb
echo "a9ca9c72e6303bc0a03b8b7f64a7300fec6092184f7efb2c8a89aca8d34ec32b  teigha.deb" | sha256sum --check

dpkg -i teigha.deb
rm teigha.deb

# remove garbage
apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
