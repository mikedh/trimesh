set -xe
apt-get update
apt-get install -y --no-install-recommends blender openscad wget bzip2 supervisor libgl1-mesa-glx libgl1-mesa-dri xvfb xauth libgeos-dev libspatialindex-c4v5 ca-certificates

# get teigha converter for DWG to DXF conversion
wget https://download.opendesign.com/guestfiles/ODAFileConverter/ODAFileConverter_QT5_lnxX64_4.7dll.deb --quiet -O teigha.deb
echo "12d7641d95f6f4bb06829de05ddb03e1d802d08554bd743fd7ad06042af1dffa  teigha.deb" | sha256sum --check

dpkg -i teigha.deb
rm teigha.deb

# remove garbage
apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
