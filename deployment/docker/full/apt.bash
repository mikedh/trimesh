apt-get update
apt-get install -y --no-install-recommends blender openscad libspatialindex-c3 wget bzip2

wget --no-check-certificate https://github.com/mikedh/v-hacd-1/raw/master/bin/linux/testVHACD
chmod +x testVHACD
mv testVHACD /usr/bin/

apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
