set -xe

# grab the VHACD (convex segmenter) binary
wget --no-check-certificate https://github.com/mikedh/v-hacd-1/raw/master/bin/linux/testVHACD
# make it executable
chmod +x testVHACD
# move it into PATH
mv testVHACD /usr/bin/
