set -xe

# grab the VHACD (convex segmenter) binary
wget --no-check-certificate https://github.com/mikedh/v-hacd-1/raw/master/bin/linux/testVHACD
# check the hash of the downloaded file
echo "e1e79b2c1b274a39950ffc48807ecb0c81a2192e7d0993c686da90bd33985130  testVHACD" | sha256sum --check
# make it executable
chmod +x testVHACD
# move it into PATH
mv testVHACD /usr/bin/
