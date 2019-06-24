set -xe

# grab the binary from a trimesh S3 bucket
# do this to avoid CI hammering the original address:
# http://www.patrickmin.com/binvox/linux64/binvox
wget https://trimesh.s3-us-west-1.amazonaws.com/binvox
# check the hash of the file before using it
echo "cc05b3ceec0b3f7061f629448c3764e87f035ec34bba46ec4dcc21e089dd40c5  binvox" | sha256sum --check
# make it executable
chmod +x binvox
# move onto path
mv binvox /usr/bin/
