# build docs in the trimesh/trimesh docker images
set -xe

# pull the base image
docker pull trimesh/trimesh:latest

# trimesh images are non-root user
# we need to copy the docs, examples and readme into the image
# the docker volume is read-only for "docker reasons" so build
# inside of the image trimesh install then copy
docker run -t --name dummy -v `pwd`/../:/trimesh trimesh/trimesh:latest bash -c "cp -R /trimesh/models /trimesh/docs /trimesh/examples /trimesh/README.md /opt/trimesh/ && python /trimesh/docker/builds/pandoc.py && cd /opt/trimesh/docs && make";

# copy the built docs out of the image
docker cp dummy:/opt/trimesh/docs/_build/html ./

docker rm -f dummy

