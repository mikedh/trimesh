set -xe

# in this directory build the image with a tag
docker build . -t renderworker

# run the container we just built
# the -v will mount the current directory
# as /output inside the container
docker run -v `pwd`:/output renderworker
