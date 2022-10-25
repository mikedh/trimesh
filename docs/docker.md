Writing Docker Images
=======================

### Docker Basics

[Docker's install guide for Ubuntu.](https://docs.docker.com/desktop/install/ubuntu/)

### Using pip

Typically when writing Dockerfiles I don't like to use non-base remote images (i.e. in the vein of `debian:buster-slim`) as they are often unmaintained and unknowable black boxes.

It should generally work fine to just use a first party base image and install trimesh via `pip` which always has the latest version:
```
FROM python:3.10-slim-bullseye
RUN pip install trimesh[easy]
```


### Using prebuilt images

The `trimesh/trimesh` docker images are based on the offical Python base images built on top of a Debian slim image: currently `python:3.10-slim-bullseye`. They are built and pushed to Docker Hub automatically in Github Actions for every release. 

If you need some of the more demanding dependencies they can be a good option. The `trimesh/trimesh` images are pushed with three tags: `latest` (for latest :), semantic version (i.e. `3.15.5`), or git short hash (i.e. `1c6178d`). These images include `embree` and `trimesh[all]` which is run in a multi-stage build to avoid including g++ in the final image.

They run as the non-root user `user` and the working directory is `/home/user`. A minimal docker file would look like:
```
FROM trimesh/trimesh:latest

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
CMD python app.py
```

Which you could then save as the file `Dockerfile` and build with:
```
docker build . -t example
docker run -t example
```

### Building Trimesh's Images

Trimesh is using a multistage build to avoid copying in things like `g++`, so you have to explicitly specify that you want to build the `output` target:

```
docker build --target output -t trimesh/trimesh:latest .
```

There is also a `Makefile` which enables Buildkit, tags the versioned images, and provides access to other targets like `test` and `docs`:
```
# will list the available options
make help

# will build and tag a `trimesh/trimesh:latest` image
# and also tag it with semantic version and git hash
make build

# will build trimesh images and then in a different
# build stage install the testing requirments and
# run all of trimesh's unit tests inside the built
# docker image.
make test
```
