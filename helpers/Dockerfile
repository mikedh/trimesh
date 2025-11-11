
# Stage 1: Build uv
FROM ghcr.io/astral-sh/uv:0.8.16 AS uv

# use a vanilla Debian base image
FROM debian:trixie-slim AS base
LABEL maintainer="mikedh@kerfed.com"

# Copy uv binary from the builder stage
COPY --from=uv uv /usr/local/bin/uv

# Create a non-root user with `uid=499` and switch to it.
RUN useradd -m -u 499 -s /bin/bash user && \
    apt-get update && \
    apt-get install --no-install-recommends -qq -y ca-certificates && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*
USER user

WORKDIR /home/user

# install a python `venv`
# this seems a little silly since we're already in a container
# but if you use Debian methods like `update-alternatives`
# it won't provide a `pip` which works easily and it isn't
# easy to know how system packages interact with pip packages
RUN uv venv --python=python3.13 venv

# Add `pip` script install location to $PATH
ENV PATH="/home/user/venv/bin:$PATH"
ENV VIRTUAL_ENV="/home/user/venv"
# bash script that redirects `pip install X`->`uv pip install X`
RUN echo '#!/bin/sh' > /home/user/venv/bin/pip && \
    echo 'exec uv pip "$@"' >> /home/user/venv/bin/pip && \
    chmod +x /home/user/venv/bin/pip

# Install helper script to PATH.
COPY --chmod=755 helpers/trimesh-setup /home/user/venv/bin

#######################################
## install things that need building
FROM base AS build

# copy in essential files
COPY --chown=499 trimesh/ /home/user/trimesh
COPY --chown=499 pyproject.toml /home/user/

# install trimesh into the venv
RUN pip install /home/user[easy]

####################################
### Build output image most things should run on
FROM base AS output

# just copy over the results of the compiled packages
COPY --chown=499 --from=build /home/user/venv /home/user/venv

###############################
#### Run Unit Tests
FROM output AS tests

# copy in tests and supporting files
COPY --chown=499 tests ./tests/
COPY --chown=499 trimesh ./trimesh/
COPY --chown=499 models ./models/
COPY --chown=499 pyproject.toml .

# codecov looks at the git history
COPY --chown=499 ./.git ./.git/

USER root
RUN trimesh-setup --install=test,gmsh,gltf_validator,llvmpipe,binvox,blender,build
USER user

RUN blender --version

# install things like pytest and make sure we're on Numpy 2.X
RUN pip install .[all] && \
    python -c "import numpy as n; assert(n.__version__.startswith('2'))"

# check for lint problems
RUN ruff check trimesh

# run pytest wrapped with xvfb for simple viewer tests
# print more columns so the short summary is usable
RUN COLUMNS=240 xvfb-run pytest \
    --cov=trimesh \
    --beartype-packages=trimesh \
    -p no:ALL_DEPENDENCIES \
    -p no:INCLUDE_RENDERING \
    -p no:cacheprovider tests


# set codecov token as a build arg to upload
ARG CODECOV_TOKEN=""
RUN curl -Os https://uploader.codecov.io/latest/linux/codecov && \
    	 chmod +x codecov && \
        ./codecov -t ${CODECOV_TOKEN} 

################################
### Build Sphinx Docs
FROM output AS build_docs

USER root
# install APT packages for docs
RUN trimesh-setup --install docs
USER user

COPY --chown=499 README.md .
COPY --chown=499 docs ./docs/
COPY --chown=499 examples ./examples/
COPY --chown=499 models ./models/
COPY --chown=499 trimesh ./trimesh/

WORKDIR /home/user/docs
RUN make

### Copy just the docs so we can output them
FROM scratch AS docs
COPY --from=build_docs /home/user/docs/built/html/ ./

### Make sure the output stage is the last stage so a simple
# "docker build ." still outputs an expected image
FROM output AS final

