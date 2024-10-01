# use a vanilla Debian base image
FROM debian:trixie-slim AS base
LABEL maintainer="mikedh@kerfed.com"

# Create a non-root user with `uid=499`.
RUN useradd -m -u 499 -s /bin/bash user && \
    apt-get update && \
    apt-get install --no-install-recommends -qq -y python3.12-venv && \
    apt-get clean -y 

USER user

WORKDIR /home/user

# install a python `venv`
# this seems a little silly since we're already in a container
# but if you use Debian methods like `update-alternatives`
# it won't provide a `pip` which works easily and it isn't
# easy to know how system packages interact with pip packages
RUN python3.12 -m venv venv

# So scripts installed from pip are in $PATH
ENV PATH="/home/user/venv/bin:$PATH"
ENV VIRTUAL_ENV="/home/user/venv"

# Install helper script to PATH.
COPY --chmod=755 docker/trimesh-setup /home/user/venv/bin

#######################################
## install things that need building
FROM base AS build

# copy in essential files
COPY --chown=499 trimesh/ /home/user/trimesh
COPY --chown=499 pyproject.toml /home/user/

# install trimesh into the venv
RUN pip install /home/user[easy]

# install FCL which currently has broken wheels on PyPi
RUN pip install https://github.com/BerkeleyAutomation/python-fcl/releases/download/v0.7.0.7/python_fcl-0.7.0.7-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

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
RUN trimesh-setup --install=test,gmsh,gltf_validator,llvmpipe,binvox
USER user

# install things like pytest and make sure we're on Numpy 2.X
RUN pip install .[all] && \
    python -c "import numpy as n; assert(n.__version__.startswith('2'))"

# check for lint problems
RUN ruff check trimesh

# run a limited array of static type checks
# TODO : get this to pass on base
RUN pyright trimesh/base.py || true

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
FROM scratch as docs
COPY --from=build_docs /home/user/docs/built/html/ ./

### Make sure the output stage is the last stage so a simple
# "docker build ." still outputs an expected image
FROM output as final

