FROM python:3.12-slim-bookworm AS base
LABEL maintainer="mikedh@kerfed.com"

# Install helper script to PATH.
COPY --chmod=755 docker/trimesh-setup /usr/local/bin/

# Create a non-root user with `uid=499`.
RUN useradd -m -u 499 -s /bin/bash user

# Required for Python to be able to find libembree.
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

# So scripts installed from pip are in $PATH
ENV PATH="/home/user/.local/bin:$PATH"

## install things that need building
FROM base AS build

# install build essentials for compiling stuff
RUN trimesh-setup --install build

# copy in essential files
COPY --chown=499 trimesh/ /home/user/trimesh
COPY --chown=499 pyproject.toml /home/user/

# switch to non-root user
USER user

# install trimesh into .local
# then delete any included test directories
# and remove Cython after all the building is complete


# TODO
# remove mapbox-earcut fork when this is merged:
# https://github.com/skogler/mapbox_earcut_python/pull/15
RUN pip install --user /home/user[easy] && \
    pip install --user --force-reinstall git+https://github.com/mikedh/mapbox_earcut_python.git && \
    find /home/user/.local -type d -name tests -prune -exec rm -rf {} \;

####################################
### Build output image most things should run on
FROM base AS output

# switch to non-root user
USER user
WORKDIR /home/user

# just copy over the results of the compiled packages
COPY --chown=499 --from=build /home/user/.local /home/user/.local

# Set environment variables for software rendering.
ENV XVFB_WHD="1920x1080x24"\
    DISPLAY=":99" \
    LIBGL_ALWAYS_SOFTWARE="1" \
    GALLIUM_DRIVER="llvmpipe"

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

# install things like pytest
# install prerelease for tests and make sure we're on Numpy 2.X
RUN pip install --pre --upgrade .[all] && \
    python -c "import numpy as n; assert(n.__version__.startswith('1'))"

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

