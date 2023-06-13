FROM python:3.11-slim-bullseye AS base
LABEL maintainer="mikedh@kerfed.com"

# Install helper script to PATH.
COPY --chmod=755 docker/trimesh-setup /usr/local/bin/

# Install base `apt` packages required for everything
RUN trimesh-setup --install base

# Create a local non-root user.
RUN useradd -m -s /bin/bash user

# Required for Python to be able to find libembree.
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

# So scripts installed from pip are in $PATH
ENV PATH="/home/user/.local/bin:$PATH"

## install things that need building
FROM base AS build

# install build essentials for compiling stuff
RUN trimesh-setup --install build

# copy in essential files
COPY --chown=user:user trimesh/ /home/user/trimesh
COPY --chown=user:user setup.py /home/user/

# switch to non-root user
USER user

# install trimesh into .local
RUN pip install /home/user[all]

####################################
### Build output image most things should run on
FROM base AS output

# switch to non-root user
USER user
WORKDIR /home/user

# just copy over the results of the compiled packages
COPY --chown=user:user --from=build /home/user/.local /home/user/.local

# Set environment variables for software rendering.
ENV XVFB_WHD="1920x1080x24"\
    DISPLAY=":99" \
    LIBGL_ALWAYS_SOFTWARE="1" \
    GALLIUM_DRIVER="llvmpipe"

###############################
#### Run Unit Tests
FROM output AS tests

# copy in tests and supporting files
COPY --chown=user:user tests ./tests/
COPY --chown=user:user models ./models/
COPY --chown=user:user setup.py .

# codecov looks at the git history
COPY --chown=user:user ./.git ./.git/

USER root
RUN trimesh-setup --install=test,gltf_validator,llvmpipe,binvox
USER user

# install things like pytest
RUN pip install `python setup.py --list-test`

# run pytest wrapped with xvfb for simple viewer tests
RUN xvfb-run pytest --cov=trimesh \
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

COPY --chown=user:user README.md .
COPY --chown=user:user docs ./docs/
COPY --chown=user:user examples ./examples/
COPY --chown=user:user models ./models/
COPY --chown=user:user trimesh ./trimesh/

WORKDIR /home/user/docs
RUN make

### Copy just the docs so we can output them
FROM scratch as docs
COPY --from=build_docs /home/user/docs/_build/html/ ./

### Make sure the output stage is the last stage so a simple
# "docker build ." still outputs an expected image
FROM output as final

