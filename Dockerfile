FROM python:3.10-slim-bullseye AS base
LABEL maintainer="mikedh@kerfed.com"

# Install the llvmpipe software renderer
# and X11 for software offscreen rendering,
# roughly 500mb of stuff.
ARG INCLUDE_X=false

# Install binary APT dependencies.
COPY --chmod=755 docker/apt-trimesh /usr/local/bin/
RUN apt-trimesh --base=true --x11=${INCLUDE_X}

# Install `embree`, Intel's fast ray checking engine
COPY docker/embree.bash /tmp/
RUN bash /tmp/embree.bash

# Create a local non-root user.
RUN useradd -m -s /bin/bash user

# Required for Python to be able to find libembree.
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
# So scripts installed from pip are in $PATH
ENV PATH="/home/user/.local/bin:$PATH"

## install things that need building
FROM base AS build

# install build-essentials
RUN apt-trimesh --build=true

# copy in essential files
COPY --chown=user:user trimesh/ /home/user/trimesh
COPY --chown=user:user setup.py /home/user/

# switch to non-root user
USER user

# install trimesh into .local
RUN pip install /home/user[all]
RUN pip install https://github.com/scopatz/pyembree/releases/download/0.1.6/pyembree-0.1.6.tar.gz

####################################
### Build output image most things should run on
FROM base AS output

# switch to non-root user
USER user
WORKDIR /home/user

# just copy over the results of the pip installs
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
COPY --chown=user:user docker/gltfvalidator.bash .
COPY --chown=user:user ./.git ./.git/

# install the khronos GLTF validator
RUN bash gltfvalidator.bash

# install things like pytest
RUN pip install `python setup.py --list-test`

# run tests
RUN pytest --cov=trimesh \
    -p no:alldep \
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
RUN apt-trimesh --docs=true
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
