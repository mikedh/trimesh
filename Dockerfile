FROM python:3.10-slim-bullseye AS trimesh
LABEL maintainer="mikedh@kerfed.com"

# Install the llvmpipe software renderer
# and X11 for software offscreen rendering,
# roughly 500mb of stuff.
ARG INCLUDE_X=false

# Install binary APT dependencies.
COPY docker/builds/apt-trimesh /usr/local/bin/
RUN apt-trimesh --base=true --x11=${INCLUDE_X}

# Install `embree`, Intel's fast ray checking engine
COPY docker/builds/embree.bash /tmp/
RUN bash /tmp/embree.bash

# Create a local non-root user.
RUN useradd -m -s /bin/bash user

# Required for Python to be able to find libembree.
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
# So scripts installed from pip are in $PATH
ENV PATH="/home/user/.local/bin:$PATH"

## install things that need building
FROM trimesh AS build

# install build-essentials
RUN apt-trimesh --build=true

# copy in essential files
RUN mkdir -p /tmp/trimesh
RUN chown user:user /tmp/trimesh
COPY --chown=user:user trimesh/ /tmp/trimesh/trimesh
COPY --chown=user:user tests/ /tmp/trimesh/tests
COPY --chown=user:user setup.py /tmp/trimesh/

# switch to non-root user
USER user

# install trimesh into .local
RUN pip install /tmp/trimesh[all]
RUN pip install https://github.com/scopatz/pyembree/releases/download/0.1.6/pyembree-0.1.6.tar.gz

FROM trimesh AS output

USER user

# just copy over the build
COPY --from=build /home/user/.local /home/user/.local

# Set environment variables for software rendering.
ENV XVFB_WHD="1920x1080x24"\
    DISPLAY=":99" \
    LIBGL_ALWAYS_SOFTWARE="1" \
    GALLIUM_DRIVER="llvmpipe"
