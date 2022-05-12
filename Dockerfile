FROM python:3.9-slim-bullseye

LABEL maintainer="mikedh@kerfed.com"

# where should this image install `trimesh`
ARG TRIMESH_PATH=/opt/trimesh

# Install the llvmpipe software renderer
# and X11 for software offscreen rendering,
# roughly 500mb of stuff.
ARG INCLUDE_X=false

# Install binary APT dependencies.
COPY docker/builds/apt.bash /tmp/
RUN bash /tmp/apt.bash ${INCLUDE_X}

# Install `embree`, Intel's fast ray checking engine
COPY docker/builds/embree.bash /tmp/
RUN bash /tmp/embree.bash

# XVFB runs in the background if you start supervisor.
COPY docker/config/xvfb.supervisord.conf /etc/supervisor/conf.d/

# Create a local non-root user.
RUN useradd -m -s /bin/bash user

# Copy minimal trimesh installation.
RUN mkdir -p "${TRIMESH_PATH}"
RUN chown user:user -R "${TRIMESH_PATH}"
COPY --chown=user:user trimesh/ "${TRIMESH_PATH}/trimesh"
COPY --chown=user:user tests/ "${TRIMESH_PATH}/tests"
COPY --chown=user:user setup.py "${TRIMESH_PATH}/"

# Switch to non-root user.
USER user

# Required for Python to be able to find libembree.
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
# So scripts installed from pip are in $PATH
ENV PATH="/home/user/.local/bin:$PATH"

# Install trimesh as `user`.
# `supervisor` is for running xvfb
# `cython` is to install embree
RUN pip install --no-cache-dir -e "${TRIMESH_PATH}[all]" cython supervisor
RUN pip install https://github.com/scopatz/pyembree/releases/download/0.1.6/pyembree-0.1.6.tar.gz

# Set environment variables for software rendering.
ENV XVFB_WHD="1920x1080x24"\
    DISPLAY=":99" \
    LIBGL_ALWAYS_SOFTWARE="1" \
    GALLIUM_DRIVER="llvmpipe"
