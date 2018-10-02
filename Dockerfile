FROM debian:stretch-slim
MAINTAINER Michael Dawson-Haggerty <mikedh@kerfed.com>

COPY docker/builds/apt.bash /tmp/
RUN bash /tmp/apt.bash

# build draco and download vhacd
COPY docker/builds/draco.bash /tmp/
COPY docker/builds/vhacd.bash /tmp/
COPY docker/builds/builds.bash /tmp/
RUN bash /tmp/builds.bash

# XVFB in background if you start supervisor
COPY docker/config/xvfb.supervisord.conf /etc/supervisor/conf.d/

# switch out of root 
RUN useradd -m -s /bin/bash user
RUN chown -R user:user /tmp
USER user

# install a conda env
COPY docker/builds/conda.bash /tmp/
RUN bash /tmp/conda.bash

# install python requirements
COPY . /tmp/trimesh
RUN /home/user/conda/bin/pip install /tmp/trimesh[all] pytest

# add user python to path 
ENV PATH="/home/user/conda/bin:$PATH"

# environment variables for software rendering
ENV XVFB_WHD="1920x1080x24"\
    DISPLAY=":99" \
    LIBGL_ALWAYS_SOFTWARE="1" \
    GALLIUM_DRIVER="llvmpipe"

# make sure build fails if tests are failing
RUN pytest /tmp/trimesh/tests