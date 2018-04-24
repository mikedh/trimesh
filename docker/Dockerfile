FROM debian:stretch-slim
MAINTAINER Michael Dawson-Haggerty <mikedh@kerfed.com>

COPY builds/apt.bash /tmp/
RUN bash /tmp/apt.bash

# build draco and libspatialindex
COPY builds/draco.bash /tmp/
COPY builds/spatialindex.bash /tmp/
COPY builds/builds.bash /tmp/
RUN bash /tmp/builds.bash

# XVFB in background if you start supervisor
COPY config/xvfb.supervisord.conf /etc/supervisor/conf.d/

# switch out of root to install python env
RUN useradd -m -s /bin/bash user
USER user
COPY builds/conda.bash /tmp/
RUN bash /tmp/conda.bash
ENV PATH="/home/user/conda/bin:$PATH"
COPY config/requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# add user python to path 
ENV PATH="/home/user/conda/bin:$PATH"

# environment variables for software rendering
ENV XVFB_WHD="1920x1080x24"\
    DISPLAY=":99" \
    LIBGL_ALWAYS_SOFTWARE="1" \
    GALLIUM_DRIVER="llvmpipe"