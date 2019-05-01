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

# install a conda env and trimesh
COPY docker/builds/conda.bash /tmp/
# copy local trimesh for install script
COPY . /tmp/trimesh

# do conda and trimesh install
RUN bash /tmp/conda.bash

# add user python to path 
ENV PATH="/home/user/conda/bin:$PATH"

# environment variables for software rendering
ENV XVFB_WHD="1920x1080x24"\
    DISPLAY=":99" \
    LIBGL_ALWAYS_SOFTWARE="1" \
    GALLIUM_DRIVER="llvmpipe"

# make sure build fails if tests are failing
# -p no:warnings suppresses 10,000 useless upstream warnings
# -p no:alldep means that tests will fail if a dependancy is missing
RUN pytest -p no:warnings -p no:alldep /tmp/trimesh/tests