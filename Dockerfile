FROM debian:buster-slim
LABEL maintainer="mikedh@kerfed.com"

COPY docker/builds/apt.bash /tmp/
RUN bash /tmp/apt.bash

# copy compile recipies to build draco and download vhacd
COPY docker/builds/draco.bash /tmp/
COPY docker/builds/vhacd.bash /tmp/
RUN bash /tmp/draco.bash && bash /tmp/vhacd.bash

# XVFB in background if you start supervisor
COPY docker/config/xvfb.supervisord.conf /etc/supervisor/conf.d/

# switch out of root
RUN useradd -m -s /bin/bash user

# copy local trimesh for install and tests
COPY --chown=user:user . /tmp/trimesh

# switch to user
USER user

# install a conda env and trimesh
RUN bash /tmp/trimesh/docker/builds/conda.bash

USER user

# add user python to path 
ENV PATH="/home/user/conda/bin:$PATH"

# make sure build fails if tests are failing
# -p no:warnings suppresses 10,000 useless upstream warnings
# -p no:alldep means that tests will fail if a dependency is missing
# -x will exit on first test failure
RUN pytest -x -p no:warnings -p no:alldep /tmp/trimesh/tests

# environment variables for software rendering
ENV XVFB_WHD="1920x1080x24"\
    DISPLAY=":99" \
    LIBGL_ALWAYS_SOFTWARE="1" \
    GALLIUM_DRIVER="llvmpipe"
