FROM debian:stretch-slim

COPY builds/apt.bash /tmp/
RUN bash /tmp/apt.bash

COPY builds/draco.bash /tmp/
COPY builds/spatialindex.bash /tmp/
COPY builds/builds.bash /tmp/
RUN bash /tmp/builds.bash

# switch out of root space to install python env
RUN useradd -m -s /bin/bash user
USER user
COPY builds/conda.bash /tmp/
RUN bash /tmp/conda.bash
ENV PATH="/home/user/conda/bin:$PATH"
COPY config/requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

COPY . /home/user/mf

# back to root for supervisor
USER root
COPY config/xvfb.supervisord.conf /etc/supervisor/conf.d/

# Setup our environment variables.
ENV XVFB_WHD="1920x1080x24"\
    DISPLAY=":99" \
    LIBGL_ALWAYS_SOFTWARE="1" \
    GALLIUM_DRIVER="llvmpipe" \
    #GALLIUM_DRIVER="swr" \
    LP_NO_RAST="false" \
    LP_DEBUG="" \
    LP_PERF="" \
    LP_NUM_THREADS=""

# run the default supervisord.conf
# it will run all of the things in conf.d
CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]