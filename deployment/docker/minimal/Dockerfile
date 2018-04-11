FROM alpine:3.7
MAINTAINER Mike Dawson-Haggerty <mikedh@kerfed.com>

# get python 3
# based on frolvlad/alpine-python3
RUN apk add --no-cache python3 && \
    python3 -m ensurepip && \
    rm -r /usr/lib/python*/ensurepip && \
    pip3 install --upgrade pip setuptools && \
    if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi && \
    if [[ ! -e /usr/bin/python ]]; then ln -sf /usr/bin/python3 /usr/bin/python; fi && \
    rm -r /root/.cache

# make sure pip and setuptools are updated
RUN pip3 install --no-cache-dir --upgrade pip setuptools

# run all the built stuff
COPY build.sh /tmp/
RUN ash /tmp/build.sh

# copy over our python requirements
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir --upgrade -r /tmp/requirements.txt

