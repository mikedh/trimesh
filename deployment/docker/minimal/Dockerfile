FROM alpine:3.4
MAINTAINER Mike Dawson-Haggerty <mikedh@kerfed.com>

# packages required to build gpcio
#ENV PACKAGES="gcc g++ linux-headers build-base libc-dev make musl-dev"
ENV PACKAGES="gcc g++ gfortran python3 python3-dev build-base wget freetype-dev libpng-dev openblas-dev"

RUN echo "http://dl-8.alpinelinux.org/alpine/edge/community" >> /etc/apk/repositories
RUN apk --no-cache --update-cache add  $PACKAGES
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

# copy over our python requirements
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir --upgrade pip setuptools
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

