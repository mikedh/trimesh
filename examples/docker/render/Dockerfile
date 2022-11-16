FROM trimesh/trimesh:latest
MAINTAINER Michael Dawson-Haggerty <mikedh@kerfed.com>

USER root
# this is a helper script that installs the
# debian packages required for offscreen rendering
# RUN apt-trimesh --x11=true
RUN apt-get update && apt-get install -y xvfb
# go back to the unprivileged user
USER user

# copy our example file which renders a sphere
COPY render.py .

# run our worker script using xvfb-run
CMD xvfb-run python render.py