FROM trimesh/trimesh-x:latest
MAINTAINER Michael Dawson-Haggerty <mikedh@kerfed.com>

# copy our example file which renders a sphere
COPY render.py .

# run our worker script using xvfb-run
CMD xvfb-run python render.py