FROM trimesh/trimesh:latest
MAINTAINER Michael Dawson-Haggerty <mikedh@kerfed.com>

# copy our example file which renders a sphere
COPY render.py /home/user/render.py

# run supervisord as root
# supervisor will downgrade workers to user "user"
USER root

# eat less output
ENV PYTHONUNBUFFERED=0

# copy a supervisord config which just runs the render script
COPY render.supervisor.conf /etc/supervisor/conf.d/

# run our worker script AND xvfb using supervisord
# the xvfb supervisor config was copied in by trimesh
# the configs all live in /etc/supervisor/conf.d/
CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]
