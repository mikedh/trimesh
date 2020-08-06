FROM trimesh/trimesh:latest
MAINTAINER Michael Dawson-Haggerty <mikedh@kerfed.com>

# run supervisord as root
# supervisor will downgrade workers to user "user"
USER root

# eat less output
ENV PYTHONUNBUFFERED=0

# copy our example file which renders models
COPY render.py /home/user/render.py

# start XVFB in the background for workers that need X
# supervisord has config that will daemonize and continue
CMD ["supervisord", "/etc/supervisor/supervisord.conf &&", "python", "-u", "/home/user/render.py"]
