from ubuntu:16.04

maintainer Dockerfiles

# copy this container directory to the filesystem
add . /home/docker/config

# graph-tool adds a ton of dependancies (including an X server)
# but speeds up a very common operation by 10-20x over networkx
run cp /home/docker/config/graphtool.list /etc/apt/sources.list.d/
run apt-key add /home/docker/config/graphtool.key
run apt-get update
run apt-get install -y `cat /home/docker/config/apt.list`

run pip3 install --upgrade pip
run pip3 install trimesh[all] --upgrade
