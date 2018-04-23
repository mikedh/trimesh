apt-get update
PACKAGES_BUILD="git ca-certificates build-essential g++ gcc cmake"
apt-get -y --no-install-recommends install $PACKAGES_BUILD

# install draco, google's mesh compression utility
bash "$(dirname $0)/draco.bash"
# for rtree
bash "$(dirname $0)/spatialindex.bash"

# remove garbage
#apt-get remove --auto-remove --purge -y $PACKAGES_BUILD
apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*



