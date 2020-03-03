# exit if anything fails
set -xe

apt-get update
# packages needed to build stuff
PACKAGES_BUILD="build-essential g++ gcc cmake"
apt-get -y --no-install-recommends install $PACKAGES_BUILD

# install draco, google's mesh compression utility
bash "$(dirname $0)/draco.bash"
# install VHACD, a mesh decomposition utility
bash "$(dirname $0)/vhacd.bash"

# remove build packages from image
apt-get remove -y --purge $PACKAGES_BUILD
# remove any orphaned packages
apt-get autoremove -y
# remove garbage
apt-get clean -y
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*



