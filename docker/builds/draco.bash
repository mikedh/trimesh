# install draco, google's mesh compression utility
set -xe

# remove any existing temporary build directories
rm -rf /tmp/draco_build
rm -rf /tmp/draco-1.3.5
rm -f /tmp/draco.zip

# fetch the archive from github releases
wget https://github.com/google/draco/archive/1.3.5.zip -O /tmp/draco.zip
cd /tmp
# sha256sum has an API designed by assholes
echo "7c45dcc085552f6d5202063eed9979762c4bf5efc20e9961bd4984bd730fb8d5  draco.zip" | sha256sum --check
# unzip
unzip draco.zip
# move to a clean directory to build
mkdir /tmp/draco_build
cd /tmp/draco_build

# actually build
cmake /tmp/draco-1.3.5
make
# move executables to our PATH
mv /tmp/draco_build/draco_encoder /usr/bin
mv /tmp/draco_build/draco_decoder /usr/bin
cd

rm -rf /tmp/draco_build
rm -rf /tmp/draco-1.3.5
rm -f /tmp/draco.zip
