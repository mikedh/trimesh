set -xe

apt-get update -y -qq
apt-get install -y -qq --no-install-recommends \
	wget bzip2 supervisor \
	libgl1-mesa-glx libgl1-mesa-dri xvfb xauth \
	libassimp-dev ca-certificates zstd unzip \
	freeglut3-dev git sudo \
	build-essential g++ gcc cmake

# install newer version of pandoc
wget https://github.com/jgm/pandoc/releases/download/2.9.2/pandoc-2.9.2-1-amd64.deb
echo "78525735ac6181f639c5c8776572d0ca10f0314c0052f5af2f369b5d0e1980b3  pandoc-2.9.2-1-amd64.deb" | sha256sum --check
sudo dpkg install 
rm -f pandoc-2.9.2-1-amd64.deb

# remove garbage
apt-get clean -qq
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
