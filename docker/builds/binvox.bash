set -xe

cd
# grab the binary
wget http://www.patrickmin.com/binvox/linux64/binvox?rnd=1560125291676946
# make it executable
chmod +x binvox
# move onto path
mv binvox /usr/bin/
