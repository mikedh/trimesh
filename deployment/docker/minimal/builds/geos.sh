git clone https://github.com/OSGeo/geos.git
git checkout b40bd637f56242d5d4ad5ef2bf6a3ec12b328645
cd geos
cmake .
make
make install
ldconfig
cd ..
rm -rf geos
