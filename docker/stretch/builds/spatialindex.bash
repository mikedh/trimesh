git clone https://github.com/libspatialindex/libspatialindex.git
cd libspatialindex
git checkout afabefc21d7f486a50089db306d82152aa8cc6a7
cmake .
make
make install
cd ..
rm -rf libspatialindex
