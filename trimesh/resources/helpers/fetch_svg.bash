#### produce a zip file full of the GLTF 2.0 JSONschema
set -xe
# get the latest schema from github
wget https://www.w3.org/Graphics/SVG/1.1/DTD/svg11-flat-20110816.dtd -O svg.dtd
# put just the schema into a ZIP file
# --junk-paths -j flattens directory structure
zip -9 svg.dtd.zip svg.dtd
# remove the artifacts
rm -f svg.dtd
mv svg.dtd.zip ..
