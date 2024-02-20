#### produce a zip file full of the GLTF 2.0 JSONschema
set -xe
# get the latest schema from github
wget https://github.com/KhronosGroup/glTF/archive/main.zip
unzip main.zip
# put just the schema into a ZIP file
# --junk-paths -j flattens directory structure
zip --junk-paths gltf2.schema.zip glTF-main/specification/2.0/schema/*.json
# remove the giant artifacts
rm -rf glTF-main
rm -f main.zip
