#### produce a zip file full of the GLTF 2.0 JSONschema
set -xe
# get the latest schema from github
wget https://github.com/KhronosGroup/glTF/archive/master.zip
unzip master.zip
# put just the schema into a ZIP file
# --junk-paths -j flattens directory structure
zip --junk-paths gltf_2_schema.zip glTF-master/specification/2.0/schema/*.json
# remove the giant artifacts
rm -rf glTF-master
rm -f master.zip
mv gltf_2_schema.zip ..
