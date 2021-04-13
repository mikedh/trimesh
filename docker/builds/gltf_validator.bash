set -xe

rm /tmp/validator.tar.xz | true
wget https://github.com/KhronosGroup/glTF-Validator/releases/download/2.0.0-dev.3.3/gltf_validator-2.0.0-dev.3.3-linux64.tar.xz  -O /tmp/validator.tar.xz


cd /tmp
tar -xvf validator.tar.xz
chmod +x gltf_validator
mv gltf_validator /usr/bin
rm validator.tar.xz
