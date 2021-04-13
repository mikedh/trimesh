set -xe

# download the binary from github releases
rm /tmp/validator.tar.xz | true
wget https://github.com/KhronosGroup/glTF-Validator/releases/download/2.0.0-dev.3.3/gltf_validator-2.0.0-dev.3.3-linux64.tar.xz  -O /tmp/validator.tar.xz

cd /tmp

# Check the hash of the downloaded file
echo "f807ebd35d46bb513cab88a920e63ac0c335b77dcf4b91cd8d09ea661b335bcd  validator.tar.xz" | sha256sum --check

# install binary
tar -xvf validator.tar.xz
chmod +x gltf_validator
mv gltf_validator /usr/bin
rm validator.tar.xz
