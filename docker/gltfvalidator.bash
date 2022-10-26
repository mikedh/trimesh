set -xe

# download the binary from github releases and check hash
URL="https://github.com/KhronosGroup/glTF-Validator/releases/download/2.0.0-dev.3.8/gltf_validator-2.0.0-dev.3.8-linux64.tar.xz"
SHA="374c7807e28fe481b5075f3bb271f580ddfc0af3e930a0449be94ec2c1f6f49a"

# an old one before they added uint32 attribute fails
# URL="https://github.com/KhronosGroup/glTF-Validator/releases/download/2.0.0-dev.3.3/gltf_validator-2.0.0-dev.3.3-linux64.tar.xz"
# SHA="f807ebd35d46bb513cab88a920e63ac0c335b77dcf4b91cd8d09ea661b335bcd"

rm validator.tar.xz | true

wget $URL -O validator.tar.xz

# Check the hash of the downloaded file
echo "$SHA  validator.tar.xz" | sha256sum --check

# install binary
tar -xvf validator.tar.xz
chmod +x gltf_validator

mv gltf_validator /home/user/.local/bin

rm validator.tar.xz
