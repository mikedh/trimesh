# update local files to latest version from main github
set -xe

# version of three.js to pull files from
VERSION=115

rm three.min.js | true
wget https://raw.githubusercontent.com/mrdoob/three.js/r${VERSION}/build/three.min.js

rm TrackballControls.js | true
wget https://raw.githubusercontent.com/mrdoob/three.js/r${VERSION}/examples/js/controls/TrackballControls.js

rm GLTFLoader.js | true
wget https://raw.githubusercontent.com/mrdoob/three.js/r${VERSION}/examples/js/loaders/GLTFLoader.js
