# update local files to latest version from main github
set -xe

rm TrackballControls.js
wget https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/js/controls/TrackballControls.js

rm GLTFLoader.js
wget https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/js/loaders/GLTFLoader.js
