# packages required to build gpcio
#ENV PACKAGES="gcc g++ linux-headers build-base libc-dev make musl-dev"
#ENV PACKAGES="gcc g++ gfortran python3 python3-dev build-base wget freetype-dev libpng-dev openblas-dev"

export PACKAGES="gcc g++ gfortran build-base libc-dev make cmake git wget openblas-dev openssh python3-dev"
apk --no-cache add $PACKAGES

mkdir ~/.ssh
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

ash "$(dirname $0)/draco.sh"
ash "$(dirname $0)/spatialindex.sh"
ash "$(dirname $0)/geos.sh"

