# exit early
set -xe

# use our built LLVM, for some reason
export PATH=/opt/llvm-4.0.1/bin:$PATH

apt-get update
apt-get install -y libxcb-dri2-0-dev libxcb-xfixes0-dev libx11-xcb-dev libexpat1-dev libexpat1

cd /opt
wget --no-check-certificate https://mesa.freedesktop.org/archive/mesa-18.0.1.tar.gz
tar -zxvf mesa-18.0.1.tar.gz
cd mesa-18.0.1

NOCONFIGURE=1 ./autogen.sh
 
##########
# Option summary:
##########

# Enable GL APIs
#  --enable-opengl --disable-gles1 --disable-gles2

# Disable extra state trackers that we don't care about
# --disable-va --disable-gbm --disable-xvmc --disable-vdpau

# Turn on GLdispatch
# --enable-shared-glapi

# Set up desired library features
# --disable-texture-float

# Turn off DRI (we're not using any of it)
# --disable-dri --with-dri-drivers=

# Turn on the Gallium infrastructure
# --enable-gallium-llvm

# Use LLVM's shared libraries so we don't bloat the binaries too much
# --enable-llvm-shared-libs

# Turn on only software rasterizers
# --with-gallium-drivers=swrast,swr

# Turn off EGL
# --disable-egl --disable-gbm  --with-egl-platforms=

# Turn on Gallium based OSMesa
# --enable-gallium-osmesa

# Turn on GLX (auto-determined backend)
# --enable-glx

# Setup flags for LTO to shrink resultng binaries
./configure \
    --enable-opengl --disable-gles1 --disable-gles2           \
    --disable-va --disable-gbm --disable-xvmc --disable-vdpau \
    --enable-shared-glapi                                     \
    --disable-texture-float                                   \
    --disable-dri --with-dri-drivers=                         \
    --enable-llvm --enable-llvm-shared-libs                   \
    --with-gallium-drivers=swrast,swr                         \
    --disable-egl --disable-gbm --with-platforms=x11          \
    --enable-gallium-osmesa                                   \
    --enable-glx                                              \
    --prefix=/opt/mesa-18.0.1 \
    --with-llvm-prefix=/opt/llvm-4.0.1
make -j$(grep -c "^processor" /proc/cpuinfo)
#make install
cd /opt

apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
