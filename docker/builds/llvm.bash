set -xe

apt-get update

apt-get -y --no-install-recommends install zlib1g zlib1g-dev  gzip tar wget git grep libtool autoconf automake libx11-dev libxext-dev libxcb1-dev bison flex make gcc g++ python python-mako pkg-config

mkdir -p /opt/
cd /opt 
wget https://cmake.org/files/v3.5/cmake-3.5.2-Linux-x86_64.tar.gz
tar -xvf cmake-3.5.2-Linux-x86_64.tar.gz
rm -f cmake-3.5.2-Linux-x86_64.tar.gz

# Build and install LLVM
cd /opt
wget http://llvm.org/releases/4.0.1/llvm-4.0.1.src.tar.xz
tar -xvf llvm-4.0.1.src.tar.xz
mkdir llvm-4.0.1.bld
cd llvm-4.0.1.bld
/opt/cmake-3.5.2-Linux-x86_64/bin/cmake \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX=/opt/llvm-4.0.1 \
	-DLLVM_BUILD_LLVM_DYLIB=ON \
	-DLLVM_LINK_LLVM_DYLIB=ON \
	-DLLVM_ENABLE_RTTI=ON \
	-DLLVM_TARGETS_TO_BUILD=X86 \
	-DLLVM_INSTALL_UTILS=ON \
	/opt/llvm-4.0.1.src
make -j$(grep -c "^processor" /proc/cpuinfo) install 
cd /opt 
rm -rf llvm-4.0.1.src.tar.xz \
   llvm-4.0.1.src llvm-4.0.1.bld \
   cmake-3.5.2-Linux-x86_64

apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
