# BigDL-core

This repo kept C++ & JNI code for [BigDL](https://github.com/intel-analytics/BigDL). Modules have been added to BigDL automatically. Please avoid reference this repo in dependency.

## Requirements

**Common Requirements:**

```bash
ICC 17
JDK 1.7
cmake 3.10+
maven 3.3.9+
Intel MKL
```

ICC and Intel MKL were packaged in `parallel_studio_xe_XXX`. Now they are available in [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html).

**Linux**

```bash
OS: CentOS 6.5/Ubuntu 16.04 or later
```

**MacOS**

```bash
MacOS 10.13
XCode 9.0
```

## Building from Source

### Prepare OpenCV 4.2.0

Download `opencv-420.jar` from maven. Copy this jar to `/opt/opencv` dir. Extract `libopencv_java420.so` (Linux) or `libopencv_java420.dylib` (macOS) from `opencv-420.jar`, and move to `/opt/opencv`.

Or you can build this 2 files from OpenCV source code.

```bash
# Download source code
wget https://github.com/opencv/opencv/archive/4.2.0.tar.gz
# Install JAVA and Apache Ant
export JAVA_HOME=..
yum install -y ant
# Build OpenCV
tar -zxvf 4.2.0.tar.gz && cd opencv-4.2.0
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -DBUILD_TESTS=OFF ..
make && make install
# Copy jar & libs to /opt/opencv
mkdir /opt/opencv
cp ./bin/opencv-420.jar /opt/opencv
# Change to libopencv_java420.dylib in MacOS
cp ./lib/libopencv_java420.so /opt/opencv
cd ../.. && rm -rf 4.2.0.tar.gz
```

### Build BigDL-Core

Download or clone source code

```bash
git clone --recursive https://github.com/intel-analytics/BigDL-core.git
```

Build for Linux

```bash
cd BigDL-core
mvn -B clean package -P linux
```

Build for MacOS

```bash
cd BigDL-core
mvn -B clean package -P mac
```

## Know issues

1. ICC has been merged into [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html).
2. Avoid building BigDL-core on MacOS 10.14. Because Apple made some change in LD_LIBRARY.
