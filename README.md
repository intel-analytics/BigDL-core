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

Build OpenCV, please refer the [doc](./opencv/README.md) for details.

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
