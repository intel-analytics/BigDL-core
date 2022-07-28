## Opencv 

### Prepare OpenCV 4.2.0
Download [OpenCV 4.2.0](https://github.com/opencv/opencv/archive/4.2.0.tar.gz) and build OpenCV.

```bash
# Download source code
wget https://github.com/opencv/opencv/archive/4.2.0.tar.gz

# Install JAVA and Apache Ant
export JAVA_HOME=..
yum install -y ant

# Build OpenCV
tar -zxvf 4.2.0.tar.gz && cd opencv-4.2.0

mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D BUILD_DOCS=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF -DWITH_1394=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_OPENCL=OFF -DOPENCV_DNN_OPENCL=OFF -DWITH_PNG=ON -DWITH_MKL=OFF -DMKL_WITH_TBB=OFF -DMKL_USE_MULTITHREAD=OFF -DMKL_WITH_OPENMP=OFF -DWITH_ITT=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_DSHOW=OFF -DWITH_MSMF=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_IPP_IW=OFF -DBUILD_ITT=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_calib3d=OFF -DBUILD_opencv_dnn=OFF -DBUILD_opencv_features2d=OFF -DBUILD_opencv_flann=OFF -DBUILD_opencv_gapi=OFF -DBUILD_opencv_highgui=OFF -DBUILD_opencv_imgcodecs=ON -DBUILD_opencv_imgproc=ON -DBUILD_opencv_java=ON -DBUILD_opencv_java_bindings_generator=ON -DBUILD_opencv_js=OFF -DBUILD_opencv_ml=OFF -DBUILD_opencv_objdetect=OFF -DBUILD_opencv_photo=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_python_bindings_generator=OFF -DBUILD_opencv_python_tests=OFF -DBUILD_opencv_stitching=OFF -DBUILD_opencv_video=OFF -DBUILD_opencv_videoio=OFF -DBUILD_PROTOBUF=OFF -DWITH_IPP=OFF -DWITH_GTK=OFF -DWITH_OPENEXR=OFF -DWITH_JASPER=OFF -DWITH_WEBP=OFF -DWITH_OPENCL=OFF -DWITH_V4L=OFF -DWITH_VTK=OFF -DWITH_TIFF=OFF ..

# build
make

# Copy jar & libs to /opt/opencv
mkdir /opt/opencv
cp ./bin/opencv-420.jar /opt/opencv
cp ./lib/libopencv_core.so.4.2.0  /opt/opencv 
cp ./lib/libopencv_imgcodecs.so.4.2.0  /opt/opencv
cp ./lib/libopencv_imgproc.so.4.2.0  /opt/opencv 
cp ./lib/libopencv_java420.so /opt/opencv

# Change to libopencv_java420.dylib in MacOS
cp ./bin/opencv-420.jar /opt/opencv
cp ./lib/libopencv_core.4.2.0.dylib  /opt/opencv
cp ./lib/libopencv_imgcodecs.4.2.0.dylib  /opt/opencv
cp ./lib/libopencv_imgproc.4.2.0.dylib  /opt/opencv
cp ./lib/libopencv_java420.dylib /opt/opencv

cd ../.. && rm -rf 4.2.0.tar.gz
```
### Known issues
As OpenCV will detect and load MKL, building OpenCV is preferred on a clean environment(with MKL).
