# Installing OpenCV on WSL:
## Get git repo:
https://github.com/opencv/opencv/
https://github.com/opencv/opencv_contrib

go and checkout both repos to some version - e.g.: git checkout 4.9.0

## Build WITH GUI:
### Install necessary GUI libs:
sudo apt install pkg-config libgtk-3-dev

### Install CUDA to WSL or check installed version:
https://dev.to/naruaika/using-opencv-on-windows-subsystem-for-linux-1ako

and check version: `nvidia-smi.exe`


### Build OpenCV
Possible arg for build:
  - -D OPENCV_EXTRA_MODULES_PATH=~/opencv_with_contrib/opencv_contrib/modules
  - -D CUDNN_LIBRARY=/usr/lib64/libcudnn.so.8 
  - -D CUDNN_INCLUDE_DIR=/usr/include
  - -D PYTHON_EXECUTABLE=$HOME/Researches/venv/bin/python 
  - -D PYTHON3_NUMPY_INCLUDE_DIRS=$HOME/Researches/venv/lib64/python3.8/site-packages/numpy/core/include
  - -D CMAKE_INSTALL_PREFIX=$HOME/Researches/venv

#### Find your version of CUDA capability for NVidia GPU:
https://developer.nvidia.com/cuda-gpus

  - RTX 20x0: 7.5
  - RTX 30x0: 8.6

Set the right opencv_contrib path

cmake -D CMAKE_BUILD_TYPE=RELEASE -D OPENCV_GENERATE_PKGCONFIG=ON -D ENABLE_PRECOMPILED_HEADERS=OFF -D BUILD_opencv_legacy=OFF -D CUDA_ARCH_BIN=7.5 -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D OPENCV_ENABLE_NONFREE=ON -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ../opencv
make -j4 #increasing the number will make building faster. Maximum value can be found by running nproc.