# Use the NVIDIA CUDA 12.1.0 base image with Ubuntu 20.04
FROM registry.cn-hangzhou.aliyuncs.com/slam_project/cuda:12.3.2-cudnn9-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# 更换清华源
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# Install essential packages
RUN apt-get update \
    && apt-get upgrade -y \ 
    && apt-get install -y \
    build-essential \
    checkinstall \
    cmake \
    curl \
    doxygen \
    freeglut3-dev \
    g++ \
    gfortran \
    git \
    libatlas-base-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavformat-dev \
    libavresample-dev \
    libavutil-dev \
    libboost-all-dev \
    libdc1394-22 \
    libdc1394-22-dev \
    libeigen3-dev \
    libepoxy-dev \
    libfaac-dev \
    libgflags-dev \
    libglew-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglib2.0-dev \
    libgphoto2-dev \
    libgoogle-glog-dev \
    libgphoto2-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libhdf5-dev \
    libjpeg-dev \
    libmetis-dev \
    libmp3lame-dev \
    libopenexr-dev \
    libopenni-dev \
    libpcl-dev \
    libpng-dev \
    libprotobuf-dev \
    libpq-dev \
    libqglviewer-dev-qt5 \
    libraw1394-dev \
    libsuitesparse-dev \
    libswscale-dev \
    libtheora-dev \
    libusb-1.0-0-dev \
    libvtk7-dev \
    libv4l-dev \
    libvorbis-dev \
    libwayland-dev \
    libx11-dev \
    libx264-dev \
    libxine2-dev \
    libxkbcommon-dev \
    libxvidcore-dev \
    ninja-build \
    ocl-icd-libopencl1 \
    opencl-headers \
    pkg-config \
    protobuf-compiler \
    python3-dev \
    python3-distutils \
    python3-pip \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-tk \
    python3-wstool \
    qt5-qmake \
    qtdeclarative5-dev \
    ssh \
    unzip \
    vim \
    v4l-utils \
    wayland-protocols \
    wget \
    x264 \
    yasm \
    zsh \
    && apt-get install -f \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# python3设置 pip 使用清华大学的镜像源
RUN mkdir -p ~/.pip && \
    echo "[global]" > ~/.pip/pip.conf && \
    echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> ~/.pip/pip.conf

RUN pip3 install --upgrade pip \
    && pip3 install numpy scipy matplotlib rosdepc

# Install fmt v0.11.0 (required by Sophus)
RUN git clone https://github.com/fmtlib/fmt.git \
    && cd fmt \
    && git checkout 10.2.1 \
    && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install \
    && cd ../.. \
    && rm -rf fmt

# Install latest Sophus 1.22.10
RUN git clone https://github.com/strasdat/Sophus.git \
    && cd Sophus \
    && git checkout 1.22.10 \
    && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install \
    && cd ../.. \
    && rm -rf Sophus

# Install robin-map v1.3.0 
RUN git clone https://github.com/Tessil/robin-map.git \
    && cd robin-map \
    && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install \
    && cd ../.. \
    && rm -rf robin-map

# Install g2o 20230806_git
RUN git clone https://github.com/RainerKuemmerle/g2o.git \
    && cd g2o \
    && mkdir build && cd build \
    && cmake .. -DBUILD_WITH_MARCH_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && make install \
    && cd ../.. \
    && rm -rf g2o

# Install Ceres 2.2.0
RUN git clone https://github.com/ceres-solver/ceres-solver.git \
    && cd ceres-solver \
    && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install \
    && cd ../.. \
    && rm -rf ceres-solver

# 使用Ninja构建，更加快速
RUN git clone --recursive https://github.com/stevenlovegrove/Pangolin.git \
    && cd Pangolin \
    && cmake -B build -G Ninja \
    && cmake --build build/ --target install \
    && cd .. \
    && rm -rf Pangolin

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.10.0.zip \
    && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.10.0.zip \
    && unzip opencv.zip \
    && unzip opencv_contrib.zip \
    && mv opencv_contrib-4.10.0 opencv-4.10.0 \
    && cd opencv-4.10.0 \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
             -D INSTALL_PYTHON_EXAMPLES=OFF \
             -D INSTALL_C_EXAMPLES=OFF \
             -D WITH_TBB=ON \
             -D WITH_CUDA=ON \
             -D BUILD_opencv_cudacodec=OFF \
             -D ENABLE_FAST_MATH=1 \
             -D CUDA_FAST_MATH=1 \
             -D WITH_CUBLAS=1 \
             -D WITH_V4L=OFF \
             -D WITH_LIBV4L=ON \
             -D WITH_QT=OFF \
             -D WITH_GTK=ON \
             -D WITH_GTK_2_X=ON \
             -D WITH_EIGEN=ON \
             -D WITH_OPENGL=ON \
             -D WITH_GSTREAMER=ON \
             -D OPENCV_GENERATE_PKGCONFIG=ON \
             -D OPENCV_PC_FILE_NAME=opencv.pc \
             -D OPENCV_ENABLE_NONFREE=ON \
             -D CUDA_nppicom_LIBRARY=stdc++ \
             -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.10.0/modules \
             -D BUILD_EXAMPLES=OFF .. \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd ../.. \
    && rm -rf opencv.zip opencv_contrib.zip opencv-4.10.0

# Install oh-my-zsh
# Uses "Spaceship" theme with some customization. Uses some bundled plugins and installs some more from github
# Uses "git", "ssh-agent" and "history-substring-search" bundled plugins
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
-t ys \
-p git \
-p docker \
-p ssh-agent \
-p 'history-substring-search' \
-p https://github.com/zsh-users/zsh-autosuggestions \
-p https://github.com/zsh-users/zsh-completions \
-p https://github.com/zsh-users/zsh-syntax-highlighting \
-x

RUN chsh -s /bin/zsh

# ROS-Noetic
RUN rm -rf /etc/apt/sources.list.d/ros-latest.list \
    && sh -c 'echo "deb https://mirrors.tuna.tsinghua.edu.cn/ros/ubuntu/ $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && cd /var/lib/dpkg \
    && rm -rf info \
    && mkdir info \
    && cd ~ \
    && apt-get update \
    && apt-get upgrade -y \
    && apt install ros-noetic-desktop-full -y \
    && echo "source /opt/ros/noetic/setup.zsh" >> ~/.zshrc

# 设置 SHELL 为 zsh
SHELL ["/bin/zsh", "-c"]

# 初始化 rosdep
RUN source ~/.zshrc \
    && rm -rf /etc/ros/rosdep/sources.list.d/20-default.list \
    # 其中rosdepc，c指的是China中国，主要用于和rosdep区分。
    && rosdepc init \
    && rosdepc update

# Clean up
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置 DISPLAY 环境变量
ENV DISPLAY=:0

# Default command
CMD ["zsh"]
