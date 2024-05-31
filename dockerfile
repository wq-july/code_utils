# Use the NVIDIA CUDA 12.1.0 base image with Ubuntu 20.04
FROM nvidia/cuda:12.1.0-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# 更换清华源
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# Install essential packages
RUN apt-get update \
    && apt upgrade -y \ 
    && apt install -y \
    build-essential \
    wget \
    curl \
    zsh \
    ssh \
    git \
    cmake \
    vim \
    libeigen3-dev \
    libgl1-mesa-glx \
    libx11-dev \
    libgtk2.0-dev \
    libglew-dev \
    libboost-all-dev \
    freeglut3-dev \
    libyaml-cpp-dev \
    pkg-config \
    libvtk7-dev \
    libusb-1.0-0-dev \
    libmetis-dev \
    libspdlog-dev \
    libsuitesparse-dev \
    qtdeclarative5-dev \
    qt5-qmake \
    libqglviewer-dev-qt5 \
    libgoogle-glog-dev \
    libgflags-dev \
    libgtest-dev \
    libatlas-base-dev \
    libpcl-dev \
    python3-pip \
    python3-tk \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    libwayland-dev \
    libxkbcommon-dev \
    wayland-protocols \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libepoxy-dev \
    libc++-dev \
    g++ \
    ninja-build \
    libjpeg-dev \
    libpng-dev \
    libavcodec-dev \
    libavutil-dev \
    libavformat-dev \
    libswscale-dev \
    libavdevice-dev \
    libdc1394-dev \
    libraw1394-dev \
    libopenni-dev \
    python3-dev \
    python3-distutils \
    && apt-get install -f \
    && apt-get install -y libopencv-dev \
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
    && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install \
    && cd ../.. \
    && rm -rf fmt

# Install latest Sophus 1.22.10
RUN git clone https://github.com/strasdat/Sophus.git \
    && cd Sophus \
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
