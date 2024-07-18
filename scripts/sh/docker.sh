#!/bin/zsh
# 获取今天的日期
# TODAY=$(date +%Y-%m-%d)

# docker login --username=aliyun2446412415 --password=QWertyuiop[ registry.cn-hangzhou.aliyuncs.com

# 定义镜像名称和容器名称
IMAGE_NAME="registry.cn-hangzhou.aliyuncs.com/slam_project/slam_practise_env:2024-07-04"
CONTAINER_NAME="code_utils"

# 检查镜像是否存在的函数
function check_image_exists {
    docker images -q $IMAGE_NAME
}

# 检查容器是否存在的函数
function check_container_exists {
    docker ps -aq -f name=$CONTAINER_NAME
}

# 构建Docker镜像的函数
function build_image {
    echo "正在构建Docker镜像..."
    docker build -t $IMAGE_NAME .
    if [ $? -ne 0 ]; then
        echo "Docker镜像构建失败!"
        exit 1
    fi
    echo "Docker镜像构建成功!"
}

# 创建Docker容器的函数
function create_container {
    echo "正在创建Docker容器..."
    xhost +local:docker
    # 如果宿主机提示cuda版本不支持驱动，就把cuda禁用，暂时先不用好了，或者修改dockerfile，使用nvidia-smi查看最高
    # 支持的cuda版本，然后修改对应的From img版本
    # -e NVIDIA_DISABLE_REQUIRE=1 \
    docker run \
        --network host \
        -p 6666:6666 \
        -e DISPLAY=$DISPLAY \
        -e QT_X11_NO_MITSHM=1 \
        -e XDG_RUNTIME_DIR=/tmp \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $HOME/.ssh:/root/.ssh \
        -v /home:/home \
        -v /mnt/e:/home/wq/data \
        -w /home \
        -itd --gpus all --name $CONTAINER_NAME $IMAGE_NAME zsh

    if [ $? -ne 0 ]; then
        echo "Docker容器创建失败!"
        exit 1
    fi
    echo "Docker容器创建成功!"
}

# 进入Docker容器的函数
function enter_container {
    echo "正在检查Docker容器状态..."
    if ! docker ps -f name=$CONTAINER_NAME --format '{{.Names}}' | grep -q $CONTAINER_NAME; then
        echo "容器未在运行中，正在启动容器..."
        docker start $CONTAINER_NAME
    fi
    echo "正在进入Docker容器..."
    if [ -d "build" ]; then
        docker exec -it $CONTAINER_NAME zsh -c "chsh -s /bin/zsh \
        && service ssh start \
        && git config --global --add safe.directory $(pwd) \
        && git config --global user.email "wqjuly@qq.com" \
        && git config --global user.name "wq"  \
        && cd $(pwd) && ./build/hello_world && zsh"
    else
        docker exec -it $CONTAINER_NAME zsh -c "chsh -s /bin/zsh \
        && service ssh start \
        && git config --global --add safe.directory $(pwd) \
        && git config --global user.email "wqjuly@qq.com" \
        && git config --global user.name "wq"  \
        && cd $(pwd) && zsh"
    fi
    echo ' ' | sudo -S chmod -R 777 .
}

# 处理命令行参数
if [ "$1" == "init" ]; then
    if [ $(check_container_exists) ]; then
        echo "正在删除现有的容器..."
        docker rm -f $CONTAINER_NAME
    fi

    if [ ! $(check_image_exists) ]; then
        echo "镜像不存在，创建镜像..."
        build_image
    fi
    create_container
    enter_container
else
    if [ ! $(check_container_exists) ]; then
        echo "容器不存在，创建容器..."
        create_container
    fi
    enter_container
fi
