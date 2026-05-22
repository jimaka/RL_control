# =============================================================================
#  RL-Actor + MPC-Shield  ::  Dockerfile
#  Base : CUDA 11.8 + cuDNN8 (Ubuntu 20.04)  +  ROS Noetic
#  Goal : One image for both Python (training/Optuna) and C++ (ROS shield node)
# =============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=Asia/Shanghai
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# -----------------------------------------------------------------------------
# 1. System packages
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl wget gnupg2 lsb-release ca-certificates software-properties-common \
        build-essential cmake git pkg-config \
        python3 python3-pip python3-dev python3-venv \
        libeigen3-dev libgoogle-glog-dev libgflags-dev \
        libspdlog-dev libfmt-dev \
        libgl1 libglib2.0-0 \
        unzip vim less htop tmux \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. ROS Noetic (desktop-full, on top of CUDA image)
# -----------------------------------------------------------------------------
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
        > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
        | apt-key add - \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-noetic-desktop-full \
        ros-noetic-tf ros-noetic-tf2 ros-noetic-tf2-ros \
        ros-noetic-dynamic-reconfigure \
        ros-noetic-rosbridge-suite \
        python3-rosdep python3-rosinstall python3-rosinstall-generator \
        python3-wstool python3-catkin-tools \
    && rosdep init && rosdep update \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 3. OSQP  (C++ embedded convex QP solver, used by the MPC-Shield)
# -----------------------------------------------------------------------------
RUN cd /tmp && git clone --recursive -b v0.6.3 https://github.com/osqp/osqp.git \
    && cd osqp && mkdir build && cd build \
    && cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release .. \
    && cmake --build . --target install -- -j$(nproc) \
    && cd /tmp && git clone https://github.com/robotology/osqp-eigen.git \
    && cd osqp-eigen && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make -j$(nproc) && make install \
    && rm -rf /tmp/osqp /tmp/osqp-eigen \
    && ldconfig

# -----------------------------------------------------------------------------
# 4. ONNX Runtime  (C++ API, GPU build matching CUDA 11.8)
# -----------------------------------------------------------------------------
ARG ORT_VERSION=1.16.3
RUN cd /opt && \
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz && \
    tar -xzf onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz && \
    mv onnxruntime-linux-x64-gpu-${ORT_VERSION} onnxruntime && \
    rm onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz
ENV ONNXRUNTIME_ROOT_PATH=/opt/onnxruntime
ENV LD_LIBRARY_PATH=${ONNXRUNTIME_ROOT_PATH}/lib:${LD_LIBRARY_PATH}

# -----------------------------------------------------------------------------
# 5. Python ML stack  (PyTorch CUDA 11.8 + SB3 + Optuna + viz)
# -----------------------------------------------------------------------------
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

RUN pip3 install --no-cache-dir \
        torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
        --extra-index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --no-cache-dir \
        numpy==1.24.4 scipy==1.10.1 pandas==2.0.3 \
        gymnasium==0.29.1 \
        stable-baselines3==2.2.1 sb3-contrib==2.2.1 \
        optuna==3.5.0 optuna-dashboard==0.14.0 \
        onnx==1.15.0 onnxruntime-gpu==1.16.3 \
        matplotlib==3.7.5 seaborn==0.13.2 plotly==5.18.0 kaleido==0.2.1 \
        tqdm==4.66.1 pyyaml==6.0.1 tensorboard==2.14.0 \
        rich==13.7.0 cloudpickle==3.0.0

# -----------------------------------------------------------------------------
# 6. Workspace + ROS sourcing
# -----------------------------------------------------------------------------
RUN echo "source /opt/ros/noetic/setup.bash"            >> /root/.bashrc \
    && echo "source /ws/devel/setup.bash 2>/dev/null || true" >> /root/.bashrc \
    && echo "export ONNXRUNTIME_ROOT_PATH=/opt/onnxruntime" >> /root/.bashrc

WORKDIR /ws
CMD ["/bin/bash"]
