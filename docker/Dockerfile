# ROS base
FROM osrf/ros:humble-desktop-full

# Choose bash shell
SHELL ["/bin/bash", "-c"]

# Environment Variables
ENV HOME=/root

# Update apt
RUN apt update

# Install git and SSH client
RUN apt install -y git openssh-client

# Add public github and bitbucket keys
RUN mkdir -p -m 0600 $HOME/.ssh
RUN ssh-keyscan github.com >> $HOME/.ssh/known_hosts
RUN ssh-keyscan bitbucket.org >> $HOME/.ssh/known_hosts

# Install system dependencies
RUN apt update && apt install -y \
    libeigen3-dev \
    libomp-dev \
    libpcl-dev \
    ros-humble-pcl*

# Install mcap support
RUN apt install -y ros-humble-rosbag2-storage-mcap

# Install gdb support
RUN apt install -y gdb gdbserver

# Python tooling for trajectory evaluation and the viser viewer (viz/).
# Kept in its own venv so it cannot bump the numpy/scipy that ROS 2 ships with.
RUN apt install -y python3-pip python3-venv && \
    python3 -m venv /opt/vizenv && \
    /opt/vizenv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/vizenv/bin/pip install --no-cache-dir viser numpy scipy evo
ENV VIZ_PYTHON=/opt/vizenv/bin/python

# Add colcon bash completion
RUN echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> $HOME/.bashrc

# Create ROS2 Workspace
RUN mkdir -p $HOME/ros2_ws/src

# Default workspace
WORKDIR $HOME/ros2_ws

# ROS base source by default
RUN echo "source /opt/ros/humble/setup.bash" >> $HOME/.bashrc
