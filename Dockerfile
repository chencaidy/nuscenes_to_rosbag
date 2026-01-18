FROM ros:jazzy-ros-core

RUN apt-get update \
# For common
&& apt-get install -y git libgeos-dev libgl1 python3-pip \
# For ros2
&& apt-get install -y ros-jazzy-foxglove-msgs ros-jazzy-rosbag2-py ros-jazzy-rosbag2-storage-mcap ros-jazzy-tf2-msgs \
# For nuscenes-devkit
&& apt-get install -y python3-cachetools python3-fire python3-matplotlib python3-parameterized python3-shapely python3-sklearn python3-tqdm \
# Clean cache
&& rm -rf /var/lib/apt/lists/*

RUN pip install nuscenes-devkit --break-system-packages
RUN pip install git+https://github.com/DanielPollithy/pypcd.git --break-system-packages

COPY . /work

WORKDIR /work
