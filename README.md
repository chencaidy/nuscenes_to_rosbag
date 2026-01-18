# nuscenes to rosbag2

> _Convert [nuScenes](https://www.nuscenes.org/) data into [rosbag2](https://docs.ros.org/en/jazzy/index.html) format_

## Introduction

nuScenes is a large-scale dataset of autonomous driving in urban environments, provided free for non-commercial use. This project provides helper scripts to download the nuScenes dataset and convert scenes into [rosbag2](https://docs.ros.org/en/jazzy/index.html) files for easy viewing in tools such as [Foxglove](https://foxglove.dev/).

## Usage

### Converting the nuScenes data to rosbag2
1. Download the [nuScenes mini dataset](https://nuscenes.org/nuscenes). You will need to make an account and agree to the terms of use.
1. Extract the following files into the `data/` directory:
    1. `can_bus.zip` to `data/`
    1. `nuScenes-map-expansion-v1.3.zip` to `data/maps`
    1. `v1.0-mini.tgz` to `data/`
1. Build and run the converter container with `./convert_mini_scenes.sh`

## License

nuscenes_to_rosbag is licensed under the [MIT License](https://opensource.org/licenses/MIT).
