#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "data" ]; then
    echo "data dir does not exist: please create and extract nuScenes data into it."
    exit 1
fi

docker build -t mcap_converter .
mkdir -p output
docker run -t --rm \
    --user $(id -u):$(id -g) \
    -v $(pwd)/data:/data -v $(pwd)/output:/output \
    mcap_converter python3 convert_to_mcap.py --data-dir /data --output-dir /output "$@"
