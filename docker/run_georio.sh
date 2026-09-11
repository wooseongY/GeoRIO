#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$DEFAULT_WORKSPACE_ROOT}"


if [[ ! -d "$WORKSPACE_ROOT/src" ]]; then
  echo "ERROR: WORKSPACE_ROOT must contain a src directory: $WORKSPACE_ROOT" >&2
  exit 1
fi

xhost +local:docker
docker run --rm -it \
  --net=host \
  -e DISPLAY="${DISPLAY:-:0}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$WORKSPACE_ROOT:/root/ros2_ws" \
  -v /media/wooseong/ws_ssdR/snail_radar:/root/data/snail \
  -v /media/wooseong/ws_ssdR/HeRCULES:/root/data/hercules \
  -v /home/wooseong/hercules_ros2:/root/hercules_ros2_player \
  -v /media/wooseong/ws_ssdR/HKUST_RIO_dataset:/root/data/hkust \
  -v /media/wooseong/ws_ssdE/orebro_4d_radar_dataset/competition_data:/root/data/radar_challenge \
  --workdir /root/ros2_ws \
  --name georio \
  georio:humble
