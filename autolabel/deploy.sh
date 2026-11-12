#!/bin/bash
# Build and deploy the SAM 3 function into the local CVAT nuclio (serverless compose must be up).
# Usage: autolabel/deploy.sh [cpu|gpu]   (default cpu; gpu needs NVIDIA driver + nvidia-container-toolkit)
# Mounts data/ (only read) and autolabel/ read-write (config.json, weights/, raw/); no weights in the image.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
MODE="${1:-cpu}"
mkdir -p "$HERE/weights" "$HERE/raw" "$HERE/wheels"

if [ "$MODE" = gpu ]; then
    docker build -t cvat.sam3.base.gpu --build-context wheels="$HERE/wheels" -f "$HERE/nuclio/Dockerfile.gpu" "$HERE/nuclio"
    FILE="$HERE/nuclio/function-gpu.yaml"
else
    docker build -t cvat.sam3.base "$HERE/nuclio"
    FILE="$HERE/nuclio/function.yaml"
fi
nuctl create project cvat --platform local 2>/dev/null || true
nuctl deploy --project-name cvat --path "$HERE/nuclio" --file "$FILE" \
    --platform local \
    --volume "$REPO/data:/data" --volume "$HERE:/autolabel" \
    --platform-config '{"attributes": {"network": "cvat_cvat"}}'
nuctl get function --platform local
