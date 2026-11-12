#!/bin/bash
# Download the gated SAM 3 checkpoint to autolabel/weights/ (git-ignored).
# Needs approved access on https://huggingface.co/facebook/sam3 and `hf auth login` first.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hf download facebook/sam3 sam3.pt config.json --local-dir "$HERE/weights"
ls -lh "$HERE/weights/sam3.pt"
