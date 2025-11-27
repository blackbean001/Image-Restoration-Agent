#!/bin/bash

# Configure service address (modify according to your setup)
HOST="localhost"
PORT="8019"
BASE_URL="http://${HOST}:${PORT}"

curl -X POST ${BASE_URL}/denoise \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "demo.png",
    "output_path": "denoised.png"
  }'

curl -X POST ${BASE_URL}/denoise \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "demo.png",
    "output_path": "denoised.png",
    "qf": 75
  }'
