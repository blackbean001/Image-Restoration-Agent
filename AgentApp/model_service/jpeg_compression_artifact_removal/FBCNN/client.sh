#!/bin/bash

# Configure service address (modify according to your setup)
HOST="localhost"
PORT="8019"
BASE_URL="http://${HOST}:${PORT}"

# 1. Health Check
echo "=== 1. Health Check ==="
curl -X GET "${BASE_URL}/health"
echo -e "\n"

# 2. Denoise image using file upload (blind mode)
echo "=== 2. Denoise Image (Blind Mode) ==="
curl -X POST "${BASE_URL}/denoise" \
  -F "image=@demo.png" \
  -F "qf=blind"
echo -e "\n"

# 3. Denoise image using file upload (with specified QF value)
echo "=== 3. Denoise Image (QF=40) ==="
curl -X POST "${BASE_URL}/denoise" \
  -F "image=@demo.png" \
  -F "qf=40"
echo -e "\n"

