#!/bin/bash

# Configure server address and port (modify according to your config)
HOST="localhost"
PORT="8020"
BASE_URL="http://${HOST}:${PORT}"

# ============================================
# 1. Health Check
# ============================================
echo "=== Health Check ==="
curl -X GET "${BASE_URL}/health"
echo -e "\n"

# ============================================
# 2. List Supported Tasks
# ============================================
echo "=== List Tasks ==="
curl -X GET "${BASE_URL}/tasks"
echo -e "\n"

# ============================================
# 3. File Upload Processing (multipart/form-data)
# ============================================
echo "=== Process Image (File Upload) ==="

# Denoising task
#curl -X POST "${BASE_URL}/process" \
#  -F "file=@demo.png" \
#  -F "task=Denoising" \
#  -o restored_denoising.png

# Deblurring task
curl -X POST "${BASE_URL}/process" \
  -F "file=@demo.png" \
  -F "task=Deblurring" \
  -o restored_deblurring.png

# Deraining task
#curl -X POST "${BASE_URL}/process" \
#  -F "file=@demo.jpg" \
#  -F "task=Deraining" \
#  -o restored_deraining.png

echo "Files saved as restored_*.png"
echo -e "\n"

