#!/bin/bash

# Configuration
HOST="localhost"
PORT="8004"
BASE_URL="http://${HOST}:${PORT}"

INPUT_IMAGE="demo.jpg"
OUTPUT_IMAGE="restored.png"

# Supported task types:
# Motion_Deblurring
# Single_Image_Defocus_Deblurring
# Deraining
# Real_Denoising
# Gaussian_Gray_Denoising
# Gaussian_Color_Denoising

#TASK="Single_Image_Defocus_Deblurring"

echo "1. Checking service health status..."
curl -X GET "${BASE_URL}/health"
echo -e "\n"

echo "2. Getting supported tasks list..."
curl -X GET "${BASE_URL}/tasks"
echo -e "\n"

echo "3: Upload image file with form-data ==="
curl -X POST "${BASE_URL}/restore" \
  -F "image=@${INPUT_IMAGE}" \
  -F "tile=512" \
  -F "tile_overlap=32" \
  --output restored_output.png

echo -e "\nResult saved to restored_output.png\n"

