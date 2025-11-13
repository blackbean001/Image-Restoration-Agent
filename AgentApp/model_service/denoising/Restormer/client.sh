#!/bin/bash

# Configure service address and port (modify according to your setup)
HOST="localhost"
PORT="8009"
BASE_URL="http://${HOST}:${PORT}"

# ============================
# 1. Health Check
# ============================
echo "=== Health Check ==="
curl -X GET "${BASE_URL}/health"
echo -e "\n"

# ============================
# 2. Query Supported Tasks
# ============================
echo "=== Query Supported Tasks ==="
curl -X GET "${BASE_URL}/tasks"
echo -e "\n"

# ============================
# 3. Examples for Different Task Types
# ============================

# Motion Deblurring
#curl -X POST "${BASE_URL}/restore" \
#  -F "image=@input.jpg" \
#  -F "task=Motion_Deblurring" \
#  --output motion_deblur.png

# Defocus Deblurring
#curl -X POST "${BASE_URL}/restore" \
#  -F "image=@input.jpg" \
#  -F "task=Single_Image_Defocus_Deblurring" \
#  --output defocus_deblur.png

# Deraining
#curl -X POST "${BASE_URL}/restore" \
#  -F "image=@input.jpg" \
#  -F "task=Deraining" \
#  --output derain.png

# Real Denoising
curl -X POST "${BASE_URL}/restore" \
  -F "image=@demo.png" \
  -F "task=Real_Denoising" \
  --output denoise.png

# Gaussian Gray Denoising
#curl -X POST "${BASE_URL}/restore" \
#  -F "image=@input.jpg" \
#  -F "task=Gaussian_Gray_Denoising" \
#  --output gray_denoise.png

# Gaussian Color Denoising
#curl -X POST "${BASE_URL}/restore" \
#  -F "image=@input.jpg" \
#  -F "task=Gaussian_Color_Denoising" \
#  --output color_denoise.png


