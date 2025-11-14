#!/bin/bash

# Set your server host and port
HOST="localhost"
PORT="8026"
BASE_URL="http://${HOST}:${PORT}"

# =============================================================================
# 1. Health Check - Check if the service is running
# =============================================================================
curl -X GET "${BASE_URL}/health"


# =============================================================================
# 2. Initialize Model - Must be called before inference
# =============================================================================
curl -X POST "${BASE_URL}/initialize" \
  -H "Content-Type: application/json" \
  -d '{
    "device": "cuda",
    "reload_swinir": false,
    "seed": 231
  }'

# Alternative: Initialize with CPU
#curl -X POST "${BASE_URL}/initialize" \
#  -H "Content-Type: application/json" \
#  -d '{
#    "device": "cpu",
#    "reload_swinir": false,
#    "seed": 231
#  }'


# =============================================================================
# 3. Inference File - Returns PNG file directly
# ============================================================================= 
# Basic inference returning file
curl -X POST "${BASE_URL}/inference_file" \
  -F "image=@demo.png" \
  -o output.png


# Full inference with parameters returning file
curl -X POST "${BASE_URL}/inference_file" \
  -F "image=@demo.png" \
  -F "steps=50" \
  -F "sr_scale=2.0" \
  -F "color_fix_type=wavelet" \
  -F "disable_preprocess_model=false" \
  -F "tiled=true" \
  -F "tile_size=512" \
  -F "tile_stride=256" \
  -o enhanced_image.png


# Batch processing example
#for img in ./input_images/*.jpg; do
#  filename=$(basename "$img" .jpg)
#  curl -X POST "${BASE_URL}/inference_file" \
#    -F "image=@${img}" \
#    -F "steps=50" \
#    -F "sr_scale=2.0" \
#    -o "./output_images/${filename}_enhanced.png"
#  echo "Processed: ${filename}"
#done


