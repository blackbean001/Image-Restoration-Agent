#!/bin/bash

BASE_URL="http://localhost:8003"

echo "========================================"
echo "1. Health Check"
echo "========================================"

curl -X GET "${BASE_URL}/health"

echo -e "\n\n"

echo "========================================"
echo "2. Single Image Deblur (PNG output)"
echo "========================================"

# Basic usage
curl -X POST "${BASE_URL}/deblur" \
  -F "image=@demo.jpg" \
  -F "format=jpg" \
  -o deblurred_output.png

echo "Saved to: deblurred_output.png"
echo -e "\n"

echo "========================================"
echo "3. Single Image Deblur (JPG output)"
echo "========================================"

curl -X POST "${BASE_URL}/deblur" \
  -F "image=@demo.jpg" \
  -F "format=jpg" \
  -o deblurred_output.jpg

echo "Saved to: deblurred_output.jpg"
echo -e "\n"


echo "========================================"
echo "4. Verbose Request (for debugging)"
echo "========================================"

curl -X POST "${BASE_URL}/deblur" \
  -F "image=demo.jpg" \
  -F "format=png" \
  -o output_verbose.png \
  -v

echo -e "\n\n"

echo "========================================"
echo "Done!"
echo "========================================"


