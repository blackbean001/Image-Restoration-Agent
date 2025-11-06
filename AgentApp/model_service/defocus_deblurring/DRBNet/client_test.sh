#!/bin/bash

HOST="localhost"
PORT="8002"
BASE_URL="http://${HOST}:${PORT}"

echo "=========================================="
echo "DRBNet Defocus Deblurring Service Demo"
echo "=========================================="
echo ""

echo "1. health check"
echo "---"
curl -X GET "${BASE_URL}/health" | jq .
echo ""
echo ""

echo "2. service info"
echo "---"
curl -X GET "${BASE_URL}/" | jq .
echo ""
echo ""

echo "3. single deblurring (base64)"
echo "---"
curl -X POST "${BASE_URL}/api/v1/deblur/single" \
  -F "image=@demo.jpg" \
  -F "format=base64" \
  | jq -r '.image' | base64 -d > output_single.png

echo "output to: output_single.png"
echo ""
echo ""

echo "4. single deblurring (image file)"
echo "---"
curl -X POST "${BASE_URL}/api/v1/deblur/single" \
  -F "image=@demo.jpg" \
  -F "format=image" \
  -o output_single_direct.png

echo "save image to: output_single_direct.png"
echo ""
echo ""

echo "5. single image deblurring (details)"
echo "---"
curl -X POST "${BASE_URL}/api/v1/deblur/single" \
  -F "image=@demo.jpg" \
  -F "format=base64" \
  | jq '{success: .success, inference_time: .inference_time, format: .format}'

echo ""
echo "Finished !"
