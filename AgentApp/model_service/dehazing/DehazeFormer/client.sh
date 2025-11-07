#!/bin/bash

# DehazeFormer Model Inference Service curl Test Script
# Modify HOST and PORT according to model_services.yaml configuration
HOST="localhost"
PORT="8005"
BASE_URL="http://${HOST}:${PORT}"

echo "=========================================="
echo "DehazeFormer API Test Script"
echo "=========================================="

# 1. Health check
echo -e "\n[1] Health check..."
curl -X GET "${BASE_URL}/health"
echo -e "\n"

# 2. Single image dehazing
echo -e "\n[2] Single image dehazing..."
if [ -f "demo.jpg" ]; then
    curl -X POST "${BASE_URL}/dehaze" \
         -F "image=@demo.jpg" \
         -o "dehazed_output.png"
    echo "Dehazing result saved to: dehazed_output.png"
else
    echo "Error: demo.jpg file not found"
fi
echo -e "\n"

# 3. Batch image dehazing
echo -e "\n[3] Batch image dehazing..."
curl -X POST "${BASE_URL}/batch_dehaze" \
     -F "images=@demo.jpg" \
     -F "images=@demo.jpg" \
     -F "images=@demo.jpg"
echo -e "\n"

# 4. Request with verbose information
echo -e "\n[5] Request with verbose information..."
curl -X POST "${BASE_URL}/dehaze" \
     -F "image=@demo.jpg" \
     -v \
     -o "dehazed_with_verbose.png" 2>&1 | grep -E "HTTP|Content"
echo -e "\n"

# 6. Error handling test - no file
echo -e "\n[6] Error handling test - no file..."
curl -X POST "${BASE_URL}/dehaze"
echo -e "\n"

# 7. Error handling test - empty filename
echo -e "\n[7] Error handling test - empty filename..."
curl -X POST "${BASE_URL}/dehaze" \
     -F "image=@"
echo -e "\n"

echo "=========================================="
echo "Testing completed!"
echo "=========================================="
