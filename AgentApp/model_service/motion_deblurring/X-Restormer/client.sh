/bash

# XRestormer Image Dehazing Service API Test Script
# Modify HOST and PORT according to your model_services.yaml configuration
HOST="localhost"
PORT="8022"
BASE_URL="http://${HOST}:${PORT}"

echo "================================"
echo "XRestormer API Test Script"
echo "================================"

# 1. Health Check
echo -e "\n[1] Health Check..."
curl -X GET "${BASE_URL}/health"
echo -e "\n"

# 2. Single Image Processing
echo -e "\n[2] Single Image Processing..."

if [ -f "demo.png" ]; then
    curl -X POST "${BASE_URL}/process" \
      -F "image=@demo.png" \
      -o "restored_output.png"
    
    if [ $? -eq 0 ]; then
        echo "✓ Image processed successfully, saved as restored_output.png"
    else
        echo "✗ Image processing failed"
    fi
else
    echo "⚠ Test image not found, please create test_image.jpg"
    echo "Example command:"
    echo "curl -X POST \"${BASE_URL}/process\" \\"
    echo "  -F \"image=@demo.png\" \\"
    echo "  -o \"restored_output.png\""
fi


echo -e "\n================================"
echo "Testing Complete"
echo "================================"

