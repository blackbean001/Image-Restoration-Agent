/bash

# XRestormer Image Dehazing Service API Test Script
# Modify HOST and PORT according to your model_services.yaml configuration
HOST="localhost"
PORT="8010"
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

if [ -f "inputs/demo.png" ]; then
    curl -X POST "${BASE_URL}/process" \
      -F "image=@inputs/demo.png" \
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
    echo "  -F \"image=@inputs/demo.png\" \\"
    echo "  -o \"restored_output.png\""
fi

# 3. Batch Image Processing
#echo -e "\n[3] Batch Image Processing..."

#if [ -f "inputs/demo.jpg" ] && [ -f "inputs/demo.jpg" ]; then
#    curl -X POST "${BASE_URL}/batch_process" \
#      -F "images=@inputs/demo.jpg" \
#      -F "images=@inputs/demo.jpg" \
#      -o "batch_results.json"
    
#    if [ $? -eq 0 ]; then
#        echo "✓ Batch processing successful, results saved as batch_results.json"
#        echo "Results preview:"
#        cat batch_results.json | python3 -m json.tool 2>/dev/null || cat batch_results.json
#    else
#        echo "✗ Batch processing failed"
#    fi
#else
#    echo "⚠ Test images not found"
#    echo "Example command:"
#    echo "curl -X POST \"${BASE_URL}/batch_process\" \\"
#    echo "  -F \"images=@inputs/demo.jpg\" \\"
#    echo "  -F \"images=@inputs/demo.jpg\" \\"
#    echo "  -F \"images=@inputs/demo.jpg\" \\"
#    echo "  -o \"batch_results.json\""
#fi

echo -e "\n================================"
echo "Testing Complete"
echo "================================"

