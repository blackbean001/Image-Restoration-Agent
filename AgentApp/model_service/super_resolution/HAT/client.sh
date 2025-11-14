#!/bin/bash

# HAT Super Resolution Service - curl Test Script
# Based on model_services.yaml config, modify HOST and PORT according to your setup

# Configuration variables (modify as needed)
HOST="localhost"
PORT="8027"
BASE_URL="http://${HOST}:${PORT}"

echo "=========================================="
echo "HAT Super Resolution Service Test Script"
echo "=========================================="
echo ""

# 1. Health check (if endpoint is implemented)
curl -X GET "${BASE_URL}/health"
echo ""
echo "=========================================="
echo ""

# 2. Test single image super-resolution (/test endpoint)
echo -e "[2] Test Single Image Super-Resolution - /test"
echo "Prepare test image: test_image.png"
echo ""

# Example command - using image file
cat << 'EOF'
curl -X POST "${BASE_URL}/test" \
  -F "image=@demo.png" \
  -F "save_img=true"
EOF

echo ""
echo "Actual execution example (replace image path):"
# Execute if test image exists
if [ -f "demo.png" ]; then
    echo "Found demo.png, executing test..."
    curl -X POST "${BASE_URL}/test" \
      -F "image=@demo.png" \
      -F "save_img=true"
    echo ""
else
    echo -e "${YELLOW}demo.png not found, skipping execution${NC}"
fi
echo ""
echo "=========================================="
echo ""


