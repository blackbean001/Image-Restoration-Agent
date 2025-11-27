#!/bin/bash

HOST="localhost"
PORT="8001"  # 根据你的配置调整端口
BASE_URL="http://${HOST}:${PORT}"

echo "========================================"
echo "Image Brightening Service Test Script"
echo "========================================"
echo ""

# 1. API Root
echo "=== 1. API Root ==="
curl -X GET "${BASE_URL}/"
echo ""
echo ""

# 2. Health Check
echo "=== 2. Health Check ==="
curl -X GET "${BASE_URL}/health"
echo ""
echo ""

# 3. Get Available Methods
echo "=== 3. Get Available Methods ==="
curl -X GET "${BASE_URL}/methods"
echo ""
echo ""

# 4. Brighten Image - Histogram Equalization (默认方法)
echo "=== 4. Brighten Image (Histogram Equalization - Default) ==="
curl -X GET "${BASE_URL}/brighten" \
  -F "file=@demo.png" \
  --output "output_histogram_equalization.png"
echo "Output saved to: output_histogram_equalization.png"
echo ""

# 5. Brighten Image - Constant Shift
echo "=== 5. Brighten Image (Constant Shift) ==="
curl -X GET "${BASE_URL}/brighten?method=constant_shift" \
  -F "file=@demo.png" \
  --output "output_constant_shift.png"
echo "Output saved to: output_constant_shift.png"
echo ""

# 6. Brighten Image - Gamma Correction
echo "=== 6. Brighten Image (Gamma Correction) ==="
curl -X GET "${BASE_URL}/brighten?method=gamma_correction" \
  -F "file=@demo.png" \
  --output "output_gamma_correction.png"
echo "Output saved to: output_gamma_correction.png"
echo ""

# 7. Brighten Image - Custom Output Format (JPEG)
echo "=== 7. Brighten Image (Output as JPEG) ==="
curl -X GET "${BASE_URL}/brighten?method=histogram_equalization&output_format=jpg" \
  -F "file=@demo.png" \
  --output "output_brightened.jpg"
echo "Output saved to: output_brightened.jpg"
echo ""

# 8. Test with different image
echo "=== 8. Brighten Different Image ==="
curl -X GET "${BASE_URL}/brighten?method=gamma_correction&output_format=png" \
  -F "file=@/path/to/your/image.jpg" \
  --output "output_custom_image.png"
echo "Output saved to: output_custom_image.png"
echo ""

# 9. Error Test - Invalid Method
echo "=== 9. Error Test (Invalid Method) ==="
curl -X GET "${BASE_URL}/brighten?method=invalid_method" \
  -F "file=@demo.png"
echo ""
echo ""

# 10. Error Test - No File
echo "=== 10. Error Test (No File) ==="
curl -X GET "${BASE_URL}/brighten?method=histogram_equalization"
echo ""
echo ""

echo "========================================"
echo "Test Completed!"
echo "========================================"
