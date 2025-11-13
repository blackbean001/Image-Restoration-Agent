HOST="localhost"
PORT="8013"
BASE_URL="http://${HOST}:${PORT}"

curl -X GET "${BASE_URL}/health"

curl -X POST "${BASE_URL}/predict" \
  -F "image=@demo.png" \
  -F "task=color_dn" \
  -F "scale=4" \
  -F "noise=15" \
  -F "jpeg=40" \
  -F "large_model=false" \
  -F "tile_overlap=32" \
  -F "return_type=file" \
  --output output.png

#curl -X POST http://localhost:5000/predict_batch \
#  -F "images=@/path/to/image1.jpg" \
#  -F "images=@/path/to/image2.jpg" \
#  -F "images=@/path/to/image3.jpg" \
#  -F "task=real_sr" \
#  -F "scale=2" \
#  -F "large_model=true" \
#  -F "tile=512" \
#  | jq '.'

