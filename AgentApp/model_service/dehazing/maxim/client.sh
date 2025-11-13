HOST="localhost"
PORT="8008"
BASE_URL="http://${HOST}:${PORT}"

curl -X GET "${BASE_URL}/health"

curl -X POST "${BASE_URL}/predict" \
  -F "image=@demo.jpg" \
  -F "task=Dehazing" \
  -F "geometric_ensemble=false" \
  -F "ensemble_times=8" \
  -F "return_format=image" \
  --output result.png

curl -X POST "${BASE_URL}/batch_predict" \
  -F "images=@demo.jpg" \
  -F "images=@demo.jpg" \
  -F "images=@demo.jpg" \
  -F "task=Dehazing" \
  -F "geometric_ensemble=false" \
  -F "ensemble_times=8"

