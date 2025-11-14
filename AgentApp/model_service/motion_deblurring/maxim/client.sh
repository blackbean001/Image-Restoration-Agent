HOST="localhost"
PORT="8023"
BASE_URL="http://${HOST}:${PORT}"

curl -X GET "${BASE_URL}/health"

curl -X POST "${BASE_URL}/predict" \
  -F "image=@demo.png" \
  -F "task=Deblurring" \
  -F "geometric_ensemble=false" \
  -F "ensemble_times=8" \
  -F "return_format=image" \
  --output result.png

curl -X POST "${BASE_URL}/batch_predict" \
  -F "images=@demo.png" \
  -F "images=@demo.png" \
  -F "images=@demo.png" \
  -F "task=Deblurring" \
  -F "geometric_ensemble=false" \
  -F "ensemble_times=8"

