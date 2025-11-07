curl -X GET http://localhost:8006/health

curl -X POST http://localhost:8006/dehaze \
  -F "image=@demo.jpg" \
  -F "max_size=1500" \
  -F "return_type=file" \
  -o dehazed_output.png
