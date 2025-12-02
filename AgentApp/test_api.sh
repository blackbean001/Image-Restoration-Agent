curl -X POST "http://0.0.0.0:1146/restoration"  \
	-F "file=./demo_input/input.png"  \
	-F "output_format=PNG"  \
	--output restored.png
