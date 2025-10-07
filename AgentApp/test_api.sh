curl -X POST "http://0.0.0.0:1146/restoration"  \
	-F "file=@/home/jason/Auto-Image-Restoration-Service/Auto-Image-Restoration/AgentApp/demo_input/001.png"  \
	-F "output_format=PNG"  \
	--output restored.png
