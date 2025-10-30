import gdown
import os

url = "https://drive.google.com/drive/folders/1gnQiI_7Dvy-ZdQUVYXt7pW0EFQkpK39B"

gdown.download_folder(url, output="./")
