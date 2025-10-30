import gdown
import os

url = "https://drive.google.com/drive/folders/1iVRX9Alw2uJXkXIhkyGc0C5m7hxjTY_d"
gdown.download_folder(url, output='./')

os.system("mv MPRNet/* ./")
os.system("rm -rf MPRNet")
