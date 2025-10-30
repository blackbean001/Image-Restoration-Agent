import gdown
import os

url = "https://drive.google.com/drive/folders/14HceLBiKbMYK0OxiIjAdtCwBN8C86714"
gdown.download_folder(url, output='./')

os.system("mv swinir/* ./")
os.system("rm -rf swinir")
