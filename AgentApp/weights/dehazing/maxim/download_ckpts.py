import gdown
import os

url = "https://drive.google.com/drive/folders/18jgfwjhgbVig6Elepro15DBMogessn5G?usp=drive_link"
gdown.download_folder(url, output='./')

os.system("mv maxim/* ./")
os.system("rm -rf maxim")
