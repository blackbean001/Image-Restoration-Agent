import gdown
import os

url = "https://drive.google.com/drive/folders/1HpmReFfoUqUbnAOQ7rvOeNU3uf_m69w0"
gdown.download_folder(url, output='./')

os.system("mv HAT_Pretrained_Models/* ./")
os.system("rm -rf HAT_Pretrained_Models")
