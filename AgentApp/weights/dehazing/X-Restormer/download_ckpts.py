import gdown
import os

url = "https://drive.google.com/drive/folders/16WxegSAN_sescgrfW4ZMO4b6TcR_7T24"
gdown.download_folder(url, output='./')

os.system("mv X-Restormer/* ./")
os.system("rm -rf X-Restormer")
