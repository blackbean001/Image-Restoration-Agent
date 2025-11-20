import gdown
import os

url = "https://drive.google.com/drive/folders/16WxegSAN_sescgrfW4ZMO4b6TcR_7T24"
gdown.download_folder(url, output='./')

os.system("mv X-Restormer/* ./")
os.system("rm -rf X-Restormer")

os.system("mv Deblurring/net_g_latest.pth deblur_300k.pth")
os.system("mv Dehazing/dehaze_300k.pth ./")
os.system("mv Denoising/denoise_300k.pth ./")
os.system("mv Deraining/net_g_155000.pth derain_155k.pth")
os.system("mv SR/net_g_latest.pth sr_300k.pth")

os.system("rm -rf Deblurring Dehazing Denoising Deraining SR")

