import gdown
import os

os.makedirs("Deraining", exist_ok=True)
os.makedirs("Motion_Deblurring", exist_ok=True)
os.makedirs("Defocus_Deblurring", exist_ok=True)
os.makedirs("Denoising", exist_ok=True)

"""
ids = ["1uuejKpyo0G_5M4DAO2J9_Dijy550tjc5",  \
       "1pwcOhDS5Erzk8yfAbu7pXTud606SB4-L",  \
       "167enijHIBa1axZRaRjkk_U6kLKm40Z43",  \
       "10v8BH3Gktl34TYzPy0x-pAKoRSYKnNZp",  \
       "1yMDOlLYUVruC9ytXDKQdQICnhnTGq09A"]

url_base = "https://drive.google.com/uc?id="

for i,_id in enumerate(ids):
    if i == 0:
        gdown.download(url_base + _id)
    if i == 1:
        gdown.download(url_base + _id)
    if i in [2, 3]:
        gdown.download(url_base + _id)
#url = "https://drive.google.com/drive/folders/1Qwsjyny54RZWa7zC4Apg7exixLBo4uF0"
"""

url = "https://drive.google.com/drive/folders/1klNZCGjiGxhj0B1o45Q5JsUeDan3It3X"
gdown.download_folder(url, output=".")

os.system("unzip pretrained_models-20251106T200129Z-1-001.zip")
os.system("mv deraining.pth Deraining/")
os.system("mv dual_pixel_defocus_deblurring.pth Defocus_Deblurring/")
os.system("mv gaussian_color_denoising_blind.pth Denoising/")
os.system("mv motion_deblurring.pth Motion_Deblurring/")
os.system("mv single_image_defocus_deblurring.pth Defocus_Deblurring/")
os.system("mv pretrained_models/* Denoising/")
os.system("rm -rf pretrained_models pretrained_models-20251106T200129Z-1-001.zip")
