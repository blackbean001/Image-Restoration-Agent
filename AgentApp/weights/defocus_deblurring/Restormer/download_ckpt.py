import gdown
import os

os.makedirs("Deraining", exist_ok=True)
os.makedirs("Motion_Deblurring", exist_ok=True)
os.makedirs("Defocus_Deblurring", exist_ok=True)
os.makedirs("Denoising", exist_ok=True)

ids = ["1uuejKpyo0G_5M4DAO2J9_Dijy550tjc5",  \
       "1pwcOhDS5Erzk8yfAbu7pXTud606SB4-L",  \
       "167enijHIBa1axZRaRjkk_U6kLKm40Z43",  \
       "10v8BH3Gktl34TYzPy0x-pAKoRSYKnNZp",  \
       "1yMDOlLYUVruC9ytXDKQdQICnhnTGq09A"]
url_base = "https://drive.google.com/uc?id="

for i,_id in enumerate(ids):
    if i == 0:
        gdown.download(url_base + _id, "Deraining")
    if i == 1:
        gdown.download(url_base + _id, "Motion_Deblurring")
    if i in [2, 3]:
        gdown.download(url_base + _id, "Defocus_Deblurring")

url = "https://drive.google.com/drive/folders/1Qwsjyny54RZWa7zC4Apg7exixLBo4uF0"
gdown.download_folder(url, output="./Denoising")
