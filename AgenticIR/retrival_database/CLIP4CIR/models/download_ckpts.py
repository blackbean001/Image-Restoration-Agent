'''
This source code is licensed under the license found in the LICENSE file.
This is the implementation of the "Learning to deblur using light field generated and real defocus images" paper accepted to CVPR 2022. 
Project GitHub repository: https://github.com/lingyanruan/DRBNet
Email: lyruanruan@gmail.com
Copyright (c) 2022-present, Lingyan Ruan
'''

## Download weight ##############
import os
import gdown
import shutil

### Google drive IDs ######
ckpt_test  = ['1yhMeQeMTw8fllGssgz5wpYZ26yoYb2_L']

#  download ckpts
print('ckpt downloading!')
for _id in ckpt_test:
    print(f"Downloading {_id}")
    gdown.download(id=_id, quiet=False) 

os.system("unzip clip_finetuned_on_imgres_RN50x4.zip")
os.system("rm clip_finetuned_on_imgres_RN50x4.zip")
