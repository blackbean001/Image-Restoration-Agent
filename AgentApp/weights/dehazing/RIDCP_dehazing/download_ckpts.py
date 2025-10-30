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
ckpt_test  = '1zykiY3kLIN_9c8QbNZzRF5PwNS3qF0P8'  # https://drive.google.com/file/d/1vGImev9LdagttXE_nN1gZGVstVTRVQHt/view?usp=sharing

#  download ckpts
print('ckpt downloading!')
gdown.download(id=ckpt_test, quiet=False) 


