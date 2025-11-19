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
ckpt_test  = ['19blz_0cci6lBnAE-pO62U5Tj2HaGuqY2',
        '1BKI2r8ZCsJZ354-FmscS-Dwc5tJXEQ-B',
        '1_QavijTzENPSHTH2fI9vbs8ezPIo1AnK',
        '1-pYQEVVL6Ou5aykxL3VwHYhl9GMyPMYk',
        '14DGsMZQJLKK2bsS84QRTl9tpaPJjaqJ2',
        '1VyQPT66_g8mymh4yZI7YqcDgHZrV_k43',
        '12ArxuSJqDfzUCAuzyDQrP_dzGh613OXg',
        '1d1GlDy_9qZVqAywTOMyrUHmvJPqJDpLs',
        '1QP-o16U9SpgJ8O8dgUMbRtkTcAUNXiv8',
        '1a2EhVI-LpaeOxOdkIY-O1VOB6YQ44xhR']  # https://drive.google.com/file/d/1vGImev9LdagttXE_nN1gZGVstVTRVQHt/view?usp=sharing

#  download ckpts
print('ckpt downloading!')
for _id in ckpt_test:
    print(f"Downloading {_id}")
    gdown.download(id=_id, quiet=False) 


