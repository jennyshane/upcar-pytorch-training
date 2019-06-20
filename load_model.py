import os 
import math
import numpy as np
import glob
import datetime
import struct

import torch
import torch.nn as nn

from model import Net

steerstats=np.load("steerstats.npz")['arr_0']
steer_mean=steerstats[0]
steer_std=steerstats[1]

net=Net()
net.load_state_dict(torch.load('5-14-2019/weight_file_epoch48'))
net.eval()

test_data='/home/jenny/winlab/upcar/data/5-13-2019/data_2019-05-13_15-06-09'

data_file=open(test_data, 'rb')
while True:
    header=data_file.read(12)
    *word, STR, THR=struct.unpack('4Bii', header)
    image_data=data_file.read(424*240*3)
    image=torch.from_numpy(np.fromstring(image_data, np.uint8).reshape(240, 424, 3)).permute(2, 0, 1).float()
    im_mean=image.mean()
    im_std=image.std()
    image=(image-im_mean)/im_std
    output=net.forward(image.unsqueeze(0)).item()
    print(output*steer_std+steer_mean, STR)

