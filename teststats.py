import os
import glob
import struct
import numpy as np

s=np.load('steerstats.npz')['arr_0']

smean=s[0]
sstd=s[1]

directory='/home/jenny/winlab/upcar/data/5-13-2019/'

files=glob.glob(os.path.join(directory, 'data*'))

steering=[]
for f in files:
    f_handle=open(f, "rb")
    while True:
        chunk=f_handle.read(12)
        if chunk:
            *word, STR, THR=struct.unpack("4Bii", chunk)
            print(word)
            steering.append(STR)
            f_handle.read(424*240*3)
        else:
            break
        
for s in steering:
    print((s-smean)/sstd)
