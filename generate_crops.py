import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from datasets import TrainDataset,EvalDataset
import imageio
from random import randrange


train_dataset = TrainDataset("output/train_holopix50k_small.h5")
train_dataloader = DataLoader(dataset = train_dataset,batch_size = 40,shuffle = True,num_workers = 10,pin_memory = True)

t = (next(iter(train_dataloader)))
crop_list = []
for k in range(0,40):
    inputt,label = t
    
    crop_list.append(label[k,:,:,:])

for i in crop_list:
    jj = i.detach().cpu().numpy()
    jj = jj.squeeze(0)
    jj = jj.transpose()
    imageio.imwrite("img_crops/random_crop_{}.png".format(randrange(1000)),jj)

