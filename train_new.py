from model import SRCNN_915,SRCNN_955
import sys
import math
import time
import datetime
import os
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import PIL.Image as pil
import matplotlib.pyplot as plt

from datasets import TrainDataset,EvalDataset

from torchvision.utils import make_grid, save_image
from torchvision import datasets, models, transforms
from torch.utils.data.dataloader import DataLoader

from utils.utils import psnr
from tqdm import tqdm

#TensorboardX
from tensorboardX import SummaryWriter
import utils.pytorch_ssim

########### TensorboardX ###########
LOG_DIR = './logs/'

now = str(datetime.datetime.now())
OUTPUTS_DIR = './output_images/'

if not os.path.exists(LOG_DIR):
	os.makedirs(LOG_DIR)

if not os.path.exists(OUTPUTS_DIR):
	os.makedirs(OUTPUTS_DIR)
OUTPUTS_DIR = OUTPUTS_DIR + now + '/'
if not os.path.exists(OUTPUTS_DIR):
	os.makedirs(OUTPUTS_DIR)
if not os.path.exists(LOG_DIR+now):
	os.makedirs(LOG_DIR+now)

writer = SummaryWriter(LOG_DIR + now)

########### Hparams and File Paths ###########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Torch device: {torch.cuda.get_device_name()}")
max_epoch = 300
#train_dir = 'output/train.h5'
train_dir = 'output/train_holopix50k.h5'
saved_weights_dir = 'saved_weights/x3/epoch_190.pth'
train_img_dir = '/Holopix50k/train/left'
val_dir = 'output/train_holopix50k_small.h5'
resume = False #Set to True if want to resume training from previously saved weights
batch = 12
batch_eval = 1
output_dir = 'output_train'
########### Model ############
torch.backends.cudnn.benchmark = True
model = SRCNN_955(device = device).to(device) #can change to 955 or new SRCNN model
print(device)
print(torch.cuda.get_device_name(device))


optimizer = optim.SGD([
                {'params': model.conv1.parameters(),'lr':1e-4},
                {'params': model.conv2.parameters(), 'lr': 1e-4},
                {'params': model.conv3.parameters(), 'lr': 1e-5},
            ], lr=1e-4)#momentum not specified

########## Dataset ###########

train_dataset = TrainDataset(train_dir)
train_dataloader = DataLoader(dataset = train_dataset,
    batch_size = batch,
    shuffle = True,
    num_workers = 10,
    pin_memory = True,
    drop_last = True)
val_dataset = TrainDataset(val_dir)
val_dataloader = DataLoader(dataset = val_dataset,
    batch_size = batch_eval,
    shuffle = False,
)


dataloaders = {'train': train_dataloader,'validation':val_dataloader}
######### Loading weights ##########


                             
                              
######## Main Training Loop ###
epoch = 0
best_loss = 100000
resume_train = True
criterion = nn.MSELoss().cuda()

transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
) #Image is transformed to calculate the psnr score

for epoch in range(max_epoch):
    model.train()
    epoch_loss = 0.0 
    with tqdm(total = (len(train_dataset) - len(train_dataset) % batch)) as t:
        t.set_description('epoch: {}/{}'.format(epoch, max_epoch - 1))

        for data in train_dataloader:
            input,labels = data

            input = input.to(device)
            labels = labels.to(device)

            preds = model(input)
            loss = criterion(preds,labels)
            epoch_loss += loss.item()/input.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix

            t.set_postfix(loss='{:.6f}'.format(loss.item()))
            t.update(len(input))
            writer.add_scalar('loss',loss.item())

    model.eval()
    val_psnr_list = 0.0
    for data in val_dataloader:
        inputs,labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0,1.0)

        # grid_inputs = torchvision.utils.make_grid(inputs)
        # writer.add_image('Input LR',grid_inputs)
        # grid_outputs = torchvision.utils.make_grid(preds)
        # writer.add_image('Output HR',grid_outputs)
        # grid_gt = torchvision.utils.make_grid(labels)
        # writer.add_image('Ground Truth HR',grid_gt)
  
        psnr1 = psnr(inputs.squeeze(0),preds.squeeze(0))
        val_psnr_list += psnr1
        writer.add_scalar("val_psnr",psnr1)

        

    print('validation psnr: {} for epoch: {}'.format(val_psnr_list/len(val_dataset),epoch))
    torch.save({
        'epoch':epoch,
        'model_state_dict': model.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(),
        'val_psnr': psnr1,
    },os.path.join('saved_weights/testing', 'epoch_{}.pth'.format(epoch)))








