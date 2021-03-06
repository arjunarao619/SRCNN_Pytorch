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
max_epoch = 500
#train_dir = 'output/train.h5'
train_dir = 'output/train_holopix50k.h5'
saved_weights_dir = 'saved_weights'
train_img_dir = '/Holopix50k/train/left'
val_dir = 'output/val_holopix50k.h5'
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
            ], lr=1e-4, momentum=0.9)#momentum not specified

########## Dataset ###########

train_dataset = TrainDataset(train_dir)
train_dataloader = DataLoader(dataset = train_dataset,batch_size = batch,shuffle = True,num_workers = 10,pin_memory = True)
val_dataset = TrainDataset(val_dir)
val_dataloader = DataLoader(dataset = val_dataset,
    batch_size = batch_eval,
    shuffle = False,
    num_workers = 10,
)

dataloaders = {'train': train_dataloader,'validation':val_dataloader}
                             
                              
######## Main Training Loop ###
epoch = 0
best_loss = 100000


transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
) #Image is transformed to calculate the psnr score



epoch_loss_l = []

for epoch in range(0,max_epoch):
    
    since = time.time()
    criterion = nn.MSELoss().cuda()
    print('Epoch {}/{}'.format(epoch, max_epoch - 1))
    print('-' * 10)
    for phase in ['train','validation']:
        if(phase == 'train'):
                model.train()
        elif(phase == 'validation'):
            model.eval()
            print("--------------------Validation-----------------------------")
        running_loss = 0.0
        for iteration, data in enumerate(dataloaders[phase]):
            

            optimizer.zero_grad()
            inputs,labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs,labels).to(device)
            
            
            
            loss.backward()
            optimizer.step
            writer.add_scalar('loss',loss.item())

            running_loss += loss.item() * inputs.size(0)
        

            print('Epoch {}, Iteration: {}, Loss: {}'.format(epoch,iteration,loss.item()))

            '''
            calculating psnr,msim for every iteration is expensive. Can calculate for every 500 iterations
            '''
            if(iteration % 100 == 0):                           
                psnr_score = psnr(inputs,outputs)
                writer.add_scalar('psnr_train',psnr_score)
                ssim_score = utils.pytorch_ssim.ssim(inputs,outputs) #calculated in batches of batch_size
                writer.add_scalar('ssim_train',ssim_score.item())

            
            if(phase == 'validation'):
                if(iteration % 50 == 0): 
                    psnr_score = psnr(inputs,outputs)
                    writer.add_scalar('psnr_validation',psnr_score)
                    ssim_score = utils.pytorch_ssim.ssim(inputs,outputs) #calculated in batches of batch_size
                    writer.add_scalar('ssim_validation',ssim_score.item())
                if(iteration % 200 == 0):
                    grid_inputs = torchvision.utils.make_grid(inputs)
                    writer.add_image('Input LR',grid_inputs)
                    grid_outputs = torchvision.utils.make_grid(outputs)
                    writer.add_image('Output HR',grid_outputs)
                    grid_gt = torchvision.utils.make_grid(labels)
                    writer.add_image('Ground Truth HR',grid_gt)
        
        epoch_loss = running_loss/len(train_dataset)
        epoch_loss_l.append(epoch_loss)
        print('{} Loss: {}'.format(phase, epoch_loss))
        torch.save(model.state_dict(), os.path.join(saved_weights_dir, '_epoch_{}.pth'.format(epoch)))

                    



        
    
    
    
    






