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
max_epoch = 10
#train_dir = 'output/train.h5'
train_dir = 'output/train_small.h5'
saved_weights_dir = 'saved_weights'
train_img_dir = '/Holopix50k/train_mini'
val_dir = 'output/val_small.h5'
resume = False #Set to True if want to resume training from previously saved weights
batch = 12
batch_eval = 1
output_dir = 'output_train'
########### Model ############
torch.backends.cudnn.benchmark = True
model = SRCNN_955(device = device).to(device) #can change to 955 or new SRCNN model



optimizer = optim.SGD([
                {'params': model.conv1.parameters(),'lr':1e-4},
                {'params': model.conv2.parameters(), 'lr': 1e-4},
                {'params': model.conv3.parameters(), 'lr': 1e-5},
            ], lr=1e-4, momentum=0.9)#momentum not specified

########## Dataset ###########

train_dataset = TrainDataset(train_dir)
train_dataloader = DataLoader(dataset = train_dataset,
                              batch_size = batch,
                              shuffle = True,
                              num_workers = 2,
                              pin_memory = True)
val_dataset = TrainDataset(val_dir)
val_dataloader = DataLoader(dataset = val_dataset,
    batch_size = batch_eval,
    shuffle = False,
    num_workers = 2,
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

while(epoch < max_epoch):
    since = time.time()
    
    
    criterion = nn.MSELoss()
    
    print('Epoch {}/{}'.format(epoch, max_epoch - 1))
    print('-' * 10)

    for phase in ['train','validation']:
        
        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, max_epoch - 1))
            for iteration, data in enumerate(dataloaders[phase]):
                if(phase == 'train'):
                    model.train()
                elif(phase == 'validation'):
                    model.eval()

                optimizer.zero_grad()
                inputs,labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs,labels)
                from IPython import embed; embed()
                
                loss.backward()
                optimizer.step
                writer.add_scalar('loss',loss.item())
                

                print('Epoch {}, Iteration: {}, Loss: {}'.format(epoch,iteration,loss.item()))

                '''
                calculating psnr,msim for every iteration is expensive. Can calculate for every 500 iterations
                '''
                if(iteration % 500 == 0):                           
                    psnr_score = psnr(inputs,outputs)
                    writer.add_scalar('psnr_train',psnr_score)
                    
                t.set_postfix(loss='{:.6f}'.format(loss.item()))
                t.update(len(inputs))
                
                if(phase == 'validation'):
                    if(iteration % 100 == 0):
                        psnr_score = psnr(inputs,outputs)
                        writer.add_scalar('psnr_validation',psnr_score)
                        
   
    epoch+=1
    if(epoch % 5 == 0): #saving weights every 5 epochs
        torch.save(model.state_dict(), os.path.join(saved_weights_dir, '_epoch_{}.pth'.format(epoch)))

                    



        
    
    
    
    






