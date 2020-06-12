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

from datasets import TrainDataset

from torchvision.utils import make_grid, save_image
from torchvision import datasets, models, transforms
from torch.utils.data.dataloader import DataLoader

from utils import psnr
from tqdm import tqdm

#TensorboardX
from tensorboardX import SummaryWriter

########### TensorboardX ###########
LOG_DIR = './logs/'

now = str(datetime.datetime.now())
# OUTPUTS_DIR = './outputs/'

if not os.path.exists(LOG_DIR):
	os.makedirs(LOG_DIR)

# if not os.path.exists(OUTPUTS_DIR):
# 	os.makedirs(OUTPUTS_DIR)
# OUTPUTS_DIR = OUTPUTS_DIR + now + '/'
# if not os.path.exists(OUTPUTS_DIR):
# 	os.makedirs(OUTPUTS_DIR)
if not os.path.exists(LOG_DIR+now):
	os.makedirs(LOG_DIR+now)

writer = SummaryWriter(LOG_DIR + now)

########### Hparams ###########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Torch device: {torch.cuda.get_device_name()}")

max_epoch = 10

train_dir = 'output/train.h5'
saved_weights_dir = 'saved_weights'
train_img_dir = '/Holopix50k/train_mini'

batch = 12
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
                             
                              
######## Main Training Loop ###
epoch = 0
best_loss = 100000


transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

while(epoch < max_epoch):
    since = time.time()
    model.train()
    criterion = nn.MSELoss()
    
    print('Epoch {}/{}'.format(epoch, max_epoch - 1))
    print('-' * 10)
    with tqdm(total=(len(train_dataset) - len(train_dataset) % batch)) as t:
        t.set_description('epoch: {}/{}'.format(epoch, max_epoch - 1))
        for iteration, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs,labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            #from IPython import embed;embed()
            loss = criterion(outputs,labels)
            

            loss.backward()
            optimizer.step
            writer.add_scalar('loss',loss.item())

            print('Epoch {}, Iteration: {}, Loss: {}'.format(epoch,iteration,loss.item()))

            '''
            calculating psnr for every iteration is expensive. Can calculate for every 500 iterations
            '''
                            
#             psnr_score = psnr(inputs,outputs)
#             writer.add_scalar('psnr',psnr_score)
                
            t.set_postfix(loss='{:.6f}'.format(loss.item()))
            t.update(len(inputs))
        epoch+=1
        if(epoch % 5 == 0):
            torch.save(model.state_dict(), os.path.join(saved_weights, '_epoch_{}.pth'.format(epoch)))
            
            
        
    
    
    
    






