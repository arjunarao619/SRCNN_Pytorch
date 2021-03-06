import numpy as np
import PIL.Image as pil 
from PIL import ImageFilter as pilfilter
import os
import torch
import time
import glob
import h5py
import sys
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm.auto import trange
import skimage.io as io
import torch.utils.data as data
from utils.utils import apply_gaussian_kernel,rgb_to_y,centre_crop



def transform_dataset(scale,patch,stride,output_path,trainorval):
    #train_dir = "/home/ec2-user/arjun/SRCNN_pytorch/Holopix50k/train_mini" #only 91 images 
    train_dir = "/research/d2/arao/Holopix50k/train/left" #
    val_dir = "/research/d2/arao/Holopix50k/val/left"
    temp_list = []
    lr_patches = []
    hr_patches = []
    h5_file = h5py.File(output_path, 'w')

    if(trainorval == "train"):
        img_dir = train_dir
    elif(trainorval == "validation"):
        img_dir = val_dir
    
    

    for img_path in sorted(glob.glob('{}/*'.format(img_dir))):

    
        input_hr = pil.open(img_path).convert('RGB')
        crop = centre_crop(input_hr,1)#no crop
        hr = crop

        # hr = input_hr.resize(((input_hr.width // scale) * scale, (input_hr.height // scale) * scale),resample = pil.BICUBIC)
        # crop = hr

        input_lr = crop.resize((crop.width // scale, crop.height // scale),resample = pil.BICUBIC)
        lr = input_lr.resize((input_lr.width * scale, input_lr.height * scale),resample = pil.BICUBIC)
        lr_blur = apply_gaussian_kernel(lr,0.73)
        lr = lr_blur

        hr = np.asarray(hr).astype(np.float32)
        lr = np.asarray(lr).astype(np.float32)
        hr = rgb_to_y(hr)
        lr = rgb_to_y(lr)
        

        '''
        patches of size = 231 along the Y channel are extracted with a stride = 98
        '''
        
        for i in range(0, lr.shape[0] - patch + 1, stride): 
            for j in range(0, lr.shape[1] - patch + 1, stride):
                lr_patches.append(lr[i:i + patch, j:j + patch])
                hr_patches.append(hr[i:i + patch, j:j + patch])
                print("LR : " + str(len(lr_patches)) + " HR: " + str(len(hr_patches)),end = '\r')
                if(len(lr_patches) == 24000): #specify here required number of sub-images. Recommended 500,000 train, 90,000 val
                    lr_patches = np.array(lr_patches)
                    hr_patches = np.array(hr_patches)
                    
                    print(lr_patches.shape)

                    h5_file.create_dataset('lr',data = lr_patches)
                    h5_file.create_dataset('hr',data = hr_patches)
                    h5_file.close()
                    return

            

    # lr_patches = np.array(lr_patches)
    # hr_patches = np.array(hr_patches)
    
    # print(lr_patches.shape)

    # h5_file.create_dataset('lr',data = lr_patches)
    # h5_file.create_dataset('hr',data = hr_patches)    

    # h5_file.close()    

'''
final patch is a square image with shape = (patch,patch)

maximum patch size = img.width/2 - 1
maximum stride = img.height
where img is the original image
Setting patch and stride as their maximum values results in getting two patches per image - left and right patch
'''
OUTPUTS_DIR = 'output'
if not os.path.exists(OUTPUTS_DIR):
	os.makedirs(OUTPUTS_DIR)
done = transform_dataset(3,120,72,"output/train_holopix50k_small.h5","train")#patch and stride are chosen according to Holopix50k's image size
transform_dataset(3,120,72,"output/val_holopix50k_small.h5","validation")

