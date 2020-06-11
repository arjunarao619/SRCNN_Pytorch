import numpy as np
import PIL.Image as pil 
from PIL import ImageFilter as pilfilter
import os
import torch
import glob
import h5py
import torchvision
import torchvision.transforms as transforms
import numpy as np
import skimage.io as io
import torch.utils.data as data
from utils import apply_gaussian_kernel,rgb_to_y,centre_crop



def transform_dataset(scale, hrlr,patch,stride,output_path):
    train_dir = "/home/ec2-user/arjun/SRCNN_pytorch/Holopix50k/train_mini" #only 91 images 
    temp_list = []
    lr_patches = []
    hr_patches = []
    h5_file = h5py.File(output_path, 'w')
    
    

    for img_path in sorted(glob.glob('{}/*'.format(train_dir))):
        input_hr = pil.open(img_path).convert('RGB')
        crop = centre_crop(input_hr,1)#no crop
        hr = crop

        input_lr = crop.resize((crop.width // scale, crop.height // scale),resample = pil.BICUBIC)
        lr = input_lr.resize((input_lr.width * scale, input_lr.height * scale),resample = pil.BICUBIC)
        #print(lr.size)
        #apply gaussian blurring
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
                print("LR : " + str(len(lr_patches)) + " HR: " + str(len(hr_patches)))
                #from IPython import embed;embed()

#         if(len(lr_patches) % 50 == 0):
#             temp_patch_lr = np.array(lr_patches)
#             temp_path_hr = np.array(hr_patches)
#             from IPython import embed;embed()
#             h5_file.create_dataset('lr_' + str(len(lr_patches)),data = temp_patch_lr)
#             h5_file.create_dataset('hr_' + str(len(hr_patches)),data = temp_patch_hr)
            

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    
    print(lr_patches.shape)

    h5_file.create_dataset('lr',data = lr_patches)
    h5_file.create_dataset('hr',data = hr_patches)    

    h5_file.close()    



transform_dataset(3,'lr',231,98,"output/train.h5")#patch and stride are chosen according to Holopix50k's image size

