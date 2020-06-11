import numpy as np
import PIL.Image as pil 
from PIL import ImageFilter as pilfilter
import os
import torch
import glob
import torchvision
import torchvision.transforms as transforms
import numpy as np
import skimage.io as io
import torch.utils.data as data
from utils import apply_gaussian_kernel,rgb_to_y,centre_crop



def transform_dataset(scale, hrlr,patch,stride):
    train_dir = "/Volumes/Transcend/Holopix50k/Holopix50k/train/left" #stored dataset in external hard-drive
    temp_list = []
    lr_patches = []
    
    
    for img_path in sorted(glob.glob('{}/*'.format(train_dir))):
        hr = 





        # if(img_path != '.DS_Store'):
        #     img = pil.open(img_path).convert('RGB') #we will convert to YCbCr Later
        
        #     if(hrlr == 'hr'):
        #         #For ground truth High resolution image, 
        #         # we just do a random centre cropping after adjusting image height and width according to scale
        #         hr_width = (img.width // scale) * scale
        #         hr_height = (img.width //scale) * scale

        #         img = img.resize((hr_width,hr_height),resample = pil.BICUBIC) #hr image resized as hr_width * hr_height
                

        #     if(hrlr == 'lr'):
        #         lr_width = (img.width // scale) * scale
        #         lr_height = (img.width //scale) * scale

        #         #First downscaling original HR image by the scale
        #         img = img.resize((lr_width,lr_height),resample = pil.BICUBIC)
        #         img = img.resize((lr_width // scale, lr_height // scale), resample=pil.BICUBIC)

        #         #Applying Gaussian Kernel
        #         img = 

        #         #Upscaling Y though bicubic-interpolation
        #         img = img.resize((img.width * scale, img.height * scale), resample=pil.BICUBIC)
                 
                
                
                

    

        




#from the pytorch implementation repo, we set patch and stride as 33 and 14

transform_dataset(3,'lr',33,14)

