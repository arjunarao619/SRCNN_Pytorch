import numpy as np
import torch
import PIL.Image as pil_image
from utils.utils import psnr
import utils.pytorch_ssim
import torchvision.transforms as transforms
import math

def weighted_loss(original,compressed):
    '''
    Original and compressed are torch tensors
    You may change the weightage given to each metric. Current loss:
    0.4 = MSE
    0.5 = PSNR
    0.1 = SSIM
    '''
    
    mse = torch.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal .# Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse)) 
    psnr = 100 - psnr #PSNR is maximized so 100-PSNR is a loss function (Or -PSNR)
    print(psnr)

    ssim_score = utils.pytorch_ssim.ssim(original,compressed)
    ssim_score = 1 - ssim_score


    weighted_loss = (0.4 * mse) + (0.5 * (psnr/100)) + (0.1 * ssim_score)

    return weighted_loss

'''
sanity check
'''
# if __name__ == '__main__':
#     transform1 = transforms.Compose([
#             transforms.ToTensor()
#         ]
#     )
#     original = pil_image.open('results/flower_srcnn_30.png').convert('RGB')
#     compressed = pil_image.open('results/flower_bicubic_30.png').convert('RGB')
#     print(weighted_loss(transform1(original),transform1(compressed)))








