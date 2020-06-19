import numpy as np
import skimage.filters as filters
import skimage
import PIL.Image as pil
import glob
from torchvision import transforms
from PIL import ImageFilter as IF
import torch
import math
import utils.pytorch_ssim
#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

train_dir = "/Volumes/Transcend/Holopix50k/Holopix50k/train/left" #stored dataset in external hard-drive


def rgb_to_y(rgbimg):
    ''' Conversion formula:
    Y= 16 + 65.738*R/256 + 129.057*G/256 + 25.064*B/256
    Cb = 128-37.945*R/256 - 74.494*G/256 + 112.439*B/256
    Cr = 128+112.439*R - 94.154*G/256 - 18.285*B/256
    '''
    return 16. + (64.738 * rgbimg[:, :, 0] + 129.057 * rgbimg[:, :, 1] + 25.064 * rgbimg[:, :, 2]) / 256.
def rgb_to_ycbcr(img):
    y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
    cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])

def apply_gaussian_kernel(img,sigma):

    blurred = img.filter(IF.GaussianBlur(sigma))
    return blurred

def centre_crop(img,scale):
    
    crop = transforms.CenterCrop(((img.size[1]//scale)*scale,(img.size[0]//scale)*scale))
    img = crop(img)
    return img
    
def psnr(original, compressed): 
    mse = torch.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal .# Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr 
def ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


# if __name__ == "__main__":
    # transform1 = transforms.Compose([
    #         transforms.ToTensor()
    #     ]
    # )
'''
Sanity check for MSSIM and SSIM score
'''
    # hr = pil.open('../results/coffee_srcnn_30.png').convert('RGB')
    # lr = pil.open('../results/coffee_bicubic_30.png').convert('RGB')
    # lr = transform1(lr)
    # hr = transform1(hr)
    # print(pytorch_ssim.ssim(torch.unsqueeze(lr,0),torch.unsqueeze(hr,0)))
'''
Sanity Check for PSNR (Peak Signal to Noise Ratio) score
'''
#     hr = pil.open('../images/hr.png').convert('RGB')
#     lr = pil.open('../images/lr.png').convert('RGB')
#     lr = transform1(lr)
#     hr = transform1(hr)
#     print(psnr(lr,hr))
'''
To check Gaussian Kernel effect. Output at </output/lr.png>
'''
#     img_path = '../images/hr.png'
#     img = pil.open(img_path).convert('RGB')
#     blurred = apply_gaussian_kernel(img,0.73) #sigma in paper was 0.55. We increase kernel size for Holopix50k
#     blurred.save("../images/blurred.png")
    
