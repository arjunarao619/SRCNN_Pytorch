import numpy as np
import skimage.filters as filters
import skimage
import PIL.Image as pil
import glob
import torchvision.transforms as transforms
from PIL import ImageFilter as IF
import cv2

train_dir = "/Volumes/Transcend/Holopix50k/Holopix50k/train/left" #stored dataset in external hard-drive


def rgb_to_y(rgbimg):
    ''' Conversion formula:
    Y= 16 + 65.738*R/256 + 129.057*G/256 + 25.064*B/256
    Cb = 128-37.945*R/256 - 74.494*G/256 + 112.439*B/256
    Cr = 128+112.439*R - 94.154*G/256 - 18.285*B/256
    '''
    return 16. + (64.738 * rgbimg[:, :, 0] + 129.057 * rgbimg[:, :, 1] + 25.064 * rgbimg[:, :, 2]) / 256.
def rgb_to_ycbcr(rgbimg):
    y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
    cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])

def apply_gaussian_kernel(img,sigma):
    #img = np.asarray(img).astype('uint8')
    blurred = img.filter(IF.GaussianBlur(sigma))
        
    #blurred = skimage.filters.gaussian(img, sigma=(sigma, sigma), truncate=3.5, multichannel=True)
    # kernel = [sigma,sigma]
    # blurred = cv2.filter2D(img,-1,kernel)
    # blurred = pil.fromarray(img)

    return blurred
def centre_crop(img,scale):
    transform_data = transforms.Compose
    (
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    crop = transforms.CenterCrop(((img.size[1]//scale)*scale,(img.size[0]//scale)*scale))
    img = crop(img)
    return img
    


if (__name__ == "__main__"):
#         for img_path in sorted(glob.glob('{}/*'.format(train_dir))):
#             if(img_path != '.DS_Store'):
#                 img = pil.open(img_path).convert('RGB')
                # blurred = apply_gaussian_kernel(img,0.73) #sigma in paper was 0.55. We increase kernel size for Holopix50k
                # blurred.save("blurred.png")
                # break;
                # img = centre_crop(img,3)
                # img.save("cropped.png")
                # break;
    