import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from model import SRCNN_955
from torchvision import transforms
import utils.utils




def eval(image_path,range_epochs,scale = 3):
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    model = SRCNN_955().to(device)
    for i in range(0,range_epochs):

        checkpoint = torch.load('saved_weights/testing/epoch_{}.pth'.format(i),map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        
        print("########## Evaluating on weights from epoch #{} ################".format(i))
        

        model.eval()

        image = pil_image.open(image_path).convert('RGB')

        image_width = (image.width // scale) * scale
        image_height = (image.height // scale) * scale
        original_img = image.copy()
        image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        image = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
        image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)
        image.save(image_path.replace('.', '_bicubic_x{}.'.format(scale)))

        image = np.array(image).astype(np.float32)
        ycbcr = utils.utils.rgb_to_ycbcr(image)

        y = ycbcr[..., 0]
        y /= 255.
        y = torch.from_numpy(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            preds = model(y).clamp(0.0, 1.0)
        psnr_score = utils.utils.psnr(preds,y)
        print('PSNR_SRCNN_epoch{}: {:.2f}'.format(i, psnr_score))

        transform1 = transforms.Compose([
            transforms.ToTensor()
            ])

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(utils.utils.ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        output.save(image_path.replace('.', '_srcnn_x{}.'.format(i)))

        



eval('results/original3.png',24)
