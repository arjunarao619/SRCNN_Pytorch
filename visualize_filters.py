import h5py
from model import SRCNN_915,SRCNN_955
from torchvision.utils import make_grid,save_image
import imageio
import torch
import numpy as np


def visualize_conv_filters(weights):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN_955().to(device)
    state_dict = model.state_dict()

    checkpoint = torch.load('saved_weights/testing/epoch_{}.pth'.format(30),map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()

    layer = 2#0,1,or 2
    filter = model.conv1.weight.detach().clone()
    print(filter.size())

    filter = filter - filter.min()
    filter = filter / filter.max()   

    filter_img = make_grid(filter,nrow = 8)
    #plt.imshow(filter_img.permute(1,2,0))
    img = save_image(filter,"fil_l0.png",nrow = 8)

visualize_conv_filters('saved_weights/testing/epoch_0.pth')
