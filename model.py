from torch import nn
import torch.nn.functional as F
import torch
'''
9-5-5 chosen over 9-1-5 architecture due to larger number of patches ~24000 for 91-image dataset vs ~100,000 for Holopix50k (1500 images)

f1 = 9, f2 = 1, f3 = 5, n1 = 64, and n2 = 32 **(although can try 9-5-5 with more compute)**
in_channels = 1 as we only consider luminance
'''

class SRCNN_915(nn.Module):
    def __init__(self,device=torch.device('cpu')):
        super(SRCNN_915,self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size = 9, padding=int(9/2))
        self.conv2 = nn.Conv2d(64,32,kernel_size = 1,padding= int(1/2))
        self.conv3 = nn.Conv2d(32,1,kernel_size = 5, padding = int(5/2))
        
    def forward(self,Y):
        Y = F.relu((self.conv1(Y)),inplace = True)
        Y = F.relu((self.conv2(Y)),inplace = True)
        Y = F.relu((self.conv3(Y)),inplace = True)
        return Y
        
class SRCNN_955(nn.Module):
    def __init__(self,device=torch.device('cpu')):
        super(SRCNN_955,self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size = 9,padding=int(9/2))
        self.conv2 = nn.Conv2d(64,32,kernel_size = 5,padding=int(5/2))
        self.conv3 = nn.Conv2d(32,1,kernel_size = 5,padding=int(5/2))
    def forward(self,Y):
        Y = F.relu((self.conv1(Y)),inplace = True)
        Y = F.relu((self.conv2(Y)),inplace = True)
        Y = F.relu((self.conv3(Y)),inplace = True)
        return Y
        