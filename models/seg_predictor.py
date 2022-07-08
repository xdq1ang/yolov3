import torch.nn as nn
import torch
from torch.nn import functional as F 
from models.PPM import PPM

class Seg_Predictor(nn.Module):
    def __init__(self, in_channel, num_class):
        super(Seg_Predictor, self).__init__()
        pool_size = [1,2,3,6]
        self.PPM = PPM(in_dim = in_channel, reduction_dim= 64 , bins=pool_size) 
        PPM_out_channel = in_channel+64*(len(pool_size))
        self.u1=nn.Sequential(
            nn.Conv2d(PPM_out_channel,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
  
        )
        self.u2=nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.o=nn.Sequential(
            nn.Conv2d(64,num_class,3,1,1)
        )
    def forward(self,x):
        x = self.PPM(x)
        x = self.u1(x)
        x = F.interpolate(x, scale_factor=4, mode="bilinear")
        x = self.u2(x)
        x = F.interpolate(x, scale_factor=8, mode="bilinear")
        x = self.o(x)
        return x
# if __name__ == '__main__':

#     test_data=torch.randn(4,1024,256,256)
#     net=Seg_Predictor(1024,10)
#     print(net)
#     out=net(test_data)
#     print(out.shape)
