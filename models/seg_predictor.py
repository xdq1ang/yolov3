import torch.nn as nn
import torch
from torch.nn import functional as F 

class Seg_Predictor(nn.Module):
    def __init__(self, in_channel, num_class):
        super(Seg_Predictor, self).__init__()
        self.u1=nn.Sequential(
            nn.Conv2d(in_channel,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
  
        )
        self.u2=nn.Sequential(
            nn.Conv2d(64 + 512, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.u3=nn.Sequential(
            nn.Conv2d(64 + 256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.o=nn.Sequential(
            nn.Conv2d(64,num_class,3,1,1)
        )
    def forward(self,x):
        f, f8, f6= x[0], x[1], x[2]
        f = self.u1(f)
        f = torch.concat([F.interpolate(f, scale_factor=2, mode="bilinear"), f8], dim=1)
        f = self.u2(f)
        f = torch.concat([F.interpolate(f, scale_factor=2, mode="bilinear"), f6], dim=1)
        f = self.u3(f)
        f = F.interpolate(f, scale_factor=8, mode="bilinear")
        f = self.o(f)
        return f
# if __name__ == '__main__':

#     test_data=torch.randn(4,1024,256,256)
#     net=Seg_Predictor(1024,10)
#     print(net)
#     out=net(test_data)
#     print(out.shape)
