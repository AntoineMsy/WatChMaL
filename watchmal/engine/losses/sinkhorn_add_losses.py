import torch
import torch.nn as nn

class SIMcLoss(nn.Module):
    def __init__(self, treshold = 1e-5, device=torch.device('cuda')):
        super(SIMcLoss, self).__init__()
        self.treshold = treshold
        self.cos_sim = nn.CosineSimilarity(dim=1)
    def forward(self, input):
        
        self.size = input.size()[0]
       
        self.simloss = 0
        """
        for i in range(self.size):
            for j in range(self.size):
                print(i,j)
                if j!=i:
                    simc = self.cos_sim(input[i],input[j])
                    if torch.abs(simc)> self.treshold:
                        self.simloss += simc**2
       """
        mat1= input.repeat_interleave(self.size,dim=0)
        mat2 = input.repeat(self.size,1)
        self.cos_mat = self.cos_sim(mat1,mat2).reshape(self.size,self.size)
        self.simloss = (torch.sum(torch.square(self.cos_mat))-self.size)/(self.size**2-self.size)
        return self.simloss
    
def gauss_kernel(size=5, device=torch.device('cuda'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1], device=x.device))

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr

class LapLoss(nn.Module):
    def __init__(self, max_levels=3, channels=3, device=torch.device('cuda')):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device)
        
    def forward(self, input, target):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))