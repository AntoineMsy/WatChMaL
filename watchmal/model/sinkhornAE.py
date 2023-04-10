import torch.nn as nn
import torch
import torch.nn.functional as F
from resnet import BasicBlock

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
    
class Autoencoder(nn.Module):
    def __init__(self, g_input_dim, cond_dim, img_size, in_channels, lat_dim):
        super(Autoencoder, self).__init__()
        self.img_size = img_size
        self.d = in_channels
        self.lat_dim = lat_dim
        self.conv1 = nn.Conv2d(in_channels=19, out_channels=self.d//4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(self.d//4)
        self.conv2 = nn.Conv2d(self.d//4, self.d//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(self.d//2)
        self.conv3 = nn.Conv2d(self.d//2, self.d, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(self.d)
        self.fc3 = nn.Linear((self.img_size[0]//8)*(self.img_size[1]//8)*self.d, self.lat_dim)

        self.fc1 = nn.Linear(self.lat_dim, (self.img_size[0]//8)*(self.img_size[1]//8)*self.d)
        self.dc1 = nn.ConvTranspose2d( self.d, self.d//2, 4, 2, 1, bias=False)
        self.dc1_bn = nn.BatchNorm2d(self.d//2)
        self.dc2 = nn.ConvTranspose2d( self.d//2, self.d//4, 4, 2, 1, bias=False)
        self.dc2_bn = nn.BatchNorm2d(self.d//4)
        self.dc3 = nn.ConvTranspose2d(self.d//4 , 19 , 4, 2, 1, bias=False)
        self.dc3_bn = nn.BatchNorm2d(1)

        #Conv blocks in between upsampling convs
        self.block1 = BasicBlock(self.d//4,self.d//4)
        self.block2 = BasicBlock(self.d//2,self.d//2)

        self.dblock1 = BasicBlock(self.d//2,self.d//2, deconv = True)
        self.dblock2 = BasicBlock(self.d//4, self.d//4, deconv = True)
        #self.dc4 = nn.ConvTranspose2d( self.d , 1, 3, 4, 1, bias=False)    
        

############        noise_gen
        self.ng_fc1 = nn.Linear(g_input_dim, self.d//2)
        self.ng_input_2 = nn.Linear(cond_dim,self.d//4)
        self.ng_fc2 = nn.Linear(self.d*3//4, self.d)
        self.ng_fc3 = nn.Linear(self.d, self.d*3//4)
        self.ng_fc4 = nn.Linear(self.d*3//4, self.lat_dim)
###############

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif (classname.find('BatchNorm') != -1):#|(classname.find('Linear') != -1):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    # forward method
    def forward(self, x):
        #Encoder part
        x = self.conv1(x)
        x = F.leaky_relu(self.bn_1(x))
        x = self.block1(x)
        x = self.conv2(x)
        x = F.leaky_relu(self.bn_2(x))
        x = self.block2(x)
        x = self.conv3(x)
        x = F.leaky_relu(self.bn_3(x))
        x = x.view(-1,(self.img_size[0]//8)*(self.img_size[1]//8)*self.d)

        #Latent vector
        y = self.fc3(x)

        #Decoder
        x = F.leaky_relu(self.fc1(y))
        x = x.view(-1,self.d,(self.img_size[0]//8),(self.img_size[1]//8))
        x = self.dc1(x)
        x = F.leaky_relu(self.dc1_bn(x))
        x = self.dblock1(x)
        x = self.dc2(x)
        x = F.leaky_relu(self.dc2_bn(x))
        x = self.dblock2(x)
        x = self.dc3(x)

        #x = F.leaky_relu(self.dc3_bn(x))

        return torch.tanh(x), y
    
    def generate(self,x):
        #Duplicate decoder part
        x = F.leaky_relu(self.fc1(x))
        x = x.view(-1,self.d,(self.img_size[0]//8),(self.img_size[1]//8))
        x = self.dc1(x)
        x = F.leaky_relu(self.dc1_bn(x))
        x = self.dblock1(x)
        x = self.dc2(x)
        x =F.leaky_relu(self.dc2_bn(x))
        x = self.dblock2(x)
        x = self.dc3(x)
        #x = F.leaky_relu(self.dc3_bn(x))
        return torch.tanh(x)
    
    def generate_noise(self, x, cond_x):
        #Noise generator MLP
        x = F.leaky_relu(self.ng_fc1(x), 0.2)
        x2 = F.leaky_relu(self.ng_input_2(cond_x), 0.2)
        x_concat = torch.cat((x,x2),1)
        x = F.leaky_relu(self.ng_fc2(x_concat), 0.2)
        x = F.leaky_relu(self.ng_fc3(x), 0.2)
        x = self.ng_fc4(x)
        return x
    
from watchmal.dataset.cnn_mpmt.cnn_mpmt_dataset import CNNmPMTDataset

cnn_dataset = CNNmPMTDataset(h5file='/gpfs02/work/pdeperio/machine_learning/data/IWCD_mPMT_Short/IWCD_mPMT_Short_emgp0_E0to1000MeV_digihits.h5',
                                    mpmt_positions_file='/gpfs02/work/pdeperio/machine_learning/data/IWCDshort_mPMT_image_positions.npz')

train_dataset, test_dataset, val_dataset = random_split(cnn_dataset, [17611162,4696309,1174078])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 20)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 20)