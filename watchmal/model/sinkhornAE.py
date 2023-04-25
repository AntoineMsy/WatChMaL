import torch.nn as nn
import torch
import torch.nn.functional as F
from watchmal.model.resnet import ResNet
from watchmal.model.resnet_encoder import resnet18encoder
from watchmal.model.resnet_decoder import resnet18decoder

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
    
def conv1x1(in_planes, out_planes, stride=1, deconv = False):
    """1x1 convolution"""
    if deconv : 
        return nn.ConvTranspose2d(out_planes, in_planes, kernel_size=1, stride=stride, bias=False)
    else :
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, padding_mode='zeros', deconv = False):
    """3x3 convolution with padding"""
    if deconv : 
        return nn.ConvTranspose2d(out_planes, in_planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode=padding_mode)
    else : 
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode=padding_mode)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, conv_pad_mode='zeros', deconv = False):
        super(BasicBlock, self).__init__()
        self.deconv = deconv
        self.conv1 = conv3x3(inplanes, planes, stride, conv_pad_mode, deconv = self.deconv)
        if deconv :
            self.bn1 = self.bn1 = nn.BatchNorm2d(inplanes)
        else : 
            self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = conv3x3(planes, planes, padding_mode=conv_pad_mode, deconv = self.deconv)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.deconv :
            out = self.conv2(x)
            out = F.leaky_relu(self.bn2(out))
          
            out = self.conv1(out)
            out = self.bn1(out)
    
        else : 
            out = self.conv1(x)

            out = self.bn1(out)
            out = F.leaky_relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.leaky_relu(out)

        return out

class BasicEncoder(nn.Module):
    def __init__(self, input_size, num_classes, img_size_x, img_size_y, in_channels, lat_dim, img_channels):
        super(BasicEncoder,self).__init__()
        self.img_size_x = img_size_x
        self.img_size_y = img_size_y
        self.d = in_channels
        self.input_size = input_size
        self.img_channels = img_channels
        self.lat_dim = lat_dim
        self.num_classes = num_classes
        #encoder layers
        self.conv1 = nn.Conv2d(in_channels=self.img_channels, out_channels=self.d//4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(self.d//4)
        self.conv2 = nn.Conv2d(self.d//4, self.d//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(self.d//2)
        self.conv3 = nn.Conv2d(self.d//2, self.d, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(self.d)
        self.conv4 = nn.Conv2d(in_channels=self.d, out_channels=self.d, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_4 = nn.BatchNorm2d(self.d)
        self.fc3 = nn.Linear((self.img_size_x//8)*(self.img_size_y//8)*self.d, self.lat_dim)

        #Conv blocks in between upsampling convs
        self.block1 = BasicBlock(self.d//4,self.d//4)
        self.block11 = BasicBlock(self.d//4,self.d//4)
        self.block2 = BasicBlock(self.d//2,self.d//2)
        self.block22 = BasicBlock(self.d//2,self.d//2)

    def forward(self,x):
         #Encoder part
        x = self.conv1(x)
        x = F.leaky_relu(self.bn_1(x))
        x = self.block1(x)
        x = self.block11(x)
        x = self.conv2(x)
        x = F.leaky_relu(self.bn_2(x))
        x = self.block2(x)
        x = self.block22(x)
        x = self.conv3(x)
       
        x = F.leaky_relu(self.bn_3(x))
        x = self.conv4(x)
        x = F.leaky_relu(self.bn_4(x))
        x = x.view(-1,(self.img_size_x//8)*(self.img_size_y//8)*self.d)

        y = self.fc3(x)

        return y

class BasicDecoder(nn.Module):
    def __init__(self, input_size, num_classes, img_size_x, img_size_y, in_channels, lat_dim, img_channels):
        super(BasicDecoder, self).__init__()
        self.img_size_x = img_size_x
        self.img_size_y = img_size_y
        self.d = in_channels
        self.input_size = input_size
        self.img_channels = img_channels
        self.lat_dim = lat_dim
        self.num_classes = num_classes

        #decoder blocks
        self.fc1 = nn.Linear(self.lat_dim, (self.img_size_x//8)*(self.img_size_y//8)*self.d)
        self.dc0 = nn.ConvTranspose2d(self.d, self.d, 3,1,1, bias=False)
        self.dc0_bn = nn.BatchNorm2d(self.d)
        self.dc1 = nn.ConvTranspose2d( self.d, self.d//2, 4, 2, 1, bias=False)
        self.dc1_bn = nn.BatchNorm2d(self.d//2)
        self.dc2 = nn.ConvTranspose2d( self.d//2, self.d//4, 4, 2, 1, bias=False)
        self.dc2_bn = nn.BatchNorm2d(self.d//4)
        self.dc3 = nn.ConvTranspose2d(self.d//4 , self.img_channels , 4, 2, 1, bias=False)


        #Conv blocks in between upsampling convs
        self.block1 = BasicBlock(self.d//4,self.d//4)
        self.block11 = BasicBlock(self.d//4,self.d//4)
        self.block2 = BasicBlock(self.d//2,self.d//2)
        self.block22 = BasicBlock(self.d//2,self.d//2)

        self.dblock1 = BasicBlock(self.d//2,self.d//2, deconv = True)
        self.dblock11 = BasicBlock(self.d//2,self.d//2, deconv = True)
        self.dblock2 = BasicBlock(self.d//4, self.d//4, deconv = True)
        self.dblock22 = BasicBlock(self.d//4, self.d//4, deconv = True)
        #self.dc4 = nn.ConvTranspose2d( self.d , 1, 3, 4, 1, bias=False)    

    def forward(self, y):
         #Decoder
        x = F.leaky_relu(self.fc1(y))
        x = x.view(-1,self.d,(self.img_size_x//8),(self.img_size_y//8))
        x = self.dc0(x)
        x = F.leaky_relu(self.dc0_bn(x))
        x = self.dc1(x)
        x = F.leaky_relu(self.dc1_bn(x))
        x = self.dblock1(x)
        x = self.dblock11(x)
        x = self.dc2(x)
        x = F.leaky_relu(self.dc2_bn(x))
        x = self.dblock2(x)
        x = self.dblock22(x)
        x = self.dc3(x)

        return x

class Autoencoder(nn.Module):
    def __init__(self, input_size, num_classes, img_size_x, img_size_y, in_channels, lat_dim, img_channels):
        super(Autoencoder, self).__init__()
        self.img_size_x = img_size_x
        self.img_size_y = img_size_y
        self.d = in_channels
        self.input_size = input_size
        self.img_channels = img_channels
        self.lat_dim = lat_dim
        self.num_classes = num_classes
        
        #self.resnet_encoder = ResNet(BasicBlock, [2, 2, 2, 2],self.img_channels, self.lat_dim)
        self.resnet_encoder = resnet18encoder(**{"num_input_channels": self.img_channels ,"num_output_channels": self.lat_dim})
        self.resnet_decoder = resnet18decoder(**{"num_input_channels": self.img_channels ,"num_output_channels": self.lat_dim, "img_size_x": self.img_size_x, "img_size_y" : self.img_size_y})
        self.encoder = BasicEncoder(input_size, num_classes, img_size_x, img_size_y, in_channels, lat_dim, img_channels)
        self.decoder = BasicDecoder(input_size, num_classes, img_size_x, img_size_y, in_channels, lat_dim, img_channels)
      
############        noise_gen
        self.ng_fc1 = nn.Linear(self.input_size, self.d//2)
        self.ng_input_2 = nn.Linear(self.num_classes+6,self.d//4)
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
        y = self.resnet_encoder(x)
        
        x = self.resnet_decoder(y)
        return x, y
    
    def classify(self,x):
        #Duplicate encoder part with changed FC end layer
        x,y = self.encoder(x)
        x = self.fcc1(x)
        return x
        
    def generate(self,x):
        #Duplicate decoder part
        return self.decoder(x)
    
    def generate_noise(self, x, cond_x):
        #Noise generator MLP
        x = F.leaky_relu(self.ng_fc1(x), 0.2)
        x2 = F.leaky_relu(self.ng_input_2(cond_x), 0.2)
        x_concat = torch.cat((x,x2),1)
        x = F.leaky_relu(self.ng_fc2(x_concat), 0.2)
        x = F.leaky_relu(self.ng_fc3(x), 0.2)
        x = self.ng_fc4(x)
        return x

"""
from watchmal.dataset.cnn_mpmt.cnn_mpmt_dataset import CNNmPMTDataset

cnn_dataset = CNNmPMTDataset(h5file='/gpfs02/work/pdeperio/machine_learning/data/IWCD_mPMT_Short/IWCD_mPMT_Short_emgp0_E0to1000MeV_digihits.h5',
                                    mpmt_positions_file='/gpfs02/work/pdeperio/machine_learning/data/IWCDshort_mPMT_image_positions.npz')

train_dataset, test_dataset, val_dataset = random_split(cnn_dataset, [17611162,4696309,1174078])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 20)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 20)
"""