import torch
import torch.nn as nn
from watchmal.engine.losses.modded_triplets import TripletMarginLossModded
import torch.nn.functional as F
from sklearn.metrics import f1_score,auc

#from models.decoders_encoders import *
#from losses.losses import SupConLoss
#from losses.mmd import compute_mmd
#import kornia
#from models.wide_resnet import WideResNet


from watchmal.model.contrastive_resnet.resnet_encoders import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)

# ae_dict = {
#     'cifar_known':(lambda:AE(input_height=32), 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
#     'cifar100_known':(lambda:AE(input_height=32,enc_type='resnet50'), 'models/ae_cifar100.ckpt'),
#     'mnist':(lambda:MNIST_AE(input_height=28,enc_type='mnist'),'models/ae_mnist.ckpt')
# }



class GMM_VAE_Contrastive(nn.Module):   
    def __init__(
        self,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        latent_dim: int = 32,
        channels: int=1,
        class_num: int = 10,
        use_sinkhorn = False,
        **kwargs
    ):

        super(GMM_VAE_Contrastive, self).__init__()
   
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.class_num = class_num
        self.input_size = 128
        self.d = 1024
        self.use_sinkhorn = use_sinkhorn
        valid_encoders = {
            'resnet18': {'enc': resnet18_encoder, 'dec': resnet18_decoder},
            'resnet50': {'enc': resnet50_encoder, 'dec': resnet50_decoder},
        }
       
        if enc_type == "resnet_18":
            self.enc_out_dim = 512
        elif enc_type == "resnet50":
            self.enc_out_dim = 2048
            
        if enc_type not in valid_encoders:
            self.encoder = resnet18_encoder(first_conv, maxpool1,channels)
        
        else:
            self.encoder = valid_encoders[enc_type]['enc'](first_conv, maxpool1,channels)
        

        self.fc_out = nn.Linear(self.enc_out_dim, self.latent_dim)
        
        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

      
        
        self.enc_type = enc_type

        ############        noise_gen
        self.ng_fc1 = nn.Linear(self.input_size, self.d//2)
        self.ng_input_2 = nn.Linear(self.class_num+6,self.d//4)
        self.ng_fc2 = nn.Linear(self.d*3//4, self.d)
        self.ng_fc3 = nn.Linear(self.d, self.d*3//4)
        self.ng_fc4 = nn.Linear(self.d*3//4, self.latent_dim)
        ###############


    def forward(self, x):
        x = self.encoder(x)
        # return x
        if self.use_sinkhorn :
            z = self.fc_out(x)
            return z
        else :
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
            p, q, z = self.sample(mu, log_var)
            return z

    def _run_step(self, x):
        x = self.encoder(x)
        # return x
        mu = self.fc_mu(x)
        
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        # return z,q
        return z, mu, log_var, q

    def sample(self, mu, log_var):
        std_0 = torch.exp(log_var / 2)
        std = std_0 + 2.2250738585072014e-64
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        
        return p, q, z

    def generate_noise(self, x, cond_x):
        #Noise generator MLP
        x = F.leaky_relu(self.ng_fc1(x), 0.2)
        x2 = F.leaky_relu(self.ng_input_2(cond_x), 0.2)
        x_concat = torch.cat((x,x2),1)
        x = F.leaky_relu(self.ng_fc2(x_concat), 0.2)
        x = F.leaky_relu(self.ng_fc3(x), 0.2)
        x = self.ng_fc4(x)
        return x