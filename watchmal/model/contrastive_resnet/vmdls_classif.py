import torch
import torch.nn as nn
from watchmal.model.contrastive_resnet.gmm_vae import GMM_VAE_Contrastive

class VMDLS_Classifier(nn.Module):
    def __init__(self, class_num, enc_type, latent_dim, channels, model_file_path):
        super(VMDLS_Classifier,self).__init__()
        state_dict = torch.load(model_file_path)["state_dict"]
        self.class_num = class_num
        self.enc_type = enc_type
        self.latent_dim = latent_dim
        self.channels = channels
        self.model_file_path = model_file_path
        self.feature_extractor = GMM_VAE_Contrastive(class_num = self.class_num, enc_type=self.enc_type, latent_dim=self.latent_dim, channels=self.channels )
        self.feature_extractor.load_state_dict(state_dict)
        self.layer_size = self.feature_extractor.enc_out_dim + self.feature_extractor.latent_dim
        

        if self.feature_extractor.enc_type == 'resnet18':
            self.layer_size += 448
        elif self.feature_extractor.enc_type == 'wresnet':
            self.layer_size += 1120
        elif self.feature_extractor.enc_type == "resnet50":
            self.layer_size += 1792
        self.layer_size = 32
        num_inputs = self.layer_size
        self.cl_fc1 = nn.Linear(num_inputs, int(num_inputs // 2))
        self.cl_fc2 = nn.Linear(int(num_inputs // 2), int(num_inputs // 4))
        self.cl_fc3 = nn.Linear(int(num_inputs // 4), int(num_inputs // 8))
        self.cl_fc4 = nn.Linear(int(num_inputs // 8), self.class_num)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Sequential(self.cl_fc1, self.relu, self.cl_fc2, self.relu, self.cl_fc3, self.relu, self.cl_fc4)
    
    def forward(self, x):
        with torch.no_grad():
            mu= self.feature_extractor.fc_mu(self.feature_extractor.encoder(x))
            z = mu
            # l_features = [torch.flatten(self.feature_extractor.encoder.get_layer_output(x,i),1) for i in range(1,5)] + [mu]
            # l_features = torch.hstack(l_features)
            l_features = mu
        out = self.classifier(l_features)
        return out

