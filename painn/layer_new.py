import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
class GaussianEmbedLayer(nn.Module):
    def __init__(self, num_gauss, x_min, x_max, sigma, num_feats, mlp_hid_dim):
        super().__init__()
        self.num_gauss = num_gauss
        self.x_min = x_min
        self.x_max = x_max
        self.sigma = sigma
        self.num_feats = num_feats
        self.mlp_hid_dim=mlp_hid_dim

        if mlp_hid_dim:
            mlp_layers = []
            for i in range(len(mlp_hid_dim) - 1):
                mlp_layers.append(nn.Linear(mlp_hid_dim[i], mlp_hid_dim[i+1]))
                mlp_layers.append(nn.ReLU())
            self.mlp = nn.Sequential(*mlp_layers)
        else:
            self.mlp = None

    def gaussian(self, x, mean):
        return torch.exp(-(x - mean) ** 2 / (2 * self.sigma ** 2))

    def embed(self, x):
        batch_size, num_points, num_feats = x.size()

        #将输入重塑为 (batch_size * num_points, num_feats) 的形状
        
        x = einops.rearrange(x, 'b n f -> (b n) f')
        
        # 对输入向量的每个元素应用高斯函数
        means = torch.linspace(self.x_min, self.x_max, self.num_gauss).unsqueeze(0)  # (1, num_gauss)
       
        #x_gauss = self.gaussian(x, means)
        x_gauss = self.gaussian(x.unsqueeze(2), means)  # (batch_size * num_points, num_feats, num_gauss)
        
        # 将形状重新调整为 (batch_size, num_points, num_feats, num_gauss)
        x_gauss = x_gauss.view(batch_size, num_points, num_feats, self.num_gauss)
        #x_gauss = einops.rearrange(x_gauss, '(b n) f g -> b n f g')
        return x_gauss
        
    def forward(self, x):
        x_gauss = self.embed(x)
        
        #print(x_gauss.size())
        x_gauss = einops.rearrange(x_gauss, 'b n f g -> b (n f g)')  # Reshape to (batch_size, num_points * num_feats * num_gauss)
        x_gauss = self.mlp(x_gauss)
        
        
        return x_gauss


  