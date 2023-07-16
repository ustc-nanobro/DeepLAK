import torch
import pytest
from layer_new import GaussianEmbedLayer
 
num_feats = 4
num_gauss = 4
x_min = 0
x_max = 1.0
sigma = 0.5
mlp_hid_dim = [64,128,256]

def test_gaussian_embed_layer():
    batch_size=3
    num_points=4
    model=GaussianEmbedLayer(num_gauss, x_min, x_max, sigma, num_feats, mlp_hid_dim)
    x = torch.randn(batch_size, num_points, num_feats)
    output = model(x)
    #print(output.size())
    assert output.shape == (batch_size, mlp_hid_dim[-1])

def test_embed_layer():
    feat=[[[0.0,0.3333,0.6666,1]]]
    feat=torch.tensor(feat)
    model=GaussianEmbedLayer(num_gauss, x_min, x_max, sigma, num_feats, mlp_hid_dim)
    x_gauss=model.embed(feat)
    is_symmetric = torch.allclose(x_gauss, x_gauss.transpose(-1, -2),rtol=1e-3, atol=1e-3)
    print("whether it is a symmetric matrix",is_symmetric)


test_embed_layer()
test_gaussian_embed_layer()