import torch
import torch
import torch.optim as optim
import sys
import os
atlasnet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', '..'))#/home/caixiaoni/Desktop/master_thesis/AtlasNet
if atlasnet_dir not in sys.path:
    sys.path.append(atlasnet_dir)
sys.path.append('./auxiliary/')
print(torch.__version__)
print(torch.cuda.is_available())

path = os.path.join(atlasnet_dir,'trained_models','model_70.pth')
print(path)
print(os.path.isfile(path))
""" network = AE_AtlasNet(num_points = 5120, bottleneck_size = 128, nb_primitives =25)
network.cuda() """