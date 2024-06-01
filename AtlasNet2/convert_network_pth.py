import trimesh
import random
import numpy as np
import torch
path = '/media/storage/guangyao/caixiaoni/AtlasNetV2.2/AtlasNet2/log/AE_AtlasNet2_20230405T2156/atlasnet2.pth'
import torch

# Load the model with the new version of PyTorch
model = torch.load(path)

# Save the model with an older file format
torch.save(model, 'atlasnet2_converted.pth', _use_new_zipfile_serialization=False)

