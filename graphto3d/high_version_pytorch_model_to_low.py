import torch

# Load the state_dict with the new version of PyTorch
state_dict = torch.load('network.pth')

# Save the state_dict with an older file format
torch.save(state_dict, 'network_old_version.pth', _use_new_zipfile_serialization=False)