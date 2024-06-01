from __future__ import print_function
import sys
import os
atlasnet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', '..'))#/home/caixiaoni/Desktop/master_thesis/AtlasNet
if atlasnet_dir not in sys.path:
    sys.path.append(atlasnet_dir)
sys.path.append('./auxiliary/')

import argparse
import random
import numpy as np
import torch
import sys
sys.path.append('./auxiliary/')
from auxiliary.datasetCustom import *
from auxiliary.model import *
from auxiliary.my_utils import *
from auxiliary.ply import *
import os
import json
import pandas as pd
opt = lambda : None
opt.model = os.path.join(atlasnet_dir, 'log/atlasnet_separate_cultery/network.pth')
opt.num_points = 5625
opt.bottleneck_size = 128
opt.nb_primitives = 25
opt.workers = 6
filepath, fullfilename = os.path.split(opt.model)
opt.objs_features_gt = os.path.join(atlasnet_dir, 'log/atlasnet_separate_cultery/objs_features_gt_atlasnet_separate_cultery.json')
# ===================CREATE network================================= #
network = AE_AtlasNet(num_points = opt.num_points, bottleneck_size = opt.bottleneck_size, nb_primitives = opt.nb_primitives)
network.cuda()
network.apply(weights_init)
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print("previous weight loaded")
network.eval()

dataset_test = ShapeNet(train=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                          shuffle=False, num_workers=int(opt.workers))
with torch.no_grad():
    dict_out = {}
    for i, data in enumerate(dataloader_test, 0):
        obj_points, cat , objpath, fn = data
        pf = torch.from_numpy(np.array(list(obj_points.numpy()), dtype=np.float32)).float().cuda().transpose(1,2)

        feats = network.encoder(pf).detach().cpu().numpy()
        print(cat[0], fn[0].split('.')[0], feats.shape)
        dict_out[fn[0].split('.')[0]] = feats.tolist()
    print(len(dataloader_test))
    with open(opt.objs_features_gt , 'w+') as fp:
        json.dump(dict_out, fp, indent=2, sort_keys=True)