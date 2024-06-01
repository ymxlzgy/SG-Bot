from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import sys
sys.path.append('./auxiliary/')
from dataset import *
from model import *
from my_utils import *
from ply import *
import os
import json
import pandas as pd

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--model', type=str, default = 'trained_models/svr_atlas_25.pth',  help='your path to the trained model')
parser.add_argument('--num_points', type=int, default = 2500,  help='number of points fed to poitnet')
parser.add_argument('--gen_points', type=int, default = 2500,  help='number of points to generate')
parser.add_argument('--nb_primitives', type=int, default = 25,  help='number of primitives')

opt = parser.parse_args()
print (opt)
# ========================================================== #



# =============DEFINE CHAMFER LOSS======================================== #
def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    P = (rx.t() + ry - 2*zz)
    return P


def NN_loss(x, y, dim=0):
    dist = pairwise_dist(x, y)
    values, indices = dist.min(dim=dim)
    return values.mean()


def distChamfer(a,b):
    x,y = a,b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2,1))
    yy = torch.bmm(y, y.transpose(2,1))
    zz = torch.bmm(x, y.transpose(2,1))
    diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2,1) + ry - 2*zz)
    return torch.min(P, 1)[0], torch.min(P, 2)[0], torch.min(P, 1)[1], torch.min(P, 2)[1]
# ========================================================== #

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


# ===================CREATE DATASET================================= #
dataset_test = ShapeNet( SVR=True, normal = False, class_choice = None, train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                          shuffle=False, num_workers=int(opt.workers))

print('testing set', len(dataset_test.datapath))
len_dataset = len(dataset_test)
# ========================================================== #

# ===================CREATE network================================= #
network = SVR_AtlasNet(num_points = opt.num_points, nb_primitives = opt.nb_primitives)
network.cuda()

network.apply(weights_init)
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print("previous weight loaded")

print(network)
network.eval()
# ========================================================== #

# =============DEFINE stuff for logs ======================================== #
train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
# ========================================================== #

# =============DEFINE ATLAS GRID ======================================== #
grain = int(np.sqrt(opt.gen_points/opt.nb_primitives))-1
grain = grain*1.0
print(grain)

#reset meters
val_loss.reset()
for item in dataset_test.cat:
    dataset_test.perCatValueMeter[item].reset()

#generate regular grid
faces = []
vertices = []
face_colors = []
vertex_colors = []
colors = get_colors(opt.nb_primitives)

for i in range(0,int(grain + 1 )):
        for j in range(0,int(grain + 1 )):
            vertices.append([i/grain,j/grain])

for prim in range(0,opt.nb_primitives):
    for i in range(0,int(grain + 1)):
        for j in range(0,int(grain + 1)):
            vertex_colors.append(colors[prim])

    for i in range(1,int(grain + 1)):
        for j in range(0,(int(grain + 1)-1)):
            faces.append([(grain+1)*(grain+1)*prim + j+(grain+1)*i, (grain+1)*(grain+1)*prim + j+(grain+1)*i + 1, (grain+1)*(grain+1)*prim + j+(grain+1)*(i-1)])
    for i in range(0,(int((grain+1))-1)):
        for j in range(1,int((grain+1))):
            faces.append([(grain+1)*(grain+1)*prim + j+(grain+1)*i, (grain+1)*(grain+1)*prim + j+(grain+1)*i - 1, (grain+1)*(grain+1)*prim + j+(grain+1)*(i+1)])
grid = [vertices for i in range(0,opt.nb_primitives)]
grid_pytorch = torch.Tensor(int(opt.nb_primitives*(grain+1)*(grain+1)),2)
for i in range(opt.nb_primitives):
    for j in range(int((grain+1)*(grain+1))):
        grid_pytorch[int(j + (grain+1)*(grain+1)*i),0] = vertices[j][0]
        grid_pytorch[int(j + (grain+1)*(grain+1)*i),1] = vertices[j][1]
print(grid_pytorch)
print("grain", grain, 'number vertices', len(vertices)*opt.nb_primitives)

results = dataset_test.cat.copy()
for i in results:
    results[i] = 0
# ========================================================== #


# =============TESTING LOOP======================================== #
#Iterate on the data
with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        img, points, cat , objpath, fn = data
        img = img.cuda()
        cat = cat[0]
        fn = fn[0]
        results[cat] = results[cat] + 1
        points = points.cuda()
        
        pointsReconstructed  = network.forward_inference(img, grid)
        dist1, dist2 = distChamfer(points, pointsReconstructed)
        loss_net = ((torch.mean(dist1) + torch.mean(dist2)))
        val_loss.update(loss_net.item())
        dataset_test.perCatValueMeter[cat].update(loss_net.item())

        if results[cat] > 20:
            #only save files for 20 objects per category
            continue
        print(results)

        if not os.path.exists(opt.model[:-4]):
            os.mkdir(opt.model[:-4])
            print('created dir', opt.model[:-4])

        if not os.path.exists(opt.model[:-4] + "/" + str(dataset_test.cat[cat])):
            os.mkdir(opt.model[:-4] + "/" + str(dataset_test.cat[cat]))
            print('created dir', opt.model[:-4] + "/" + str(dataset_test.cat[cat]))

        write_ply(filename=opt.model[:-4] + "/" + str(dataset_test.cat[cat]) + "/" + fn+"_GT", points=pd.DataFrame(points.cpu().data.squeeze().numpy()), as_text=True)
        b = np.zeros((len(faces),4)) + 3
        b[:,1:] = np.array(faces)
        write_ply(filename=opt.model[:-4] + "/" + str(dataset_test.cat[cat]) + "/" + fn+"_gen" + "_grain_" + str(int(opt.gen_points)), points=pd.DataFrame(torch.cat((pointsReconstructed.cpu().data.squeeze(), grid_pytorch), 1).numpy()), as_text=True, text=True, faces = pd.DataFrame(b.astype(int)))


    log_table = {
      "val_loss" : val_loss.avg,
      "gen_points" : opt.gen_points,
    }
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg)
        log_table.update({item: dataset_test.perCatValueMeter[item].avg})
    print(log_table)

    with open('stats.txt', 'a') as f: #open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
