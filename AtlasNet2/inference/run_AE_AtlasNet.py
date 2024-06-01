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

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--model', type=str, default = '/media/storage/guangyao/caixiaoni/master_thesis_backup/AtlasNet2/log/AE_AtlasNet2_20230408T2110/atlasnet2.pth',  help='yuor path to the trained model')
parser.add_argument('--num_points', type=int, default = 5625,  help='number of points fed to poitnet')
parser.add_argument('--gen_points', type=int, default = 5625,  help='number of points to generate, put 30000 for high quality mesh, 2500 for quantitative comparison with the baseline')
parser.add_argument('--nb_primitives', type=int, default = 25,  help='number of primitives')
parser.add_argument('--bottleneck_size', type=int, default = 128, help='bottleneck_size of model')

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
    x,y = a,b   #! torch.Size([1, 5625, 3]), torch.Size([1, 5625, 3])
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

#blue = lambda x:'\033[94m' + x + '\033[0m'
opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# ===================CREATE DATASET================================= #
dataset_test = ShapeNet(train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                          shuffle=False, num_workers=int(opt.workers))

print('testing set', len(dataset_test.data_metadata))
len_dataset = len(dataset_test)
# ========================================================== #

# ===================CREATE network================================= #
network = AE_AtlasNet(num_points = opt.num_points, bottleneck_size = opt.bottleneck_size, nb_primitives = opt.nb_primitives)
network.cuda()
network.apply(weights_init)
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print("previous weight loaded")
network.eval()
# ========================================================== #

# =============DEFINE stuff for logs ======================================== #
train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
# ========================================================== #

# =============DEFINE ATLAS GRID ======================================== #
grain = int(np.sqrt(opt.gen_points/opt.nb_primitives))-1
grain = grain*1.0

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
""" 这段代码主要用于生成一个由多个基元（如平面、圆柱等）组成的网格。以下是代码的详细解释：

初始化空列表，用于存储顶点、面、顶点颜色和面颜色。
生成颜色列表，其中 opt.nb_primitives 表示基元的数量。
循环遍历所有的基元，生成顶点和颜色信息：
对于每个基元，遍历所有的 i 和 j，这里的 i 和 j 可以看作是二维网格的坐标。将这些坐标归一化，存储在 vertices 列表中。
同时，将每个基元的颜色信息添加到 vertex_colors 列表中。
根据当前的 i 和 j 值，生成三角形面片。这些面片将用于构造每个基元的网格。将生成的面片信息存储在 faces 列表中。
构建一个包含所有基元顶点信息的 grid 列表。这将生成一个形状为 (opt.nb_primitives * (grain+1) * (grain+1), 2) 的 grid_pytorch 张量。
打印生成的 grid_pytorch 张量以及其他一些关于网格的信息。
在这段代码中，grain 控制网格的密度。较大的 grain 值将导致更密集的网格，从而产生更多的顶点和面片。 """

""" 为了使len(vertices) * opt.nb_primitives等于5100，我们需要首先计算每个基元需要的顶点数。因为有25个基元，所以每个基元需要 5100 / 25 = 204 个顶点。
现在，我们需要找到一个合适的 grain 值，使得每个基元有204个顶点。请注意，这里的顶点是在一个规则的二维网格上生成的。所以我们要找到一个正方形网格，其顶点总数接近204。
我们知道，对于一个边长为 grain 的正方形网格，其顶点总数为 (grain + 1) * (grain + 1)。要找到一个合适的 grain 值，我们可以尝试寻找一个最接近 204 的平方根的整数值。204 的平方根约等于 14.28，所以我们可以选择 grain = 14。
当 grain = 14 时，每个基元的顶点数为 (14 + 1) * (14 + 1) = 15 * 15 = 225。这样，len(vertices) * opt.nb_primitives = 225 * 25 = 5625。 """
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
print("grain", grain, 'number vertices', len(vertices)*opt.nb_primitives)

results = dataset_test.cat.copy()
for i in results:
    results[i] = 0
# ========================================================== #

# =============TESTING LOOP======================================== #
#Iterate on the data
with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        points, cat, filename, fn = data
        cat = cat[0] #fork
        fn = fn[0].split('.')[0] #fork_1
        results[cat] = results[cat] + 1
        points = points.transpose(2,1).contiguous()
        points = points.cuda()  #! GT
        pointsReconstructed  = network.forward_inference(points, grid)
        dist1, dist2,_,_ = distChamfer(points.transpose(2,1).contiguous(), pointsReconstructed)
        loss_net = ((torch.mean(dist1) + torch.mean(dist2)))
        val_loss.update(loss_net.item())
        dataset_test.perCatValueMeter[cat].update(loss_net.item())

        if not os.path.exists(opt.model[:-4]): #network
            os.mkdir(opt.model[:-4])
            print('created dir', opt.model[:-4])

        if not os.path.exists(opt.model[:-4] + "/" + str(dataset_test.cat[cat])):
            os.mkdir(opt.model[:-4] + "/" + str(dataset_test.cat[cat]))
            print('created dir', opt.model[:-4] + "/" + str(dataset_test.cat[cat]))

        write_ply(filename=opt.model[:-4] + "/" + str(dataset_test.cat[cat]) + "/" + fn+"_GT", points=pd.DataFrame(points.transpose(2,1).contiguous().cpu().data.squeeze().numpy()), as_text=True)
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
