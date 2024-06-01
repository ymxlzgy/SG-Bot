import os
import sys
atlasnet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', '..'))#/home/caixiaoni/Desktop/master_thesis/AtlasNet
if atlasnet_dir not in sys.path:
    sys.path.append(atlasnet_dir)
import torch.utils.data as data
import os.path
import torch
import numpy as np
import os
import pickle
from os.path import join, dirname, exists
import trimesh
from auxiliary.my_utils import *
import auxiliary.pointcloud_processor as pointcloud_processor
from copy import deepcopy
import random

POINT_CLOUD_VERTICES_SIZE = 5625

class ShapeNet(data.Dataset):
    def __init__(self, train=True):
        self.train = train
        self.normalization = 'BoundingBox' #! normalization: ['UnitBall', 'BoundingBox', 'Identity']
        self.init_normalization()
        self.sample = False
        

        print('Create Shapenet Dataset...')
        # Define core path array
        self.datapath = []
        self.category_datapath = {}

        # Load classes
        self.pointcloud_path = "/media/storage/guangyao/caixiaoni/models"

        self.classes = ['bowl', 'box', 'can', 'cup', 'tablespoon', 'teaspoon', 'fork','pitcher', 'obstacle', 'plate', 'teapot' ,'support_table','knife']
        self.cat = {i: i for i in  self.classes}

        # Create Cache path
        self.path_dataset = join(atlasnet_dir, 'data', 'cache')
        if not exists(self.path_dataset):#! 不存在,则创建缓存
            os.mkdir(self.path_dataset)
        
        self.path_dataset = join(self.path_dataset, self.normalization + str(train))

        self.perCatValueMeter = {}
        for item in self.cat:
            self.perCatValueMeter[item] = AverageValueMeter()

        # Compile list of pointcloud path by selected category
        for category in self.classes:#! category: class_name
            dir_pointcloud = join(self.pointcloud_path, category) #!/media/caixiaoni/xiaonicai-u/models/bowl

            list_pointcloud = []
            for file in os.listdir(dir_pointcloud): #将obj文件名存储到list_pointcloud   
                if file.endswith('obj'):
                    list_pointcloud.append(file)
            list_pointcloud = sorted(list_pointcloud)

            if self.train:
                list_pointcloud = list_pointcloud[:int(len(list_pointcloud) * 1)]#list_pointcloud[:int(len(list_pointcloud) * 0.9)]
            else:
                list_pointcloud = list_pointcloud[:int(len(list_pointcloud) * 1)]#list_pointcloud[int(len(list_pointcloud) * 0.9:]

            print(
                '    category '
                + category
                + ' Number Files :'
                + str(len(list_pointcloud))
            )

            if len(list_pointcloud) != 0:
                self.category_datapath[category] = []
                for pointcloud in list_pointcloud:
                    pointcloud_path = join(dir_pointcloud, pointcloud)#!/media/caixiaoni/xiaonicai-u/models/bowl/bowl_1.obj

                    self.category_datapath[category].append((pointcloud_path, None, pointcloud, category))
                    
        for item in self.classes:
            for pointcloud in self.category_datapath[item]:
                self.datapath.append(pointcloud)

        # Preprocess and cache files
        self.preprocess()


    def preprocess(self):#! 预处理数据集并将结果缓存到磁盘。如果缓存已经存在，它会直接从缓存中加载数据。
        if exists(self.path_dataset + "info.pkl"):
            # Reload dataset
            print(f"Reload dataset : {self.path_dataset}")
            with open(self.path_dataset + "info.pkl", "rb") as fp:
                self.data_metadata = pickle.load(fp)

            self.data_points = torch.load(self.path_dataset + "points.pth")
        else:#! 不存在二进制
            # Preprocess dataset and put in cache for future fast reload
            print("preprocess dataset...")
            self.datas = [self._getitem(i) for i in range(self.__len__())]

            # Concatenate all proccessed files
            self.data_points = [a[0] for a in self.datas]
            self.data_points = torch.cat(self.data_points, 0)#! 
            
            self.data_metadata = [{'pointcloud_path': a[1], 'image_path': a[2], 'name': a[3], 'category': a[4]} for a in
                                    self.datas]

            # Save in cache
            with open(self.path_dataset + "info.pkl", "wb") as fp:  # Pickling
                pickle.dump(self.data_metadata, fp)
            torch.save(self.data_points, self.path_dataset + "points.pth",_use_new_zipfile_serialization=False)

        print("Dataset Size: " + str(len(self.data_metadata)))

    def init_normalization(self): #!用于根据给定的参数初始化归一化函数。
        print("Dataset normalization : " + self.normalization)

        if self.normalization == "UnitBall":
            self.normalization_function = pointcloud_processor.Normalization.normalize_unitL2ball_functional
        elif self.normalization == "BoundingBox":
            self.normalization_function = pointcloud_processor.Normalization.normalize_bounding_box_functional
        else:
            self.normalization_function = pointcloud_processor.Normalization.identity_functional


    def __len__(self): #! 返回数据集的长度。
        return len(self.datapath)

    def _getitem(self, index):#! 根据给定索引获取单个数据项。它从原始数据中读取点云数据并进行归一化。
        pointcloud_path, image_path, pointcloud, category = self.datapath[index]
        
        mesh = trimesh.load(pointcloud_path)
        points = mesh.vertices
        print(f"previous pc vertices num: {points.shape}")

        if points.shape[0] < POINT_CLOUD_VERTICES_SIZE:#! 若点不够, 则use repetitions to fill some more points
            choice = np.arange(len(points))
            choice2 = np.random.choice(len(points), POINT_CLOUD_VERTICES_SIZE - choice.shape[0], replace=True)
            choice = np.concatenate([choice, choice2], 0)
            random.shuffle(choice)
            points = points[choice, :]
        elif points.shape[0] > POINT_CLOUD_VERTICES_SIZE:#! 若点多, 则FPS采样
            points = farthest_point_sampling(points, POINT_CLOUD_VERTICES_SIZE)
        assert points.shape[0] == POINT_CLOUD_VERTICES_SIZE#! 保证最后点一样多

        points = torch.from_numpy(points).float()
        points[:, :3] = self.normalization_function(points[:, :3])
        #print(f"After normalized points shape: {points.shape}, {(points.unsqueeze(0)).shape}")
        return points.unsqueeze(0), pointcloud_path, image_path, pointcloud, category
    
    def __getitem__(self, index):#! 根据给定索引返回一个包含点云和图像数据的字典。它从预处理过的数据中提取数据，并根据需要进行采样和图像处理。
        return_dict = deepcopy(self.data_metadata[index])
        # Point processing
        points = self.data_points[index]
        points = points.clone()
        if self.sample:#! 若sample, 则随机取点
            choice = np.random.choice(points.size(0), self.num_sample, replace=True)
            points = points[choice, :]
        return_dict['points'] = points[:, :3].contiguous()

        return return_dict['points'], return_dict['category'], return_dict['pointcloud_path'], return_dict['name'] 

def farthest_point_sampling(vertices, num_samples):#! FPS采样
    vertices = np.array(vertices)
    num_vertices = vertices.shape[0]
    if num_samples >= num_vertices:
        raise ValueError(f"{num_samples} shouldn't be more than {num_vertices}")
        return vertices
    centroids = np.zeros((num_samples, 3))
    centroids[0] = vertices[np.random.randint(num_vertices)]#!随机选择输入顶点集中的一个顶点作为第一个提取的顶点
    distance = np.linalg.norm(vertices - centroids[0], axis=1)#! 计算输入顶点集中的所有顶点与第一个提取的顶点之间的距离，并将结果存储在distance数组中
    for i in range(1, num_samples):#! 开始一个循环，从第二个提取的顶点开始，直到提取num_samples个顶点。
        centroids[i] = vertices[np.argmax(distance)] #! 找到距离数组distance中具有最大值的索引，并将对应的输入顶点作为下一个提取的顶点存储在centroids中。
        distance = np.minimum(distance, np.linalg.norm(vertices - centroids[i], axis=1))#! 更新distance数组，将每个顶点与当前提取的顶点之间的距离与原始距离相比较，保留较小的距离值。这一步用于确保在选择下一个提取的顶点时，我们找到的是距离当前已提取顶点集的最远顶点。
    return centroids

if __name__ == '__main__':
    print('Testing Shapenet dataset')
    opt = {"normalization": "BoundingBox", "class_choice": ["airplane"], "SVR": False, "sample": True, "number_points":2500,"npoints": 2500,
           "shapenet13": True, "models_path": '/media/caixiaoni/xiaonicai-u/models'}
    #It should output a python dictionary with the entry keys image, points, pointcloud_path, image_path, name, and category.
    d = ShapeNet(train=True)
    print(d[2])
    a = len(d)
    print(a)
