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
from auxiliary.my_utils import *
import auxiliary.pointcloud_processor as pointcloud_processor
import random
import json
from collections import defaultdict

POINT_CLOUD_VERTICES_SIZE = 5625

class ShapeNet(data.Dataset):
    def __init__(self, train=True):
        self.train = train
        self.normalization = 'BoundingBox' #! normalization: ['UnitBall', 'BoundingBox', 'Identity']
        self.init_normalization()
        self.sample = False
        
        print('Create Shapenet Dataset...')
        # Define core path array
        self.data_metadata = []
        self.category_datapath = defaultdict(list)

        # Load classes
        self.pointcloud_path = os.path.join(atlasnet_dir, "partial_pc_data")

        self.classes = ['bowl', 'box', 'can', 'cup', 'tablespoon', 'teaspoon', 'fork','pitcher', 'obstacle', 'plate', 'teapot' ,'support_table','knife']#['knife','tablespoon', 'teaspoon', 'fork']#['bowl', 'box', 'can', 'cup', 'tablespoon', 'teaspoon', 'fork','pitcher', 'obstacle', 'plate', 'teapot' ,'support_table','knife']
        self.cat = {i: i for i in  self.classes}

        self.perCatValueMeter = {}
        for item in self.cat:
            self.perCatValueMeter[item] = AverageValueMeter()
        #按type按比例分配
        with open(os.path.join(atlasnet_dir, 'partial_pc_data_splits.json'), 'r') as fp:
            all_files = json.load(fp)
        if train:
            dataset_path = all_files['train']
        else:
            dataset_path = all_files['test']
            
        #! 读取partial_pc_data中每一个pkl缓存文件, 获取类名+点云的点
        if os.path.exists(self.pointcloud_path):#! Train+Validation相同
            for filename in dataset_path:
                cache_file_path = os.path.join(self.pointcloud_path, filename)
                with open(cache_file_path, 'rb') as f: #! 每个view都是单独的文件
                    cache = pickle.load(f)
                for objectname in cache.keys():
                    category = 'support_table' if 'support_table' in objectname else objectname.split('_')[0] #cup_1 -> cup
                    #! [view1/2的物体的partial点云, 所属场景名, 物体名如cup_1, 类名]
                    self.category_datapath[category].append(filename.split('.')[0]+objectname)#7b4acb843fd4b0b335836c728d324152_0001_4_scene-X-bowl-X_type-X-in-X_goal_view-2_partial_pc_cup_1
                    self.data_metadata.append((filename, objectname, category))
        else:
            raise ValueError(f"{self.pointcloud_path} does not exists!")

        #! 输出每个类有多少文件
        for category in self.classes:
            print(
                '    category '
                + category
                + ' Number Files :'
                + str(len(self.category_datapath[category]))
            )

    def init_normalization(self): #!用于根据给定的参数初始化归一化函数。
        print("Dataset normalization : " + self.normalization)

        if self.normalization == "UnitBall":
            self.normalization_function = pointcloud_processor.Normalization.normalize_unitL2ball_functional
        elif self.normalization == "BoundingBox":
            self.normalization_function = pointcloud_processor.Normalization.normalize_bounding_box_functional
        else:
            self.normalization_function = pointcloud_processor.Normalization.identity_functional

    def __len__(self): #! 返回数据集的长度。
        return len(self.data_metadata)

    def getpointcloud(self, index):#! 根据给定索引获取单个数据项。它从原始数据中读取点云数据并进行归一化。
        filename, objectname, category = self.data_metadata[index]
        cache_file_path = os.path.join(self.pointcloud_path, filename)
        with open(cache_file_path, 'rb') as f:
            cache = pickle.load(f)
        
        points = cache[objectname]
        if points.shape[0] < POINT_CLOUD_VERTICES_SIZE:#! 若点不够, 则use repetitions to fill some more points
            choice = np.arange(len(points))
            choice2 = np.random.choice(len(points), POINT_CLOUD_VERTICES_SIZE - choice.shape[0], replace=True)
            choice = np.concatenate([choice, choice2], 0)
            random.shuffle(choice)
            points = points[choice, :]
        elif points.shape[0] > POINT_CLOUD_VERTICES_SIZE:#! 若点多, 则FPS采样
            points = farthest_point_sampling(points, POINT_CLOUD_VERTICES_SIZE)
        assert points.shape[0] == POINT_CLOUD_VERTICES_SIZE#! 保证最后点一样多

        #! 归一化 减中心
        points = torch.from_numpy(points).float()[:, :3]
        points[:, :3] = self.normalization_function(points[:, :3])

        #! 因此，返回的 points 是一个形状为 [1, NUM_POINTS, 3] 的张量，其中第一个维度表示批次大小，第二个维度表示点的数量，第三个维度表示每个点的坐标。
        return points, filename, objectname, category
    
    def __getitem__(self, index):#! 根据给定索引返回一个包含点云和图像数据的字典。它从预处理过的数据中提取数据，并根据需要进行采样和图像处理。
        points, filename, objectname, category = self.getpointcloud(index)
        return points, category, filename, objectname 


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
