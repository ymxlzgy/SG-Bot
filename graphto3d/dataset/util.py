import numpy as np

import json

def get_label_name_to_global_id_new(scene_path):
    labelName2InstanceId = {}  # {'teapot_2': 8}
    instanceId2class = {}  # { 1: 'fork'}

    json_path = scene_path
    with open(json_path, 'r') as f:
        dict_out = json.load(f)
        for obj in dict_out['objects']:
            labelName2InstanceId[obj['name']] = obj['global_id']
            instanceId2class[obj['global_id']] = obj['class']
    if not labelName2InstanceId or not instanceId2class:
        raise ValueError(f"labelName2classId or classId2ClassName of {scene_path} shouldn't be empty!")
    return labelName2InstanceId, instanceId2class

def get_label_name_to_global_id(file):
    labelName2InstanceId = {} #{'teapot_2': 8}
    instanceId2class = {} #{ 1: 'fork'}

    json_path = file.split('.')[0]+'_view-2.json'
    with open(json_path, 'r') as f:
        dict_out = json.load(f)
        for obj in dict_out['objects']:
            labelName2InstanceId[obj['name']] = obj['global_id']
            instanceId2class[obj['global_id']] = obj['class']
    if not labelName2InstanceId or not instanceId2class:
        raise ValueError(f"labelName2classId or classId2ClassName of {file} shouldn't be empty!")
    return labelName2InstanceId, instanceId2class

import random
def farthest_point_sampling(vertices: np.ndarray, num_samples: int) -> np.ndarray:
    vertices = np.array(vertices)
    num_vertices = vertices.shape[0]
    if num_samples >= num_vertices:#! 点少则复制点
        choice = np.arange(num_vertices)
        choice2 = np.random.choice(num_vertices, num_samples - choice.shape[0], replace=True)
        choice = np.concatenate([choice, choice2], 0)
        random.shuffle(choice)
        vertices = vertices[choice, :]
        return vertices
    centroids = np.zeros((num_samples, 3))
    centroids[0] = vertices[np.random.randint(num_vertices)]#!随机选择输入顶点集中的一个顶点作为第一个提取的顶点
    distance = np.linalg.norm(vertices - centroids[0], axis=1)#! 计算输入顶点集中的所有顶点与第一个提取的顶点之间的距离，并将结果存储在distance数组中
    for i in range(1, num_samples):#! 开始一个循环，从第二个提取的顶点开始，直到提取num_samples个顶点。
        centroids[i] = vertices[np.argmax(distance)] #! 找到距离数组distance中具有最大值的索引，并将对应的输入顶点作为下一个提取的顶点存储在centroids中。
        distance = np.minimum(distance, np.linalg.norm(vertices - centroids[i], axis=1))#! 更新distance数组，将每个顶点与当前提取的顶点之间的距离与原始距离相比较，保留较小的距离值。这一步用于确保在选择下一个提取的顶点时，我们找到的是距离当前已提取顶点集的最远顶点。
    #print(len(centroids))
    return centroids

def get_data_from_json_file(json_path):
     #1. 读取json, 获得内参, 
    with open(json_path) as json_file:
        dict_string = json.load(json_file)
    #print(dict_string)
    camera_data = dict_string['camera_data']
    fx = camera_data['intrinsics']["fx"]
    fy = camera_data['intrinsics']["fy"]
    cx = camera_data['intrinsics']["cx"]
    cy = camera_data['intrinsics']["cy"]
    classid2name = {}
    for obj in dict_string['objects']:
        classid2name[obj['class_id']] = obj['name']    
    return fx,fy,cx,cy,classid2name

def get_points_instances_from_mesh(file, labelName2InstanceId, num_samples=5625):
    """ 一行一行读obj, 若是v, 则是vertex, 若是o, 则是object的name, 每存储一个v, 则存储一个instance_id
    需要name_to_id的dict
    Input: file: e.g. .../7b4acb843fd4b0b335836c728d324152_0001_5_scene-X-bowl-X_mid.obj 
           labelName2InstanceId { "cup_1":50 ....}
    Output: vertices(num_of_vertices,3); instance(num_of_vertices, ) """    
    instances_points = {}

    with open(file, 'r') as f:
        for line in f:
            if line.startswith('o'):
                label_name = line.strip()[2:]
                instance_id = labelName2InstanceId[label_name]
                instances_points[instance_id] = []
            elif line.startswith('v'):
                vertex = list(map(float, line.split()[1:]))
                if len(vertex) == 3:
                    instances_points[instance_id].append(vertex)
    #利用fps采取点
    instances = np.array([])
    points = np.empty((0,3))
    for id in instances_points:#key: instance_id
        instances_points[id] = farthest_point_sampling(instances_points[id], num_samples=num_samples)
        instances = np.concatenate(( instances, np.full(len(instances_points[id]),id) ))
        points = np.concatenate(( points,  instances_points[id] ))
    return points, instances
import os
import pickle
import sys
dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', '..'))#/home/caixiaoni/Desktop/master_thesis/AtlasNet
root_dir = os.path.abspath(os.path.join(dir, '..'))

def visualize_for_test(point):
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point)
    vis.add_geometry(pcd) 
    
    vis.poll_events()
    vis.run()
    vis.destroy_window()

def get_partial_point_cloud_of_initial_from_goal(file, labelName2InstanceId, atlas2folder, num_samples=5625):
    #! 通过obj_path 提取初始的 #!  ..../7b4acb843fd4b0b335836c728d324152_0007_5_scene-X-bowl-X_type-X-in-X_goal.obj
    filepath, filename = os.path.split(file)
    if 'goal' in file: #! 
        init_filename = filename[:-9] #! -> 7b4acb843fd4b0b335836c728d324152_0007_5_scene-X-bowl-X_type-X-in-X
    else: # 'mid' in file #! .../7b4acb843fd4b0b335836c728d324152_0007_5_scene-X-bowl-X_mid.obj
        filename = filename[:-8] #! 7b4acb843fd4b0b335836c728d324152_0007_5_scene-X-bowl-X
        all_files = os.listdir(filepath)
        for f in all_files:
            if f.endswith('_goal.obj') and (filename in f): #! 找到f是7b4acb843fd4b0b335836c728d324152_0007_5_scene-X-bowl-X_type-X-in-X_goal.obj
                init_filename = f[:-9]
    #file: 7b4acb843fd4b0b335836c728d324152_0007_5_scene-X-bowl-X_type-X-in-X
    
    #! 根据名称找到之前已经存储的cache，7b4acb843fd4b0b335836c728d324152_0007_5_scene-X-bowl-X_type-X-in-X_view-2_partial_pc.pkl
    partial_pc_cache_file = os.path.join(atlas2folder, 'partial_pc_data', init_filename+'_view-2_partial_pc.pkl')

    with open(partial_pc_cache_file, 'rb') as f:
        cache_data = pickle.load(f) #cache_data[class_id] = points 还需要一个class_id和instance_id的dict
    
    #利用fps采取点
    instances = np.array([])
    points = np.empty((0,3))
    filter_keywords = ['obstacle', 'spoon', 'fork', 'knife'] 
    for label_name in cache_data.keys():
        if 'mid' in file and any(key in label_name for key in filter_keywords):
            continue
        elif 'goal' in file and 'obstacle' in label_name:
            continue
        instance_id = labelName2InstanceId[label_name]
        #sampled_points = cache_data[label_name]
        sampled_points = farthest_point_sampling(cache_data[label_name], num_samples=num_samples)
        #! 进行归一化
        from . import pointcloud_processor
        normalization_function = pointcloud_processor.Normalization.normalize_bounding_box_functional
        sampled_points = normalization_function(sampled_points)

        assert(len(sampled_points)==num_samples)
        assert(sampled_points.shape[1]==3)
        instances = np.concatenate(( instances, np.full(len(sampled_points), instance_id) ))
        points = np.concatenate(( points,  sampled_points ))
        #visualize_for_test(sampled_points)
    unique_instances = np.unique(instances)
    return points, instances
