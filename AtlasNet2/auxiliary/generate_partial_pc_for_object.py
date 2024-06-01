import pickle
import os
import json
import numpy as np
import cv2

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
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

AtlasNet2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', '..'))
Particle_PC_DIR = os.path.join(AtlasNet2_DIR, 'partial_pc_data')

def depth2pcd(depth, intrinsics, pose=None):
    inv_K = np.linalg.inv(intrinsics)
    inv_K[2, 2] = -1
    depth = np.flipud(depth)
    y, x = np.where(depth != 0)
    # image coordinates -> camera coordinates
    points = np.dot(inv_K, np.stack([x, y, np.ones_like(x)] * depth[y, x], 0)) #[3, num_points] 
    if pose is None:
        return points.T
    else:
        # camera coordinates -> world coordinates
        points = np.dot(pose, np.concatenate([points, np.ones((1, points.shape[1]))], 0)).T[:, :3] # [num_points,3]
        return points
    
def get_data_from_json_file(json_path):
     #1. 读取json, 获得内参, 
    with open(json_path) as json_file:
        dict_string = json.load(json_file)

    camera_data = dict_string['camera_data']
    fx = camera_data['intrinsics']["fx"]
    fy = camera_data['intrinsics']["fy"]
    cx = camera_data['intrinsics']["cx"]
    cy = camera_data['intrinsics']["cy"]
    width, height = camera_data['width'], camera_data['height']
    classid2name = {}
    for obj in dict_string['objects']:
        classid2name[obj['class_id']] = obj['name']    

    return fx,fy,cx,cy,width,height,classid2name

def extract_pc_from_scene(root_path):
    for folder in os.listdir(root_path):#root
        if os.path.isdir(root_path+'/'+folder):
            folder_path = root_path+'/'+folder
            for filename in os.listdir(folder_path):
                if filename.endswith('.obj') and ('mid' not in filename) and (len(filename.split('_'))>4):
                    #!7b4acb843fd4b0b335836c728d324152_0001_4_scene-X-bowl-X_type-X-in-X_goal.obj
                    #!7b4acb843fd4b0b335836c728d324152_0001_4_scene-X-bowl-X_type-X-in-X.obj
                    
                    for index in range(1,3): #view-1/2 save seprate file
                        save_cache_path = os.path.join(Particle_PC_DIR, filename.split('.')[0]+'_view-'+str(index)+'_partial_pc.pkl')
                        if os.path.exists(save_cache_path):
                            continue
                        json_path = os.path.join(folder_path, filename.split('.')[0]+'_view-'+str(index)+'.json')
                        img_path = json_path.replace('.json','.png')
                        depth_path = os.path.join(folder_path, filename.split('.')[0]+'_view-'+str(index)+'_depth.pkl')
                        seg_path = os.path.join(folder_path, filename.split('.')[0]+'_view-'+str(index)+'_seg.exr')

                        #!1. 读取json
                        fx,fy,cx,cy,width,height,classid2name = get_data_from_json_file(json_path)
                        #!2. 读取mask
                        mask = cv2.imread(seg_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0]
                        exist_classes = np.unique(mask)

                        #!3. 读取深度图
                        with open(depth_path, 'rb') as f:
                            depth = pickle.load(f)
                        #!4. 读取RGB image
                        rgb_image = cv2.imread(img_path)
                        if rgb_image.shape[-1] == 4:
                            rgb_image = rgb_image[..., :3]

                        #!分割后点云
                        point_clouds = {}
                        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                        
                        for class_id in exist_classes:
                            if class_id==0:
                                continue
                            mask_obj = (mask==class_id)
                            #!将一个布尔掩码 mask_obj 应用于深度图像 depth，将属于物体的像素深度值保留，将非物体区域的深度值设置为零。
                            depth_obj = np.where(mask_obj, depth, 0)
                            rgb_obj = np.where(mask_obj[..., np.newaxis], rgb_image, 0)
                            
                            #!将一个布尔掩码 mask_obj 应用于 RGB 彩色图像 rgb_image，将属于物体的像素颜色值保留，将非物体区域的像素颜色值设置为零。
                            points = depth2pcd(depth_obj, intrinsics)
                            point_clouds[classid2name[class_id]] = points
                        #!存储为cache
                        with open(save_cache_path, 'wb') as f:
                            pickle.dump(point_clouds, f)                        
            print(f"Finished {folder}")

atlasnet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', '..'))#/home/caixiaoni/Desktop/master_thesis/AtlasNet
def generate_train_sample():
    records_file = {}
    import random
    import json   
    pointcloud_path = os.path.join(atlasnet_dir, "partial_pc_data")
    file_list = os.listdir(pointcloud_path)
    random.shuffle(file_list)  # 随机打乱文件列表顺序

    split_index = int(len(file_list) * 0.9)  # 计算分割索引
    train_files = file_list[:split_index]  # 前 90% 的文件为训练集
    test_files = file_list[split_index:]  # 后 10% 的文件为测试集     
    records_file['train'] = train_files
    records_file['test'] = test_files

    with open(os.path.join(atlasnet_dir, 'partial_pc_data_splits.json'), 'w+') as fp:
        json.dump(records_file, fp, indent=2)

if __name__ == '__main__':
    root_path = '/media/caixiaoni/xiaonicai-u/test_pipeline_dataset/raw'
    #extract_pc_from_scene(root_path)
    generate_train_sample()

    """ load_path = '/media/caixiaoni/xiaonicai-u/AtlasNetV2.2/AtlasNet2/partial_pc_data/7b4acb843fd4b0b335836c728d324152_0001_4_scene-X-bowl-X_type-X-in-X_goal_partial_pc.pkl'
    with open(load_path, 'rb') as f:
        points = pickle.load(f)
    key = list(points.keys())[1]
    visualize_for_test(points[key][1]) """
