# @markdown Collect demonstrations with a scripted expert
import copy
import sys
import shutil
import os.path
import random
from moviepy.editor import ImageSequenceClip
import numpy as np
import pybullet
import pybullet_utils.transformations as trans
import pickle
import json
import torch
import open3d as o3d
import seaborn as sns
import matplotlib.pyplot as plt
from initial import RearrangeEnv
from graphto3d.model.VAE import VAE as VAEv2
from graphto3d.helpers.visualize_scene import render, render_per_shape, render_comparison, render_ini_and_goal, render_result
from graphto3d.model.atlasnet import AE_AtlasNet
from graphto3d.helpers.util import bool_flag, batch_torch_denormalize_box_params, fit_shapes_to_box, fit_shapes_to_boxv2
from graphto3d.dataset.util import get_label_name_to_global_id_new
from graphto3d.dataset.dataset_use_features_gt import RIODatasetSceneGraph, collate_fn_vaegan, collate_fn_vaegan_points
import extension.dist_chamfer as ext
chamfer = ext.chamferDist()
# Generate new dataset.
# @markdown Collect demonstrations with a scripted expert, or download a pre-generated dataset.

mode = 'robot'
assert mode == 'oracle' or mode == 'robot'
all_scene = False
ee_to_tip_franka_common = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.118970], [0, 0, 0, 1]])
ee_to_tip_franka_cutlery = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.105070], [0, 0, 0, 1]])
dynamic_dict={'cutlery':{'mass':0.15,'restitution':0.7,'lateralFriction':0.75},
                  'bowl':{'mass':0.25,'restitution':0.7,'lateralFriction':0.7},
                  'box':{'mass':0.25,'restitution':0.5,'lateralFriction':0.7},
                  'can':{'mass':0.2,'restitution':0.5,'lateralFriction':0.7},
                  'cup':{'mass':0.25,'restitution':0.7,'lateralFriction':0.7},
                  'pitcher':{'mass':0.3,'restitution':0.7,'lateralFriction':0.7},
                  'plate':{'mass':0.3,'restitution':0.5,'lateralFriction':0.7},
                  'teapot':{'mass':0.3,'restitution':0.7,'lateralFriction':0.7},
                  'obstacle':{'mass':0.2,'restitution':0.5,'lateralFriction':0.7}}

def close_dis(points1,points2):
    dist = -2 * np.matmul(points1, points2.transpose())
    dist += np.sum(points1 ** 2, axis=-1)[:, None]
    dist += np.sum(points2 ** 2, axis=-1)[None, :]
    dist = np.sqrt(dist)
    return np.min(dist)

def calculate_icp(source_pc, target_pc, init_trans_guess=None, threshold=None):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_pc)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_pc)

    if init_trans_guess is not None:  # 若init_guess不为None
        trans_init = init_trans_guess
    else:
        trans_init = np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000))

    return reg_p2p.transformation, reg_p2p.inlier_rmse

def get_key(target_dict, target_value):
    return [key for key, value in target_dict.items() if value == target_value]

def execute_rearrange(env, grasp_cam, opening, score, cam_to_world, ee_to_tip_franka, score_weight, rel_pose=None, post=False):
    grasps_w = np.matmul(cam_to_world, grasp_cam)
    success = env.rearrange_step(grasps_w, opening, score, ee_to_tip_franka, weight=score_weight, rel_pose=rel_pose, post=post)
    # stop for a while and wait for everything to be settled
    for _ in range(400):
        env.step_sim_and_render()
    # Move robot to home configuration.
    for i in range(len(env.joint_ids)):
        pybullet.setJointMotorControl2(env.robot_id, env.joint_ids[i], pybullet.POSITION_CONTROL,
                                       targetPosition=env.home_joints[i])
    # release the gripper
    env.gripper.release()
    for _ in range(1000):
        env.step_sim_and_render()
    return success

def oracle_rearrange(env, obj_id, rel_pose):
    current_pos, current_ori = pybullet.getBasePositionAndOrientation(obj_id)
    tmp_rot_matrix = np.eye(4)
    if rel_pose is not None:
        current_rot_mat = pybullet.getMatrixFromQuaternion(current_ori)
        current_rot_mat = np.reshape(current_rot_mat, (3, 3))
        new_rot_mat = np.matmul(rel_pose[:3,:3], current_rot_mat)
        tmp_rot_matrix[:3,:3] = new_rot_mat
        new_ori = trans.quaternion_from_matrix(tmp_rot_matrix)
        new_pos = np.matmul(rel_pose[:3,:3], np.array(current_pos)) + rel_pose[:3,3] + np.array([0, 0, 0.05])
    else:
        new_ori, new_pos = current_ori, np.array([0.5,-0.4,0.05])
    pybullet.resetBasePositionAndOrientation(obj_id, new_pos, new_ori)
    for _ in range(1000):
        env.step_sim_and_render()
    return 1

def check_occupancy(target_id, target_pose, pose_dict):
    for i, key in enumerate(pose_dict):
        if key!=target_id:
            target_xy, pose_xy = target_pose[:2,3], pose_dict[key][:2,3]
            dis = np.linalg.norm(target_xy-pose_xy)
            if dis < 0.08:
                return True
    return False

# check minimum distance between two point clouds if there is occupancy or not
def check_occupancy_new(env, target_id, target_points, current_points_dict):
    target_points_xyz = copy.deepcopy(target_points)
    target_points_xyz[:,2:3] = 0
    target_points_ = torch.tensor(target_points_xyz, dtype=torch.float).cuda()
    for i, key in enumerate(current_points_dict):
        obj_name = get_key(env.pybullet_id_dict, key)[0]
        if key != target_id and obj_name.split('_')[0] not in ['support', 'plate']:
            current_pc_w = np.matmul(env.real_cam_to_world[:3, :3], current_points_dict[key].T) + env.real_cam_to_world[:3, 3].reshape(3, 1)
            current_pc_w[2:3,:] = 0
            current_pc_w_ = torch.tensor(current_pc_w, dtype=torch.float).cuda().transpose(0, 1).contiguous()
            dist1, dist2 = chamfer(current_pc_w_.unsqueeze(0), target_points_.unsqueeze(0))
            min_dist = torch.min(torch.cat((torch.sqrt(dist1),torch.sqrt(dist2)),1))
            # render_comparison(current_pc_w.T,target_points)
            if min_dist < 0.0001:
                description = f"there is occupied by {obj_name} in the target area, {min_dist}m far from the goal. I need to skip this move for now."
                env.description_list.append(description)
                print(description)
                return True
    return False

def eval_atlasnet(path2atlas):
    saved_atlasnet_model = torch.load(path2atlas)
    point_ae = AE_AtlasNet(num_points=5625, bottleneck_size=128, nb_primitives=25)
    point_ae.load_state_dict(saved_atlasnet_model, strict=True)
    if torch.cuda.is_available():
        point_ae = point_ae.cuda()
    point_ae = point_ae.eval()
    return point_ae

def eval_g2s(path2g2s, vocab, epoch=600):
    sys.path.append('./graphto3d/')
    model = VAEv2(type="shared", vocab=vocab, replace_latent=True, with_changes=True, residual=True)
    model.load_networks(exp=path2g2s, epoch=str(epoch))
    if torch.cuda.is_available():
        model = model.cuda()
    model = model.eval()
    return model

def load_goal_data(goal_scene_path, obj_classes, path2atlas):
    _, atlasname = os.path.split(path2atlas)
    atlasname = atlasname.split('.')[0]
    data = {}
    labelName2InstanceId, instanceId2class = get_label_name_to_global_id_new(goal_scene_path)
    keys = list(instanceId2class.keys())  # ! all instance_ids
    view = goal_scene_path[-6]
    scene_graph_path = goal_scene_path.replace(f'_goal_view-{view}', '_goal_scene_graph')
    with open(scene_graph_path) as f:
        scene_graph_info = json.load(f)
    feats_path = os.path.join(goal_scene_path.rsplit('/',1)[0], f'{atlasname}_features_'+goal_scene_path.rsplit('/',1)[1].replace(f'_view-{view}.json','.pkl'))
    feats_dic = pickle.load(open(feats_path, 'rb'))
    shape_priors_in = feats_dic['shape_priors']
    order = np.asarray(feats_dic['instance_order'])
    counter = 0
    instance2mask = {}
    instance2mask[0] = 0
    ordered_shape_priors = []
    class_list = []
    triples = []
    for key in keys:
        feats_in_instance = key == order
        ordered_shape_priors.append(shape_priors_in[:-1][feats_in_instance])
        class_list.append(obj_classes[instanceId2class[key]])
        instance2mask[key] = counter + 1
        counter += 1
    for r in scene_graph_info['relationships']:  # create relationship triplets from data
        if r[0] in instance2mask.keys() and r[1] in instance2mask.keys():  # r[0], r[1] -> instance_id
            subject = instance2mask[r[0]] - 1
            object = instance2mask[r[1]] - 1
            predicate = r[2]
            if subject >= 0 and object >= 0:
                triples.append([subject, predicate, object])
    scene_idx = len(class_list)
    for i, ob in enumerate(class_list):
        triples.append([i, 0, scene_idx])
    class_list.append(0)
    ordered_shape_priors.append(np.zeros([1, shape_priors_in.shape[1]]))
    shape_priors_in = list(np.concatenate(ordered_shape_priors, axis=0))
    data['object_names'] =  list(labelName2InstanceId.keys())
    data['objs'] = class_list
    data['triples'] = triples
    data['shape_priors'] = shape_priors_in
    return data

# TODO generate shape priors
def generate_goal_data(goal_scene_path):
    scene_graph_path = goal_scene_path.replace('_goal_view-1', '_goal_scene_graph')
    with open(scene_graph_path) as f:
        scene_graph_info = json.load(f)
    labelName2InstanceId, instanceId2class = get_label_name_to_global_id_new(goal_scene_path)
    keys = list(instanceId2class.keys())  # ! all instance_ids
    dec_objs, dec_triples, dec_shape_priors = data['objs'], data['triples'], data['shape_priors']

def scene_imagination(data, g2s_model, point_ae, fit_box=False, visual=False):
    dec_objs, dec_triples, dec_shape_priors = torch.tensor(data['objs'], dtype=torch.int64), torch.tensor(data['triples'], dtype=torch.int64), torch.tensor(data['shape_priors'], dtype=torch.float32)
    dec_objs, dec_triples, dec_shape_priors = dec_objs.cuda(), dec_triples.cuda(), dec_shape_priors.cuda()
    boxes_pred, (points_pred, shape_code_pred) = g2s_model.sample_box_and_shape(point_ae, dec_objs, dec_triples,
                                                                               dec_shape_priors)
    boxes_pred_den = batch_torch_denormalize_box_params(boxes_pred) # TODO if change coordinates, this should be changed
    if visual:
        color_palette = np.array(sns.color_palette('hls', len(boxes_pred)))
        # for i in range(len(points_pred)):
        #     render_per_shape(points_pred[i].detach().cpu(), colors=color_palette[i])
        render(boxes_pred_den, shapes_pred=points_pred.detach().cpu(), colors=color_palette, render_boxes=True)
    denorm_shape_list = []
    if fit_box:
        color_list = []
        for i in range(len(boxes_pred_den)-1):
            denorm_shape = fit_shapes_to_box(boxes_pred_den[i], points_pred[i], withangle=False)
            denorm_shape_list.append(copy.deepcopy(denorm_shape))
        boxes_pred_den = boxes_pred_den.detach().cpu().numpy().tolist()
        return boxes_pred_den[:-1], denorm_shape_list # remove _scene_ node

    boxes_pred_den = boxes_pred_den.detach().cpu().numpy().tolist()
    shapes_pred = shape_code_pred.detach().cpu().numpy().tolist()
    points_pred = points_pred.detach().cpu().numpy().tolist()
    return boxes_pred_den[:-1], points_pred[:-1] # remove _scene_ node

def post_processing(bboxes, points, data, table_height):
    new_boxes_dict = {}
    new_points_dict = {}
    h_corr = None
    for box, obj_name in zip(bboxes, data['object_names']):
        if 'support' in obj_name:
            l, w, h, cx, cy, cz = box
            table_height_imagined = cz + h/2
            h_corr = table_height - table_height_imagined
            break
    #adjust the height
    assert h_corr is not None
    for box, point, obj_name in zip(bboxes, points, data['object_names']):
        box[5:6] += h_corr
        point[:,2:3] += h_corr
        new_boxes_dict[obj_name] = box
        new_points_dict[obj_name] = point
    return new_boxes_dict, new_points_dict


def adjust_matrix(source, target, R):
    # z axis of R
    z_dir = R[:3,2]

    # rotation axis and angle
    rotation_axis = np.cross(z_dir, np.array([0, 0, 1]))
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    angle = np.arccos(np.dot(z_dir, np.array([0, 0, 1])))

    # Rodrigues' formula
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                  [rotation_axis[2], 0, -rotation_axis[0]],
                  [-rotation_axis[1], rotation_axis[0], 0]])
    adjustment_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


    adjusted_R = adjustment_matrix @ R # no rotation on x and y
    tmp_pc = np.matmul(source, adjusted_R.T)
    center_tmp = np.mean(tmp_pc, axis=0)
    center_target = np.mean(target, axis=0)
    t = center_target - center_tmp
    t[2] = 0 # no movement on z
    return {'R':adjusted_R, 't':t}

def Rt_calculation(source, target, split=20, thread=0.02, cal_mode='lowest_rmse', flip_z=False):
    init_transformation_matrix = np.eye(4)
    rotation_x_list = [np.eye(3)]
    rmse_list = []
    Rt_list = []

    if flip_z:
        rotation_x_list.append(np.array([
            [1, 0, 0],
            [0, np.cos(np.pi), -np.sin(np.pi)],
            [0, np.sin(np.pi), np.cos(np.pi)]
        ]))

    for rotation_x in rotation_x_list:
        for i in range(0, split):
            angle = 2 * np.pi * i / split
            rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0],
                            [0, 0, 1]])
            rot = np.matmul(rot, rotation_x)
            c_source = np.mean(source, axis=0)
            pc_temp = source - c_source
            intermedia_pc = np.matmul(pc_temp, rot.T)
            translation_vector = np.mean(target,axis=0) - np.mean(intermedia_pc,axis=0)
            init_transformation_matrix[:3, 3] = translation_vector
            init_transformation_matrix[:3, :3] = np.eye(3)
            icp_transformation_matrix, rmse = calculate_icp(intermedia_pc, target,
                                                              init_trans_guess=init_transformation_matrix, threshold=thread)
            # print(rmse)
            rmse_list.append(rmse)

            # calculate R and t
            R_target_source = np.matmul(icp_transformation_matrix[:3,:3], rot)
            t_target_source = -np.matmul(R_target_source, c_source) + icp_transformation_matrix[:3,3]
            # transformed_points = np.matmul(source, R_target_source.T) + t_target_source
            # render_comparison(target, transformed_points)
            # print(i)
            Rt_list.append({'R':R_target_source, 't':t_target_source})

    if cal_mode == 'lowest_rmse':
        Rt_idx = np.argmin(np.array(rmse_list))
        return Rt_list[Rt_idx]

    # this is cutlry-only and has severe consequences if not proper.
    elif cal_mode == 'minimal_R':
        dot_list = []
        Rt_list_copy = copy.deepcopy(Rt_list)
        rmse_list_copy = copy.deepcopy(rmse_list)
        for Rt, rmse in zip(Rt_list_copy, rmse_list_copy):
            x_dot = np.matmul(np.array([1, 0, 0]),Rt['R'][:3, 0]) # x_axis
            y_dot = np.matmul(np.array([0, 1, 0]), Rt['R'][:3, 1])  # y_axis
            z_dot = np.matmul(np.array([0, 0, 1]), Rt['R'][:3, 2])  # z_axis
            # transformed_points = np.matmul(source, Rt['R'].T) + Rt['t']
            # render_comparison(target, transformed_points)
            if z_dot < 0:
                Rt_list = [item for item in Rt_list if not np.array_equal(item['R'], Rt['R'])]
                rmse_list.remove(rmse)
                continue
            dot_list.append(x_dot + y_dot + z_dot)
        # Rt_idx = np.argmax(np.array(dot_list))
        Rt_idx = np.argmin(np.array(rmse_list)) # TODO what if len(rmse_list) = 0?
        return Rt_list[Rt_idx]
    else:
        raise ValueError('Only support two modes.')

def match_pc(bboxes_, points_, pc_segments, name_dict, cam_to_world):
    Rt_dict = {}
    for name in points_.keys():
        if 'support' in name:
            continue
        id = name_dict[name]

        # if current view doesn't have this object (maybe something unexpected happened and this object is outside the view)
        if id not in list(pc_segments.keys()):
            continue
        ini_pc = pc_segments[id]
        ini_pc_w = np.matmul(cam_to_world[:3,:3], ini_pc.T) + cam_to_world[:3,3].reshape(3,1)
        bottom_z = np.min(ini_pc_w.T, axis=0)[2]
        # render_comparison(points_[name], ini_pc_w.T)
        if name.split('_')[0] in ['teaspoon', 'tablespoon', 'fork', 'knife']:
            # TODO think about how to post process cutlery
            # print(name)
            flip_z = True if mode == 'oracle' else False
            cal_mode_cutlery = 'lowest_rmse' if mode == 'oracle' else 'minimal_R'
            Rt = Rt_calculation(ini_pc_w.T, points_[name], cal_mode=cal_mode_cutlery, flip_z=flip_z)
            Rt_dict[name] = Rt
            transformed_points = np.matmul(Rt['R'], ini_pc_w) + Rt['t'].reshape(3, 1)
            # render_comparison(points_[name], transformed_points.T) # visualize matching
            continue
        shape_ini = np.max(ini_pc_w.T, axis=0) - np.min(ini_pc_w.T, axis=0)
        shape_ = np.max(points_[name], axis=0) - np.min(points_[name], axis=0)
        rate = shape_ini[2] / shape_[2]
        center_ = np.mean(points_[name], axis=0)
        target_points = copy.deepcopy(points_[name])
        target_points -= center_
        target_points *= rate
        target_points += center_
        bottom_z_ = np.min(target_points, axis=0)[2]
        target_points += np.array([0,0,bottom_z-bottom_z_])
        # render_comparison(points_[name], ini_pc_w.T) # visualize initial
        Rt = Rt_calculation(ini_pc_w.T, target_points, cal_mode='lowest_rmse')

        # only for non-cutlery objects
        # Rt = adjust_matrix(ini_pc_w.T, points_[name], Rt['R'])
        transformed_points = np.matmul(Rt['R'], ini_pc_w) + Rt['t'].reshape(3,1)
        # render_comparison(target_points, transformed_points.T) # visualize matching
        Rt_dict[name] = Rt
    return Rt_dict

def align_with_plate(plate_name, bboxes_dict, points_dict, ini_plate_bbox):
    rel_xy = ini_plate_bbox[3:5] - bboxes_dict[plate_name][3:5]
    for i,k in enumerate(bboxes_dict):
        bboxes_dict[k][3:5] += rel_xy
        points_dict[k][:,:2] += rel_xy
    return bboxes_dict, points_dict

def change_dynamics_obj(obj_id, obj_name):
    obj_name = obj_name.split('_')[0]
    if obj_name in ['tablespoon', 'teaspoon', 'knife', 'fork']:
        pybullet.changeDynamics(obj_id, -1, mass=dynamic_dict['cutlery']['mass'], restitution=dynamic_dict['cutlery']['restitution'], lateralFriction=dynamic_dict['cutlery']['lateralFriction'])
    else:
        pybullet.changeDynamics(obj_id, -1, mass=dynamic_dict[obj_name]['mass'], restitution=dynamic_dict[obj_name]['restitution'], lateralFriction=dynamic_dict[obj_name]['lateralFriction'])
def scene_making(env, pc_segments):
    from teasor_vis import random_sampling
    color_dict = {'cup': np.array([250, 217, 213]), 'fork': np.array([176, 227, 230]),
                  'knife': np.array([250, 215, 172]), 'plate': np.array([208, 206, 226]),
                  'teapot': np.array([177,221,240]), 'pitcher': np.array([213,232,212]),
                  'obstacle': np.array([153,153,153])}
    pc_list = []
    color_list = []
    for i,k in enumerate(pc_segments):
        name = get_key(env.pybullet_id_dict, k)[0]
        points = random_sampling(pc_segments[k], 1500)
        pc_list.append(points)
        color_list.append(color_dict[name.split('_')[0]]/255)
    render_result(pc_list, colors=color_list, render_shapes=True, render_boxes=False)


def main():
    rearrange_dataset_base = '/media/ymxlzgy/Data/Dataset/sgbot_dataset'
    graspnet_checkpoint = './contact_graspnet/graspnet_checkpoints/scene_test_2048_bs3_hor_sigma_001' if mode == 'robot' else None
    atlasnet_path = "./AtlasNet/log/AE_AtlasNet_add_cup_20230815T1718/atlasnet_add_cup.pth"
    atlasnet2_path = "./AtlasNet2/log/AE_AtlasNet2_20230408T2110/atlasnet2.pth"
    g2s2_path = "./graphto3d/experiments/graphto3d_2_world_add_cup"

    test_save_path = f'./sg_bot_{mode}_results'
    view_idx = 1
    BOUNDS = np.float32([[-0.4, 0.4], [-0.4, 0.4], [0, 2]])  # X Y Z
    PIXEL_SIZE = (BOUNDS[0, 1] - BOUNDS[0, 0]) / 224
    forward_passes = 5
    point_ae = eval_atlasnet(atlasnet_path)
    # point_ae2 = eval_atlasnet(atlasnet2_path)

    # used to collect train statistics
    stats_dataset = RIODatasetSceneGraph(
        root=rearrange_dataset_base,
        atlas=None,
        atlas2=point_ae,
        path2atlas=atlasnet_path,
        path2atlas2=atlasnet2_path,
        root_raw=os.path.join(rearrange_dataset_base, 'raw'),
        label_file='.obj',
        npoints=5625,
        split='train_scenes',
        use_points=False,
        use_scene_rels=True,
        with_changes=False,
        vae_baseline=False,
        eval=False,
        with_feats=True,
        features_gt="objs_features_gt_atlasnet_separate_cultery.json")
    # dataloader to collect train data statistics
    stats_dataloader = torch.utils.data.DataLoader(
        stats_dataset,
        batch_size=1,
        collate_fn=collate_fn_vaegan,
        shuffle=False,
        num_workers=0)
    obj_classes = stats_dataset.classes
    # in world frame
    g2s_model = eval_g2s(g2s2_path, stats_dataset.vocab, 600)
    g2s_model.compute_statistics(exp=g2s2_path, epoch=600, stats_dataloader=stats_dataloader, force=False)



    pick_place_pair_list = []
    with open(os.path.join(rearrange_dataset_base, f'demo_{mode}.txt'), 'r') as f:
        for file_name in f.readlines():
            file_name = file_name.strip('\n')
            if file_name.split('_')[-1] == 'mid':
                continue
            table_id = file_name.split('_')[0]
            scene_id = file_name.split('_')[1]
            ini_path = os.path.join(rearrange_dataset_base, 'raw', table_id, file_name+f'_view-{view_idx}.json')
            goal_path = os.path.join(rearrange_dataset_base, 'raw', table_id, file_name+f'_goal_view-{view_idx}.json')
            pick_place_pair_list.append((goal_path, ini_path))

    obj_path = os.path.join(rearrange_dataset_base, 'models')
    env = RearrangeEnv(pick_place_pair_list, obj_path, graspnet_checkpoint, forward_passes)
    for id in range(0,len(pick_place_pair_list)):
        result_dict = {}

        # set a scene
        env.reset_rearrange(id)
        result_dict['gt'] = {'init':{'poses':env.ini_pose_dict, 'bboxes': env.ini_bbox_dict}, 'goal':{'poses':env.target_pose_dict, 'bboxes': env.target_bbox_dict}}
        result_dict['pred'] = {'rel_poses':{}, 'final_poses_bullet':{}}
        result_name = pick_place_pair_list[id][0].split('/')[-1].replace(f'_goal_view-{view_idx}.json','')
        result_dir = os.path.join(test_save_path, result_name)
        os.makedirs(result_dir, exist_ok=True)
        shutil.copy(pick_place_pair_list[id][0].replace('json','png'), result_dir)
        shutil.copy(pick_place_pair_list[id][1].replace('json', 'png'), result_dir)


        load_preprocessed = 1

        # generate a goal graph
        if load_preprocessed:
            data = load_goal_data(env.pick_place_info[id][0], obj_classes, atlasnet_path) # !! only 2 is supported
        else:
            data = generate_goal_data(env.pick_place_info[id][0])

        # graph to a scene
        bboxes, points = scene_imagination(data, g2s_model, point_ae, visual=False, fit_box=True)

        # post-process the scene to alleviate the ambiguity
        bboxes_dict, points_dict = post_processing(bboxes, points, data, env.table_height)

        # if there is a plate in the scene, we simply set it as a still object and a reference.
        for i,k in enumerate(env.pybullet_id_dict):
            if 'plate' in k:
                ini_plate_bbox = env.ini_bbox_dict[k] # this can be replaced using the center of points on the plate
                bboxes_dict, points_dict = align_with_plate(k, bboxes_dict, points_dict, ini_plate_bbox)
                break

        result_dict['gt']['imagination'] = {'bboxes': bboxes_dict, 'points': points_dict}
        finished_ids = []
        step=0

        while True:
            # observe the scene
            obs = env.get_observation(BOUNDS, PIXEL_SIZE)

            # predict grasps
            # pc_segments now in camera frame, no grasp inference when oracle
            pred_grasps_cam, scores, openings, pc_segments = env.grasps_infer(obs, forward_passes=forward_passes, mode=mode)
            keys = list(pc_segments.keys())
            random.shuffle(keys)
            pc_segments = {key: pc_segments[key] for key in keys}
            # color_palette = np.array(sns.color_palette('hls', len(pc_segments.keys())))
            # for i, k in enumerate(pc_segments):
            #     render_per_shape(pc_segments[k], color_palette[i])

            # render_ini_and_goal(bboxes_dict, points_dict, obs["points_w"], pc_segments, env.pybullet_id_dict, env.real_cam_to_world)

            rel_dict = match_pc(bboxes_dict, points_dict, pc_segments, env.pybullet_id_dict, env.real_cam_to_world)

            before = env.get_camera_image()
            # exclude the table and previous objs
            pc_segments.pop(env.pybullet_id_dict['support_table'])

            ## temporal coding:
            # scene_making(env, pc_segments)

            for finished_id in finished_ids:
                if finished_id in pc_segments.keys():
                    pc_segments.pop(finished_id)
                print(f'already finished {get_key(env.pybullet_id_dict, finished_id)[0]}.')

            # only has a plate left, don't need to manipulate it. go to the next scene
            if len(list(pc_segments.keys())) == 1:
                if get_key(env.pybullet_id_dict, list(pc_segments.keys())[0])[0].split('_')[0] == 'plate':
                    break
            pc_segments_no_cutlery = copy.deepcopy(pc_segments)

            #### pick one obj and execute the grasp
            cutlery_inside = 0
            success = 0
            super_count = 0 # this is for counting all trials in the scene. only increase once when the trial fails.
            conflict_names = []
            if len(pc_segments) >= 1: # TODO this condition needs to be changed.

                # if cutlery inside, grasp it first.
                execute_cutlery = 0
                for obj_id in list(pc_segments.keys()):
                    cutlery_name = get_key(env.pybullet_id_dict, obj_id)[0]
                    if cutlery_name.split('_')[0] in ['knife', 'fork', 'tablespoon', 'teaspoon']:
                        cutlery_inside = 1
                        pc_segments_no_cutlery.pop(env.pybullet_id_dict[cutlery_name], None)

                        description = f'I think I find {cutlery_name} and will grasp it.'
                        print(description)
                        env.description_list.append(description)

                        change_dynamics_obj(obj_id, cutlery_name)
                        if check_occupancy_new(env, obj_id, points_dict[cutlery_name], pc_segments):
                            super_count += 1
                            conflict_names.append(cutlery_name)
                            continue
                        rel_pose = np.eye(4)
                        rel_pose[:3, :3], rel_pose[:3, 3] = rel_dict[cutlery_name]['R'], rel_dict[cutlery_name]['t']
                        result_dict['pred']['rel_poses'][cutlery_name] = rel_pose

                        if mode == 'oracle':
                            success = oracle_rearrange(env, obj_id, rel_pose)
                            execute_cutlery = 1
                        elif np.any(pred_grasps_cam[obj_id]): # graspnet and have grasps
                            success = execute_rearrange(env, pred_grasps_cam[obj_id], openings[obj_id], scores[obj_id], env.real_cam_to_world, ee_to_tip_franka_cutlery, [0.1, 0.9], rel_pose=rel_pose, post=True)
                            execute_cutlery = 1
                        else: # graspnet and no grasp
                            super_count += 1
                            description = f'It seems like there is no grasps on {cutlery_name}'
                            print(description)
                            env.description_list.append(description)
                            continue

                        # assume it is done and for now exclude it
                        if success:
                            description = f'I assume {cutlery_name} is done.'
                            print(description)
                            env.description_list.append(description)
                            finished_ids.append(obj_id) # TODO: add some conditions
                        break


                # if no cutlery, pick one obj and grasp
                if not cutlery_inside or not execute_cutlery:
                    count = 0 # this is only for counting the trials of common objects. only increase once when the trial fails.
                    execute_obj = 0
                    # continue doing this if no execution or if haven't inspected all objects graspable or not
                    while execute_obj == 0 and count < len(list(pc_segments_no_cutlery.keys())):
                        obj_id_ = list(pc_segments_no_cutlery.keys())[count]
                        obj_name = get_key(env.pybullet_id_dict, obj_id_)[0]
                        description = f'I think I find {obj_name} and will grasp it.'
                        print(description)
                        env.description_list.append(description)
                        change_dynamics_obj(obj_id_, obj_name)
                        rel_pose = None

                        # if no obstacle
                        if obj_name in list(rel_dict.keys()):
                            if check_occupancy_new(env, obj_id_, points_dict[obj_name], pc_segments):
                                count += 1
                                super_count += 1
                                conflict_names.append(obj_name)
                                continue
                            rel_pose = np.eye(4)
                            rel_pose[:3, :3], rel_pose[:3, 3] = rel_dict[obj_name]['R'], rel_dict[obj_name]['t']
                            result_dict['pred']['rel_poses'][obj_name] = rel_pose

                        if mode == 'oracle':
                            success = oracle_rearrange(env, obj_id_, rel_pose)
                            execute_obj = 1
                        elif np.any(pred_grasps_cam[obj_id_]): # graspnet and have grasps
                            success = execute_rearrange(env, pred_grasps_cam[obj_id_], openings[obj_id_], scores[obj_id_], env.real_cam_to_world, ee_to_tip_franka_common, [0.6, 0.4], rel_pose=rel_pose)
                            execute_obj = 1
                        else: # graspnet and no grasp
                            description = f'It seems like there is no grasps on {obj_name}'
                            print(description)
                            env.description_list.append(description)
                            count += 1
                            super_count += 1
                            continue
                        # assume it is done and for now exclude it
                        if success:
                            description = f'I assume {obj_name} is done.'
                            print(description)
                            env.description_list.append(description)
                            finished_ids.append(obj_id_) # TODO: add some conditions

                # Show camera image after pick and place.
                plt.figure(dpi=150)

                ax1 = plt.subplot(1, 2, 1)
                ax1.set_title('Before')
                plt.imshow(before)
                ax1.set_xticks([])
                ax1.set_yticks([])

                ax2 = plt.subplot(1, 2, 2)
                ax2.set_title('After')
                after = env.get_camera_image()
                plt.imshow(after)
                ax2.set_xticks([])
                ax2.set_yticks([])
                plt.savefig(os.path.join(result_dir,result_name+f"_step{step}_{mode}.png"))
                # plt.show()

                # if every trial fails, exit the biggest loop and go to next scene.
                if super_count >= len(list(pc_segments.keys())):
                    description = f'something is wrong. I can\'t rearrange {conflict_names}'
                    print(description)
                    env.description_list.append(description)
                    break

            # all objects are manipulated once, then go to next scene.
            else:
                break

            debug_clip = ImageSequenceClip(env.cache_video, fps=25)
            debug_clip.write_videofile(os.path.join(result_dir,f"step_{step}.mp4"), codec='libx264')
            env.cache_video = []
            step += 1

        # Show camera image after pick and place.
        plt.figure(dpi=150)
        plt.axis('off')
        final = env.get_camera_image()
        plt.imshow(final)
        plt.savefig(os.path.join(result_dir, result_name + f"_final_{mode}.png"), bbox_inches='tight', pad_inches=0)
        for name in list(env.pybullet_id_dict.keys()):
            result_dict['pred']['final_poses_bullet'][name] = env.get_current_pose(env.pybullet_id_dict[name])
        with open(os.path.join(result_dir, "result.pkl"), "wb") as file:
            pickle.dump(result_dict, file)

        with open(os.path.join(result_dir, "description.txt"), "w") as file:
            for item in env.description_list:
                file.write("%s\n" % item)

if __name__ == "__main__":
    main()