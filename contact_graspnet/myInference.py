import os
import sys
import argparse
import numpy as np
import time
import glob
import cv2

# ros interface
import rospy
from graspnet_rosinterface import GraspNetRosInterface


import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
import config_utils
# from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

def inference(global_config, 
            checkpoint_dir, 
            input_paths, 
            K=None, 
            local_regions=True, 
            skip_border_objects=False, 
            filter_grasps=True, 
            segmap_id=None, 
            z_range=[0.2,1.8], 
            forward_passes=1
            ):
    """
    Predict 6-DoF grasp distribution for given model and input data
    
    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments. 
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """
    
    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')
    os.makedirs('results', exist_ok=True)

    # ros interface
    graspnet_ros = GraspNetRosInterface()
    while not rospy.is_shutdown():
        
        if graspnet_ros.get_trigger_grasp_get():
            print('Getting PointCloud2 data...')
            # parent_frame = "world"
            # child_frame = "camera_color_optical_frame" 
            
            # parent_frame = "panda_link0"
            # child_frame = "kinect_camera_link" # TODO
            # c2w = graspnet_ros.get_posemat_from_tf_tree(parent_frame=parent_frame, child_frame=child_frame)
            
            
            # easy handeye calib result # TODO
            # trans = [0.2657049369274584, 0.7582260587787445, 0.6898710527756048]
            # quat = [-0.1304642168938061, 0.8338424773061525, -0.5287970746720457, 0.08977452293669871] 
            
            # trans = [0.6846692054034355, 0.6573998750006423, 0.7118518523067432]
            # quat = [-0.051883965984521836, -0.8496333947144131, 0.5204285198689669, 0.06771487552063563]
            # c2w = graspnet_ros.get_tfmat_from_trans_quat(trans, quat)   
            
            parent_frame = "panda_link0"
            child_frame = "rgb_camera_link" # TODO
            c2w = graspnet_ros.get_posemat_from_tf_tree(parent_frame=parent_frame, child_frame=child_frame)
            
                     
            pc_full, pc_colors, _ = graspnet_ros.get_pc_info(c2w=c2w, 
                                                            view_filter=1.0,     # TODO to be configured
                                                            height_filter=0.19,   # TODO to be configured
                                                            flag_get_colors=True) # time consumption: around 1.8s
            # graspimg = graspnet_ros.get_image()
            print('Number of points: {}'.format(len(pc_full)))  
            if len(pc_full) == 0:
                time.sleep(0.5)
                continue
            print('Generating Grasps...')
            pc_segments = dict()
            pc_segments[1] = pc_full
            pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(
                                                                    sess, 
                                                                    pc_full, 
                                                                    pc_segments=pc_segments, 
                                                                    local_regions=False, 
                                                                    filter_grasps=True, 
                                                                    forward_passes=1)
            if len(pred_grasps_cam): 
                print('Visualizing Grasps...')
                grasp_posemat = pred_grasps_cam[1][np.argmax(scores[1])]  # panda_link8 pose
                
                # link8_T_ee = np.array([[ 0.7071, 0.7071,      0,      0], # TODO why?
                #                        [-0.7071, 0.7071,      0,      0],
                #                        [      0,      0,      1, 0.1034],
                #                        [      0,      0,      0,      1]])
                link8_T_ee = np.array([[      1,      0,      0,      0],  # TODO why?
                                       [      0,      1,      0,      0],
                                       [      0,      0,      1,   0.0934],  # 0.1034
                                       [      0,      0,      0,      1]])
                grasp_posemat = np.dot(grasp_posemat, link8_T_ee)         # panda_ee pose
                grasp_posemat = graspnet_ros.pose_rotation_rectify(grasp_posemat, 0, 0, -np.pi/2) # TODO adjust the transformation
                # print('grasp_posemat:', grasp_posemat)
                
                # TODO
                z_backward = np.identity(4)
                z_backward[2, 3] = -0.02
                grasp_posemat = np.dot(grasp_posemat, z_backward)
                
                
                # graspnet_ros.pub_grasping_pose(grasp_posemat, frame_id=child_frame)
                grasp_posemat = np.dot(c2w, grasp_posemat)
                graspnet_ros.pub_grasping_pose(grasp_posemat, frame_id="panda_link0")
                # show_image(graspimg, segmap=None)
                visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
                graspnet_ros.set_grasp_update(True)

        graspnet_ros.rate.sleep()

# region COMMENT
    # # Process example test scenes
    # for p in glob.glob(input_paths):
    #     print('Loading ', p)

    #     pc_segments = {}
    #     segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=K)

    #     if segmap is None and (local_regions or filter_grasps):
    #         raise ValueError('Need segmentation map to extract local regions or filter grasps')

    #     if pc_full is None:
    #         print('Converting depth to point cloud(s)...')
    #         pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
    #                                                                                 skip_border_objects=skip_border_objects, z_range=z_range)
    #         print(type(pc_segments))
    #         print(pc_segments.keys())

    #     print('Generating Grasps...')
    #     filter_grasps = True
    #     pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, 
    #                                                                                       local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)  

    #     # print('pred_grasps_cam:', pred_grasps_cam)

    #     # Save results
    #     np.savez('results/predictions_{}'.format(os.path.basename(p.replace('png','npz').replace('npy','npz'))), 
    #               pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)

    #     # Visualize results          
    #     show_image(rgb, segmap)
    #     visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
        
    # if not glob.glob(input_paths):
    #     print('No files found: ', input_paths)
# endregion COMMENT


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--np_path', default='test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    # parser.add_argument('--filter_grasps', action='store_true', default=True,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    
    print(str(global_config))
    print('pid: %s'%(str(os.getpid())))

    inference(global_config, FLAGS.ckpt_dir, FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path, z_range=eval(str(FLAGS.z_range)),
                K=FLAGS.K, local_regions=FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps, segmap_id=FLAGS.segmap_id, 
                forward_passes=FLAGS.forward_passes, skip_border_objects=FLAGS.skip_border_objects)

