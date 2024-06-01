import os
import sys
import argparse
import numpy as np
import time
import glob
import cv2

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import regularize_pc_point_count, depth2pc
from PIL import Image

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image


def depth2xyz(depth_map,depth_cam_matrix,flatten=False,depth_scale=1000):
    fx,fy = depth_cam_matrix[0,0],depth_cam_matrix[1,1]
    cx,cy = depth_cam_matrix[0,2],depth_cam_matrix[1,2]
    h,w=np.mgrid[0:depth_map.shape[0],0:depth_map.shape[1]]
    z=depth_map/depth_scale
    x=(w-cx)*z/fx
    y=(h-cy)*z/fy
    xyz=np.dstack((x,y,z)) if flatten==False else np.dstack((x,y,z)).reshape(-1,3)
    #xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)
    return xyz

def load_available_input_data(p, K=None):
    """
    Load available data from input file path.

    Numpy files .npz/.npy should have keys
    'depth' + 'K' + (optionally) 'segmap' + (optionally) 'rgb'
    or for point clouds:
    'xyz' + (optionally) 'xyz_color'

    png files with only depth data (in mm) can be also loaded.
    If the image path is from the GraspNet dataset, corresponding rgb, segmap and intrinic are also loaded.

    :param p: .png/.npz/.npy file path that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: All available data among segmap, rgb, depth, cam_K, pc_full, pc_colors
    """

    dataset_base = "/home/ymxlzgy/code/polargrasp/dataset/HAMMER/_dataset_processed/test"
    method = p.split("/")[-2]
    scene, id = p.split("/")[-1].split("-")
    rgb_path = os.path.join(dataset_base, scene, 'polarization/rgb', id)
    depth_gt_path = os.path.join(dataset_base, scene, 'polarization/_gt', id)
    segmap, rgb, depth, pc_full, pc_colors = None, None, None, None, None

    if K is not None:
        cam_K = np.array(K).reshape(3, 3)


    if method == 'transcg_example_544_416':
        halfdepth_transcg = np.array(Image.open(p), dtype=np.float32)/1000
        depth = cv2.pyrUp(halfdepth_transcg)
    else:
        depth = np.array(Image.open(p), dtype=np.float32)/1000
    rgb = np.array(Image.open(rgb_path))
    depth_gt = np.array(Image.open(depth_gt_path), dtype=np.float32)/1000
    pc_full, pc_colors= depth2pc(depth,cam_K,rgb=rgb)
    pc_gt, pc_gt_colors=depth2pc(depth_gt,cam_K,rgb=rgb)


    return segmap, rgb, depth, cam_K, pc_full, pc_colors, pc_gt, pc_gt_colors

def inference(global_config, checkpoint_dir, input_paths, K=None, local_regions=True, skip_border_objects=False,
              filter_grasps=True, segmap_id=None, z_range=[0.2, 1.8], forward_passes=1):
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


    save_path = 'results/{}'.format(input_paths.split('/')[-2])
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # Process example test scenes
    camera_intrinsic = np.loadtxt(os.path.join("../polar_intrinsics.txt"))
    for p in glob.glob(input_paths):
        print('Loading ', p)

        pc_segments = {}
        segmap, rgb, depth, cam_K, pc_full, pc_colors, pc_gt, pc_gt_colors = load_available_input_data(p, K=camera_intrinsic)
        pc_segments = dict()
        pc_segments[1] = pc_full

        # if segmap is None and (local_regions or filter_grasps):
        #     raise ValueError('Need segmentation map to extract local regions or filter grasps')

        if pc_full is None:
            print('Converting depth to point cloud(s)...')
            pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                                   skip_border_objects=skip_border_objects,
                                                                                   z_range=z_range)

        print('Generating Grasps...')
        pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full,
                                                                                       pc_segments=pc_segments,
                                                                                       local_regions=local_regions,
                                                                                       filter_grasps=filter_grasps,
                                                                                       forward_passes=forward_passes)

        # print('pred_grasps_cam:', pred_grasps_cam)

        # Save results

        save_path_ = '{}/predictions_{}'.format(save_path,os.path.basename(p.replace('png', 'npz').replace('npy', 'npz')))
        np.savez(save_path_, pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)

        # Visualize results
        show_image(rgb, segmap)
        visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)

    if not glob.glob(input_paths):
        print('No files found: ', input_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='../checkpoints/scene_test_2048_bs3_hor_sigma_001',
                        help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--np_path', default='../transcg_example_544_416/*.png',
                        help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2, 1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False,
                        help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,
                        help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,
                        help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,
                        help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0, help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes,
                                             arg_configs=FLAGS.arg_configs)

    print(str(global_config))
    print('pid: %s' % (str(os.getpid())))

    inference(global_config, FLAGS.ckpt_dir, FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path,
              z_range=eval(str(FLAGS.z_range)),
              K=FLAGS.K, local_regions=FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps,
              segmap_id=FLAGS.segmap_id,
              forward_passes=FLAGS.forward_passes, skip_border_objects=FLAGS.skip_border_objects)

