import copy
import os
import glob
import json
import random
import trimesh
import open3d as o3d
import numpy as np
import pybullet
import pybullet_data
import pybullet_utils.transformations as trans
import threading
import time
try:
    import tensorflow._api.v2.compat.v1 as tf
except:
    print("no tensorflow. assume oracle mode.")
from contact_graspnet.contact_grasp_estimator import extract_point_clouds

from moviepy.editor import ImageSequenceClip

PICK_TARGETS = {
    "blue block": None,
    "red block": None,
    "green block": None,
    "yellow block": None,
}
STANDARD_COLORS = ["White"]

# STANDARD_COLORS = [
#     "AliceBlue", "Chartreuse", "Aqua", "Aquamarine", "Azure", "Beige", "Bisque",
#     "BlanchedAlmond", "BlueViolet", "BurlyWood", "CadetBlue", "AntiqueWhite",
#     "Chocolate", "Coral", "CornflowerBlue", "Cornsilk", "Cyan",
#     "DarkCyan", "DarkGoldenRod", "DarkGrey", "DarkKhaki", "DarkOrange",
#     "DarkOrchid", "DarkSalmon", "DarkSeaGreen", "DarkTurquoise", "DarkViolet",
#     "DeepPink", "DeepSkyBlue", "DodgerBlue", "FloralWhite",
#     "ForestGreen", "Fuchsia", "Gainsboro", "GhostWhite", "Gold", "GoldenRod",
#     "Salmon", "Tan", "HoneyDew", "HotPink", "Ivory", "Khaki",
#     "Lavender", "LavenderBlush", "LawnGreen", "LemonChiffon", "LightBlue",
#     "LightCoral", "LightCyan", "LightGoldenRodYellow", "LightGray", "LightGrey",
#     "LightGreen", "LightPink", "LightSalmon", "LightSeaGreen", "LightSkyBlue",
#     "LightSlateGray", "LightSlateGrey", "LightSteelBlue", "LightYellow", "Lime",
#     "LimeGreen", "Linen", "Magenta", "MediumAquaMarine", "MediumOrchid",
#     "MediumPurple", "MediumSeaGreen", "MediumSlateBlue", "MediumSpringGreen",
#     "MediumTurquoise", "MediumVioletRed", "MintCream", "MistyRose", "Moccasin",
#     "NavajoWhite", "OldLace", "Olive", "OliveDrab", "Orange",
#     "Orchid", "PaleGoldenRod", "PaleGreen", "PaleTurquoise", "PaleVioletRed",
#     "PapayaWhip", "PeachPuff", "Peru", "Pink", "Plum", "PowderBlue", "Purple",
#     "RosyBrown", "RoyalBlue", "SaddleBrown", "Green", "SandyBrown",
#     "SeaGreen", "SeaShell", "Sienna", "Silver", "SkyBlue", "SlateBlue",
#     "SlateGray", "SlateGrey", "Snow", "SpringGreen", "SteelBlue", "GreenYellow",
#     "Teal", "Thistle", "Tomato", "Turquoise", "Violet", "Wheat", "White",
#     "WhiteSmoke", "Yellow", "YellowGreen"
# ]

COLORS = {
    "blue": (78 / 255, 121 / 255, 167 / 255, 255 / 255),
    "red": (255 / 255, 87 / 255, 89 / 255, 255 / 255),
    "green": (89 / 255, 169 / 255, 79 / 255, 255 / 255),
    "yellow": (237 / 255, 201 / 255, 72 / 255, 255 / 255),
}

PLACE_TARGETS = {
    "blue block": None,
    "red block": None,
    "green block": None,
    "yellow block": None,

    "blue bowl": None,
    "red bowl": None,
    "green bowl": None,
    "yellow bowl": None,

    "top left corner": (-0.3 + 0.05, -0.2 - 0.05, 0),
    "top right corner": (0.3 - 0.05, -0.2 - 0.05, 0),
    "middle": (0, -0.5, 0),
    "bottom left corner": (-0.3 + 0.05, -0.8 + 0.05, 0),
    "bottom right corner": (0.3 - 0.05, -0.8 + 0.05, 0),
}

PIXEL_SIZE = 0.00267857
BOUNDS = np.float32([[-0.3, 0.3], [-0.8, -0.2], [0, 0.15]])  # X Y Z


class Robotiq2F85:
    """Gripper handling for Robotiq 2F85."""

    def __init__(self, robot, tool):
        self.robot = robot
        self.tool = tool
        pos = [0.1339999999999999, -0.49199999999872496, 0.5]
        rot = pybullet.getQuaternionFromEuler([np.pi, 0, np.pi])
        urdf = "robotiq_2f_85/robotiq_2f_85.urdf"
        self.body = pybullet.loadURDF(urdf, pos, rot)
        self.n_joints = pybullet.getNumJoints(self.body)
        self.activated = False

        # Connect gripper base to robot tool.
        pybullet.createConstraint(self.robot, tool, self.body, 0, jointType=pybullet.JOINT_FIXED, jointAxis=[0, 0, 0],
                                  parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, -0.07],
                                  childFrameOrientation=pybullet.getQuaternionFromEuler([0, 0, np.pi / 2]))

        # Set friction coefficients for gripper fingers.
        for i in range(self.n_joints):
            pybullet.changeDynamics(self.body, i, lateralFriction=10.0, spinningFriction=1.0, rollingFriction=1.0,
                                    frictionAnchor=True)

        self.joint_name_dict = {}

        for i in range(self.n_joints):
            joint_info = pybullet.getJointInfo(self.body, i)
            self.joint_name_dict[joint_info[1].decode('utf-8')] = joint_info[0]
            pybullet.enableJointForceTorqueSensor(self.body, joint_info[0], 1)

        # Start thread to handle additional gripper constraints.
        self.motor_joint = 1
        self.constraints_thread = threading.Thread(target=self.step)
        self.constraints_thread.daemon = True
        self.constraints_thread.start()

        pybullet.changeDynamics(self.body, linkIndex=4, lateralFriction=0.8)
        pybullet.changeDynamics(self.body, linkIndex=9, lateralFriction=0.8)

    def list_joint_names(self):
        num_joints = pybullet.getNumJoints(self.body)
        for i in range(num_joints):
            joint_info = pybullet.getJointInfo(self.body, i)
            joint_id = joint_info[0]
            joint_name = joint_info[1].decode('utf-8')
            print(f"Joint ID: {joint_id}, Joint Name: {joint_name}")

    # Control joint positions by enforcing hard contraints on gripper behavior.
    # Set one joint as the open/close motor joint (other joints should mimic).
    def step(self):
        while True:
            try:
                currj = [pybullet.getJointState(self.body, i)[0] for i in range(self.n_joints)]
                indj = [6, 3, 8, 5, 10]
                targj = [currj[1], -currj[1], -currj[1], currj[1], currj[1]]
                pybullet.setJointMotorControlArray(self.body, indj, pybullet.POSITION_CONTROL, targj,
                                                   positionGains=np.ones(5))
            except:
                return
            time.sleep(0.001)

    # Close gripper fingers.
    def activate(self, target_position=None, target_control=False, max_torque=100, torque_control=False):
        if target_control:
            pybullet.setJointMotorControl2(self.body, self.motor_joint, pybullet.POSITION_CONTROL, targetVelocity=1,
                                           force=10, targetPosition=target_position)
        elif torque_control:
            pybullet.setJointMotorControl2(self.body, self.motor_joint, pybullet.TORQUE_CONTROL, force=max_torque)
        else:
            pybullet.setJointMotorControl2(self.body, self.motor_joint, pybullet.VELOCITY_CONTROL, targetVelocity=0.8,
                                           force=13)
        self.activated = True

    def stop(self):
        pybullet.setJointMotorControl2(self.body, self.motor_joint, pybullet.VELOCITY_CONTROL, targetVelocity=0)
        self.activated = False

    # Open gripper fingers.
    def release(self):
        pybullet.setJointMotorControl2(self.body, self.motor_joint, pybullet.VELOCITY_CONTROL, targetVelocity=-1,
                                       force=10)
        self.activated = False

    # If activated and object in gripper: check object contact.
    # If activated and nothing in gripper: check gripper contact.
    # If released: check proximity to surface (disabled).
    def detect_contact(self):
        obj, _, ray_frac = self.check_proximity()
        if self.activated:
            empty = self.grasp_width() < 0.01
            cbody = self.body if empty else obj
            if obj == self.body or obj == 0:
                return False
            return self.external_contact(cbody)

    #   else:
    #     return ray_frac < 0.14 or self.external_contact()

    # Return if body is in contact with something other than gripper
    def external_contact(self, body=None):
        if body is None:
            body = self.body
        pts = pybullet.getContactPoints(bodyA=body)
        pts = [pt for pt in pts if pt[2] != self.body]
        return len(pts) > 0  # pylint: disable=g-explicit-length-test

    def check_grasp(self):
        while self.moving():
            time.sleep(0.001)
        success = self.grasp_width() > 0.01
        return success

    def grasp_width(self):
        lpad = np.array(pybullet.getLinkState(self.body, 4)[0])
        rpad = np.array(pybullet.getLinkState(self.body, 9)[0])
        dist = np.linalg.norm(lpad - rpad) - 0.047813
        return dist

    def check_finger_force(self):

        lpad_link_id = 4
        rpad_link_id = 9

        # Get contact points for the fingertip
        lcontact_points = pybullet.getContactPoints(bodyA=self.body, linkIndexA=lpad_link_id)
        rcontact_points = pybullet.getContactPoints(bodyA=self.body, linkIndexA=rpad_link_id)

        # Calculate the total force on the fingertip
        ltotal_force = sum([cp[9] for cp in lcontact_points])
        rtotal_force = sum([cp[9] for cp in rcontact_points])
        print(ltotal_force, rtotal_force)
        return ltotal_force, rtotal_force

    def check_proximity(self):
        ee_pos = np.array(pybullet.getLinkState(self.robot, self.tool)[0])
        tool_pos = np.array(pybullet.getLinkState(self.body, 0)[0])
        vec = (tool_pos - ee_pos) / np.linalg.norm((tool_pos - ee_pos))
        ee_targ = ee_pos + vec
        ray_data = pybullet.rayTest(ee_pos, ee_targ)[0]
        obj, link, ray_frac = ray_data[0], ray_data[1], ray_data[2]
        return obj, link, ray_frac


# @markdown Gym-style environment code
class RearrangeEnv():

    def __init__(self, pick_place_info, obj_path, graspnet_checkpoint=None, forward_passes=5):
        self.dt = 1 / 480
        self.sim_step = 0
        self.pick_place_info = pick_place_info
        self.graspnet_checkpoint = graspnet_checkpoint
        if graspnet_checkpoint is not None:
            import contact_graspnet.config_utils as config_utils
            from contact_graspnet.contact_grasp_estimator import GraspEstimator
            from contact_graspnet.visualization_utils import show_image, plot_grasp_img, maximize_score_and_z_alignment

            global_config = config_utils.load_config(graspnet_checkpoint, batch_size=forward_passes,
                                                     arg_configs=[])
            self.grasp_estimator = GraspEstimator(global_config)
            self.grasp_estimator.build_network()
            self.show_image = show_image
            self.plot_grasp_img = plot_grasp_img
            self.maximize_score_and_z_alignment = maximize_score_and_z_alignment
        # Configure and start PyBullet.
        # python3 -m pybullet_utils.runServer
        # pybullet.connect(pybullet.SHARED_MEMORY)  # pybullet.GUI for local GUI.
        pybullet.connect(pybullet.GUI)  # pybullet.GUI for local GUI.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        pybullet.setPhysicsEngineParameter(enableFileCaching=0)
        assets_path = os.path.dirname(os.path.abspath(""))
        pybullet.setAdditionalSearchPath(assets_path)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setTimeStep(self.dt)

        # self.home_joints = (
        #     np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 3 * np.pi / 2, 0)  # Joint angles: (J0, J1, J2, J3, J4, J5).
        self.home_joints = (
            np.pi * 3 / 4, -np.pi / 2, np.pi / 4, -np.pi / 3, 3 * np.pi / 2, 0) # Joint angles: (J0, J1, J2, J3, J4, J5).
        self.home_ee_euler = (np.pi, 0, np.pi)  # (RX, RY, RZ) rotation in Euler angles.
        self.ee_link_id = 9  # Link ID of UR5 end effector.
        self.tip_link_id = 10  # Link ID of gripper finger tips.
        self.gripper = None
        self.model_base = obj_path

    def grasps_infer(self, obs, local_regions=True, filter_grasps=True, skip_border_objects=False, z_range=[0.2, 1.8],
                     forward_passes=1, mode='oracle'):

        os.makedirs('grasp_results', exist_ok=True)
        segmap, rgb, depth, cam_K, pc_full = obs['seg'], obs['vis'], obs['depth'], obs['cam_K'], obs['points_cam']
        pc_full, pc_segments, pc_colors = extract_point_clouds(depth, cam_K, segmap=segmap,
                                                                                    rgb=rgb,
                                                                                    skip_border_objects=skip_border_objects,
                                                                                    z_range=z_range)
        if mode == 'oracle':
            return None, None, None, pc_segments

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Load weights
        self.grasp_estimator.load_weights(sess, saver, self.graspnet_checkpoint, mode='test')

        print('Generating Grasps...')

        # pred_grasps_cam{seg_id:[n,4x4]}, scores{seg_id:[n,1]}
        pred_grasps_cam, scores, contact_pts, openings = self.grasp_estimator.predict_scene_grasps(sess, pc_full,
                                                                                                   pc_segments=pc_segments,
                                                                                                   local_regions=local_regions,
                                                                                                   filter_grasps=filter_grasps,
                                                                                                   forward_passes=forward_passes)
        # Visualize results
        self.show_image(rgb, segmap)
        self.plot_grasp_img(rgb, segmap, pred_grasps_cam, scores, obs['cam_K'])
        # visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
        return pred_grasps_cam, scores, openings, pc_segments

    def load_model_from_obj(self, cat, obj_path, scale, pos, ori):
        obj = trimesh.load(obj_path)
        min_point = obj.bounds[0]
        y_min = min_point[1]
        z_min = min_point[2]
        if cat != 'tablespoon' and cat != 'teaspoon':
            mass_origin = np.array([obj.centroid[0], obj.centroid[1], z_min]) * scale
        else:
            mass_origin = np.array([obj.centroid[0], y_min, obj.centroid[2]]) * scale

        visualShapeId = pybullet.createVisualShape(shapeType=pybullet.GEOM_MESH,
                                                   fileName=obj_path,  # Replace with your mesh file
                                                   visualFramePosition=[0, 0, 0],
                                                   meshScale=[scale, scale, scale])
        collisionShapeId = pybullet.createCollisionShape(shapeType=pybullet.GEOM_MESH,
                                                         fileName=obj_path,  # Replace with your mesh file
                                                         collisionFramePosition=[0, 0, 0],
                                                         meshScale=[scale, scale, scale])
        multiBodyId = pybullet.createMultiBody(baseMass=0,
                                               baseInertialFramePosition=mass_origin,
                                               baseCollisionShapeIndex=collisionShapeId,
                                               baseVisualShapeIndex=visualShapeId,
                                               basePosition=pos,
                                               baseOrientation=ori,
                                               useMaximalCoordinates=True)
        pos_, ori_ = pybullet.getBasePositionAndOrientation(multiBodyId)

        return multiBodyId

    def reset_rearrange(self, id, use_ur5=True):
        self.pybullet_id_dict = {}
        self.description_list = []
        self.rel_pose_dict = {}
        self.target_pose_dict = {}
        self.target_bbox_dict = {}
        self.ini_pose_dict = {}
        self.ini_bbox_dict = {}
        self.use_ur5 = use_ur5
        pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
        pybullet.setGravity(0, 0, -9.8)
        self.cache_video = []

        # # 创建坐标轴的长度
        # axis_length = 1.0
        #
        # # 创建坐标轴的原点
        # origin = [0, 0, 0]
        #
        # # 创建x轴
        # pybullet.addUserDebugLine(origin, [axis_length, 0, 0], [1, 0, 0])
        #
        # # 创建y轴
        # pybullet.addUserDebugLine(origin, [0, axis_length, 0], [0, 1, 0])
        #
        # # 创建z轴
        # pybullet.addUserDebugLine(origin, [0, 0, axis_length], [0, 0, 1])
        #
        # # 创建轴标签
        # pybullet.addUserDebugText('X', [axis_length, 0, 0], textColorRGB=[1, 0, 0])
        # pybullet.addUserDebugText('Y', [0, axis_length, 0], textColorRGB=[0, 1, 0])
        # pybullet.addUserDebugText('Z', [0, 0, axis_length], textColorRGB=[0, 0, 1])

        # Temporarily disable rendering to load URDFs faster.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        # Add robot.
        pybullet.loadURDF("plane.urdf", [0, 0, -0.001])
        if self.use_ur5:
            self.robot_id = pybullet.loadURDF("ur5e/ur5e.urdf", [0, -0.5, 0],
                                              flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL)
            self.ghost_id = pybullet.loadURDF("ur5e/ur5e.urdf", [0, 0, -10])  # For forward kinematics.
            self.joint_ids = [pybullet.getJointInfo(self.robot_id, i) for i in
                              range(pybullet.getNumJoints(self.robot_id))]
            self.joint_ids = [j[0] for j in self.joint_ids if j[2] == pybullet.JOINT_REVOLUTE]
        else:
            self.robot_id = pybullet.loadURDF("franka_description/robots/franka_panda.urdf", [0, 0, 0],
                                              flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL)
            self.ghost_id = pybullet.loadURDF("franka_description/robots/franka_panda.urdf",
                                              [0, 0, -10])  # For forward kinematics.
            self.joint_ids = [pybullet.getJointInfo(self.robot_id, i) for i in
                              range(pybullet.getNumJoints(self.robot_id))]
            self.joint_ids = [j[0] for j in self.joint_ids if j[2] == pybullet.JOINT_REVOLUTE]
        self.pybullet_id_dict['robot'] = self.robot_id
        self.pybullet_id_dict['ghost'] = self.ghost_id

        # Move robot to home configuration.
        for i in range(len(self.joint_ids)):
            pybullet.resetJointState(self.robot_id, self.joint_ids[i], self.home_joints[i])

        if self.use_ur5:
            # Add gripper.
            if self.gripper is not None:
                while self.gripper.constraints_thread.is_alive():
                    self.constraints_thread_active = False
            self.gripper = Robotiq2F85(self.robot_id, self.ee_link_id)
            self.pybullet_id_dict['gripper'] = self.gripper.body
            self.gripper.release()
            # self.gripper.list_joint_names()

            # ee_to_tip transformation
            ee_state = pybullet.getLinkState(self.robot_id, self.ee_link_id)
            tip_state = pybullet.getLinkState(self.robot_id, self.tip_link_id)
            tip_rot_mat = pybullet.getMatrixFromQuaternion(tip_state[1])
            ee_rot_mat = pybullet.getMatrixFromQuaternion(ee_state[1])
            tip_transform_mat = np.eye(4)
            ee_transform_mat = np.eye(4)
            tip_transform_mat[:3, :3] = np.reshape(tip_rot_mat, (3, 3))
            ee_transform_mat[:3, :3] = np.reshape(ee_rot_mat, (3, 3))
            tip_transform_mat[:3, 3] = tip_state[0]
            ee_transform_mat[:3, 3] = ee_state[0]
            self.ee_to_tip = np.dot(np.linalg.inv(tip_transform_mat), ee_transform_mat)

        # Load objects according to config.
        place_info, pick_info = self.pick_place_info[id][0], self.pick_place_info[id][1]
        print(f'now processing {pick_info}')
        with open(pick_info) as f:
            self.source_info = json.load(f)
        with open(place_info) as f:
            self.target_info = json.load(f)

        self.cam_info = self.source_info['camera_data']
        z_support_max = 0
        for source in self.source_info['objects']:
            class_name = source['class']
            object_name = source['name']
            if class_name != 'support_table':
                urdf_path = os.path.join(self.model_base, class_name, object_name + '.urdf')
            else:
                urdf_path = os.path.join(self.model_base, class_name, object_name.split('_')[0] + '.urdf')
                support_mesh = trimesh.load(urdf_path.replace('.urdf', '.obj')).apply_scale(source['scale'])
                z_support_max = support_mesh.bounds[1][2]
                self.table_height = z_support_max
                object_name = 'support_table'
            # obj = trimesh.load(urdf_path.replace('.urdf','.obj')).apply_scale(info['scale'])
            ini_pose = np.array(source['local_to_world_matrix']).reshape(4, 4).T
            ini_bbox = np.array(source['param6'])

            # normalize scale sR /= s  s = Frobenius
            rot_F = np.linalg.norm(ini_pose[:3, :3], ord='fro')
            ini_pose[:3, :3] /= (rot_F / np.sqrt(3))
            # obj_list.append(obj.copy().apply_transform(pose))
            object_position = ini_pose[:3, 3]
            object_orieation = trans.quaternion_from_matrix(ini_pose)
            object_id = pybullet.loadURDF(urdf_path, object_position, object_orieation)
            self.pybullet_id_dict[object_name] = object_id
            self.ini_pose_dict[object_name] = ini_pose
            self.ini_bbox_dict[object_name] = ini_bbox

            # do target pose statistics. TODO This is not the final version. Only for the transformation test. Target poses should be calculated by ICP or sth similar.
            target_pose = None
            target_bbox = None

            for target in self.target_info['objects']:
                if target['name'] == object_name:
                    target_pose = np.array(target['local_to_world_matrix']).reshape(4, 4).T
                    target_bbox = np.array(target['param6'])
                    # normalize scale sR /= s  s = Frobenius
                    rot_target = np.linalg.norm(target_pose[:3, :3], ord='fro')
                    target_pose[:3, :3] /= (rot_target / np.sqrt(3))
                    break
            self.target_pose_dict[object_name] = target_pose
            self.target_bbox_dict[object_name] = target_bbox
            # rel_pose = None
            # for target in target_info['objects']:
            #     if target['name'] == object_name:
            #         target_pose = np.array(target['local_to_world_matrix']).reshape(4,4).T
            #         # normalize scale sR /= s  s = Frobenius
            #         rot_target = np.linalg.norm(target_pose[:3, :3], ord='fro')
            #         target_pose[:3, :3] /= (rot_target / np.sqrt(3))
            #         rel_pose = np.matmul(target_pose, np.linalg.inv(ini_pose))
            #         break
            # self.rel_pose_dict[object_name] = rel_pose

            # TODO adjust mass
            pybullet.changeDynamics(object_id, -1, mass=0)
            # pybullet.changeDynamics(object_id, -1, mass=0.2, restitution=0.5)
            # if class_name == 'knife' or class_name == 'fork' or class_name == 'tablespoon' or class_name == 'teaspoon':
            #     pybullet.changeDynamics(object_id, -1, mass=0.1, restitution=0.5)
            if class_name == 'support_table':
                position, _ = pybullet.getBasePositionAndOrientation(object_id)
                pybullet.changeDynamics(object_id, -1, mass=0)
                # pybullet.createConstraint(
                #     parentBodyUniqueId=object_id,
                #     parentLinkIndex=-1,  # 表示物体的基础部分
                #     childBodyUniqueId=-1,  # 表示世界坐标系
                #     childLinkIndex=-1,
                #     jointType=pybullet.JOINT_FIXED,  # 表示这是一个固定的约束
                #     jointAxis=(0, 0, 0),
                #     parentFramePosition=position,
                #     childFramePosition=(0, 0, 0)
                # )

        # trimesh.scene.Scene(obj_list).show()

        # Reset the robot position to a higher place
        position_robot, _ = pybullet.getBasePositionAndOrientation(self.robot_id)
        position_robot = np.array(position_robot) + np.array([0, 0, self.table_height])
        pybullet.resetBasePositionAndOrientation(self.robot_id, position_robot,
                                                 pybullet.getQuaternionFromEuler([0, 0, -180]))
        # Move robot to home configuration.
        for i in range(len(self.joint_ids)):
            pybullet.resetJointState(self.robot_id, self.joint_ids[i], self.home_joints[i])

        # Re-enable rendering.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        for _ in range(1000):
            pybullet.stepSimulation()
            time.sleep(1. / 240.)
        return

    def check_limit(self):
        for joint_id in self.joint_ids:
            joint_info = pybullet.getJointInfo(self.robot_id, joint_id)
            joint_lower_limit = joint_info[8]  # 最小限位
            joint_upper_limit = joint_info[9]  # 最大限位
            joint_state = pybullet.getJointState(self.robot_id, joint_id)
            joint_position = joint_state[0]
            if joint_position < joint_lower_limit or joint_position > joint_upper_limit:
                print(f'Joint {joint_id} of the robot is out of limits')
                # # Move robot to home configuration.
                # for i in range(len(self.joint_ids)):
                #     pybullet.setJointMotorControl2(self.robot_id, self.joint_ids[i], pybullet.POSITION_CONTROL,
                #                                    targetPosition=self.home_joints[i])
                # for _ in range(720):
                #     self.step_sim_and_render()
                return True
        return False

    def servoj(self, joints, gain=0.01):
        """Move to target joint positions with position control."""
        pybullet.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.joint_ids,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=joints,
            positionGains=[gain] * 6)

    def movep(self, position, gain=0.01):
        """Move to target end effector position."""
        joints = pybullet.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.tip_link_id,
            targetPosition=position,
            targetOrientation=pybullet.getQuaternionFromEuler(self.home_ee_euler),
            maxNumIterations=100)
        self.servoj(joints, gain)

    def move_matrix(self, matrix, gain=0.01):
        """Move to target end effector position."""
        position, orientation_q = matrix[:3, 3], trans.quaternion_from_matrix(matrix)
        joints = pybullet.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.tip_link_id,
            targetPosition=position,
            targetOrientation=orientation_q,
            maxNumIterations=100)
        self.servoj(joints, gain)

    def move_pos_ori(self, position, orientation_q, gain=0.01):
        """Move to target end effector position."""
        joints = pybullet.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.tip_link_id,
            targetPosition=position,
            targetOrientation=orientation_q,
            maxNumIterations=100)
        self.servoj(joints, gain)

    def check_obj_gripper(self, gripper_id):
        object_in_gripper = pybullet.getContactPoints(bodyA=gripper_id)
        if not object_in_gripper:
            description = "I guess the object is not in my gripper, returning.."
            print(description)
            self.description_list.append(description)
        return object_in_gripper

    def pick_step(self, grasps_w, opening, score, ee_to_tip):
        """Do pick and place motion primitive."""
        done = False
        best_grasp_w, best_opening = self.maximize_score_and_z_alignment(grasps_w, score, opening,
                                                                    target_z=np.array([0, 0, -1]), score_weight=1,
                                                                    alignment_weight=0)
        grasp_tip = np.matmul(best_grasp_w, np.linalg.inv(ee_to_tip))
        # # change gravity
        # pybullet.changeDynamics(obj_id, -1, mass=0.1)

        # climb to the higher space
        tip_state = pybullet.getLinkState(self.robot_id, self.tip_link_id)
        tip_xyz = np.float32(tip_state[0])
        hover_xyz_start = tip_xyz + np.float32([0, 0, 0.2])
        hover_xyz_lift = grasp_tip[:3, 3] + np.float32([0, 0, 0.2])
        dist_p = 100
        while dist_p > 0.01:
            if self.check_limit():
                return done
            tip_state = pybullet.getLinkState(self.robot_id, self.tip_link_id)
            tip_xyz = np.float32(tip_state[0])
            dist_p = np.linalg.norm(hover_xyz_start - tip_xyz)
            self.movep(hover_xyz_start)
            self.step_sim_and_render()

        # Move close to object.
        dist_p, dist_r = 100, 100
        close_pos = grasp_tip[:3, 3] - grasp_tip[:3, 2] * 0.03
        close_ori = trans.quaternion_from_matrix(grasp_tip)
        while dist_p > 0.01 or dist_r > 0.01:
            if self.check_limit():
                return done
            tip_state = pybullet.getLinkState(self.robot_id, self.tip_link_id)
            tip_xyz, tip_rot = np.float32(tip_state[0]), np.float32(tip_state[1])
            dist_p = np.linalg.norm(close_pos - tip_xyz)
            dist_r = 2 * np.arccos(abs(np.dot(close_ori, tip_rot)))
            self.move_pos_ori(close_pos, close_ori)
            self.step_sim_and_render()

        # Move to object.
        dist_p, dist_r = 100, 100
        while dist_p > 0.01 or dist_r > 0.01:
            if self.check_limit():
                return done
            tip_state = pybullet.getLinkState(self.robot_id, self.tip_link_id)
            tip_xyz, tip_rot = np.float32(tip_state[0]), np.float32(tip_state[1])
            dist_p = np.linalg.norm(grasp_tip[:3, 3] - tip_xyz)
            dist_r = 2 * np.arccos(abs(np.dot(trans.quaternion_from_matrix(grasp_tip), tip_rot)))
            self.move_matrix(grasp_tip, gain=0.005)
            self.step_sim_and_render()

        # Pick up object.
        force_threshold = 500
        t = 0
        while t < 2400:
            # right follower
            right_finger_info = pybullet.getJointState(bodyUniqueId=self.gripper.body,
                                                       jointIndex=self.gripper.joint_name_dict[
                                                           'robotiq_2f_85_right_follower_joint'])
            right_reaction_force = np.linalg.norm(right_finger_info[2][:3])  # right finger force
            left_finger_info = pybullet.getJointState(bodyUniqueId=self.gripper.body,
                                                      jointIndex=self.gripper.joint_name_dict[
                                                          'robotiq_2f_85_left_follower_joint'])
            left_reaction_force = np.linalg.norm(left_finger_info[2][:3])  # left finger force
            # print(f'the force on the left finger is {left_reaction_force}, the force on the right finger is {right_reaction_force}')

            # if abs(right_reaction_force) > force_threshold and abs(left_reaction_force) > force_threshold:
            #     self.gripper.stop()  # Stop closing the gripper
            #     break
            # else:
            self.gripper.activate()
            t += 1
            self.step_sim_and_render()

        # climb up with the object.
        dist_p = 100
        while dist_p > 0.01:
            if self.check_limit():
                return done
            tip_state = pybullet.getLinkState(self.robot_id, self.tip_link_id)
            tip_xyz = np.float32(tip_state[0])
            dist_p = np.linalg.norm(hover_xyz_lift - tip_xyz)
            self.movep(hover_xyz_lift, 0.005)
            self.step_sim_and_render()

        # Move to a temporary location.
        dist_p = 100
        tem_loc = np.array([0.5, 0.5, 0.5])
        while dist_p > 0.01:
            if self.check_limit():
                return done
            tip_state = pybullet.getLinkState(self.robot_id, self.tip_link_id)
            tip_xyz = np.float32(tip_state[0])
            dist_p = np.linalg.norm(tem_loc - tip_xyz)
            self.movep(tem_loc, 0.001)
            self.step_sim_and_render()

        # drop the object
        self.gripper.release()
        for _ in range(720):
            self.step_sim_and_render()

        # Move robot to home configuration.
        for i in range(len(self.joint_ids)):
            pybullet.setJointMotorControl2(self.robot_id, self.joint_ids[i], pybullet.POSITION_CONTROL,
                                           targetPosition=self.home_joints[i])
        for _ in range(720):
            self.step_sim_and_render()

        # observation = self.get_observation()
        # reward = self.get_reward()
        done = True
        info = {}
        debug_clip = ImageSequenceClip(self.cache_video, fps=25)
        return done

    def rearrange_step(self, grasps_w, opening, score, ee_to_tip, weight=[0.5, 0.5], rel_pose=None, post=False):
        """Do pick and place motion primitive."""
        done = False
        best_grasp_w, best_opening = self.maximize_score_and_z_alignment(grasps_w, score, opening,
                                                                    target_z=np.array([0, 0, -1]),
                                                                    score_weight=weight[0], alignment_weight=weight[1],
                                                                    rel_pose=rel_pose, post=post)
        grasp_tip = np.matmul(best_grasp_w, np.linalg.inv(ee_to_tip))
        place_pos = np.array([0.5, -0.4, grasp_tip[2, 3] + 0.2])
        place_ori = trans.quaternion_from_matrix(grasp_tip)

        # rel_pose is None means it is needed to be thrown away
        if rel_pose is not None:
            target = np.matmul(rel_pose, grasp_tip)
            target_ori = trans.quaternion_from_matrix(target)
            target_pos = target[:3, 3]
            place_pos = np.array([target_pos[0], target_pos[1], target_pos[2] + 0.2])
            place_ori = target_ori

        # # climb to the higher space
        # tip_state = pybullet.getLinkState(self.robot_id, self.tip_link_id)
        # tip_xyz = np.float32(tip_state[0])
        # hover_xyz_start = tip_xyz + np.float32([0, 0, 0.2])
        # dist_p = 100
        # while dist_p > 0.01:
        #     if self.check_limit():
        #         return done
        #     tip_state = pybullet.getLinkState(self.robot_id, self.tip_link_id)
        #     tip_xyz = np.float32(tip_state[0])
        #     dist_p = np.linalg.norm(hover_xyz_start - tip_xyz)
        #     self.movep(hover_xyz_start)
        #     self.step_sim_and_render()

        # Move close to object.
        dist_p, dist_r = 100, 100
        close_pos = grasp_tip[:3, 3] - grasp_tip[:3, 2] * 0.03
        close_ori = trans.quaternion_from_matrix(grasp_tip)
        t = 0
        while dist_p > 0.01 or dist_r > 0.01:
            if self.check_limit():
                return done
            tip_state = pybullet.getLinkState(self.robot_id, self.tip_link_id)
            tip_xyz, tip_rot = np.float32(tip_state[0]), np.float32(tip_state[1])
            dist_p = np.linalg.norm(close_pos - tip_xyz)
            dist_r = 2 * np.arccos(abs(np.dot(close_ori, tip_rot)))
            self.move_pos_ori(close_pos, close_ori)
            self.step_sim_and_render()
            t += 1
            if t > 10000:
                description = "cant reach the obj"
                self.description_list.append(description)
                print(description)
                return done

        # Move to object.
        dist_p, dist_r = 100, 100
        t = 0
        while dist_p > 0.01 or dist_r > 0.01:
            if self.check_limit():
                return done
            tip_state = pybullet.getLinkState(self.robot_id, self.tip_link_id)
            tip_xyz, tip_rot = np.float32(tip_state[0]), np.float32(tip_state[1])
            dist_p = np.linalg.norm(grasp_tip[:3, 3] - tip_xyz)
            dist_r = 2 * np.arccos(abs(np.dot(trans.quaternion_from_matrix(grasp_tip), tip_rot)))
            self.move_matrix(grasp_tip)
            self.step_sim_and_render()
            t += 1
            if t > 10000:
                description = "cant move to the obj"
                self.description_list.append(description)
                print(description)
                return done

        # Pick up object.
        force_threshold = 500
        t = 0
        while t < 2400:
            # right follower
            right_finger_info = pybullet.getJointState(bodyUniqueId=self.gripper.body,
                                                       jointIndex=self.gripper.joint_name_dict[
                                                           'robotiq_2f_85_right_follower_joint'])
            right_reaction_force = np.linalg.norm(right_finger_info[2][:3])  # right finger force
            left_finger_info = pybullet.getJointState(bodyUniqueId=self.gripper.body,
                                                      jointIndex=self.gripper.joint_name_dict[
                                                          'robotiq_2f_85_left_follower_joint'])
            left_reaction_force = np.linalg.norm(left_finger_info[2][:3])  # left finger force
            # print(f'the force on the left finger is {left_reaction_force}, the force on the right finger is {right_reaction_force}')

            # if abs(right_reaction_force) > force_threshold and abs(left_reaction_force) > force_threshold:
            #     self.gripper.stop()  # Stop closing the gripper
            #     break
            # else:
            # TODO adjust this function
            self.gripper.check_finger_force()
            self.gripper.activate()
            t += 1
            self.step_sim_and_render()

        # climb up with the object with the same gesture.
        hover_xyz_lift = grasp_tip[:, 3] + np.float32([0, 0, 0.2, 0])
        hover_lift_m = np.concatenate((grasp_tip[:, 0:3], hover_xyz_lift.reshape(4, -1)), axis=1)
        dist_p, dist_r = 100, 100
        t = 0
        while dist_p > 0.01 or dist_r > 0.01:
            if self.check_limit() or not self.check_obj_gripper(self.gripper.body):
                return done
            tip_state = pybullet.getLinkState(self.robot_id, self.tip_link_id)
            tip_xyz, tip_rot = np.float32(tip_state[0]), np.float32(tip_state[1])
            dist_p = np.linalg.norm(hover_xyz_lift[:3] - tip_xyz)
            dist_r = 2 * np.arccos(abs(np.dot(trans.quaternion_from_matrix(hover_lift_m), tip_rot)))
            self.gripper.activate()
            self.move_matrix(hover_lift_m, 0.005)
            self.step_sim_and_render()
            self.gripper.check_finger_force()
            t += 1
            if t > 10000:
                print("cant climb up")

        # Move to the target location using the targeted ori gesture.
        dist_p, dist_r = 100, 100
        t = 0
        unreach_flag = 0
        while dist_p > 0.01 or dist_r > 0.01:
            if self.check_limit() or not self.check_obj_gripper(self.gripper.body):
                return done
            tip_state = pybullet.getLinkState(self.robot_id, self.tip_link_id)
            tip_xyz, tip_rot = np.float32(tip_state[0]), np.float32(tip_state[1])
            dist_p = np.linalg.norm(place_pos - tip_xyz)
            dist_r = 2 * np.arccos(abs(np.dot(place_ori, tip_rot)))
            self.move_pos_ori(place_pos, place_ori, 0.001)
            self.step_sim_and_render()
            self.gripper.check_finger_force()
            t += 1
            if t > 10000:
                description = "can't move to the target location using the targeted ori gesture. I will drop it"
                self.description_list.append(description)
                print(description)
                unreach_flag = 1
                break

        # low down the object
        lowdown_xyz = place_pos - np.float32([0, 0, 0.17])
        dist_p, dist_r = 100, 100
        while dist_p > 0.01 or dist_r > 0.01:
            if self.check_limit() or not self.check_obj_gripper(self.gripper.body):
                return done
            tip_state = pybullet.getLinkState(self.robot_id, self.tip_link_id)
            tip_xyz, tip_rot = np.float32(tip_state[0]), np.float32(tip_state[1])
            dist_p = np.linalg.norm(lowdown_xyz - tip_xyz)
            dist_r = 2 * np.arccos(abs(np.dot(place_ori, tip_rot)))
            self.move_pos_ori(lowdown_xyz, place_ori, 0.005)
            self.step_sim_and_render()
            self.gripper.check_finger_force()

        # release the gripper
        self.gripper.release()
        for _ in range(720):
            self.step_sim_and_render()

        # let the gripper retreat 3cm
        ori_matrix = trans.quaternion_matrix(place_ori)
        retreat_pos = lowdown_xyz - np.array(ori_matrix[:3, 2]) * 0.03
        dist_p, dist_r = 100, 100
        while dist_p > 0.01 or dist_r > 0.01:
            if self.check_limit():
                return done
            tip_state = pybullet.getLinkState(self.robot_id, self.tip_link_id)
            tip_xyz, tip_rot = np.float32(tip_state[0]), np.float32(tip_state[1])
            dist_p = np.linalg.norm(retreat_pos - tip_xyz)
            dist_r = 2 * np.arccos(abs(np.dot(place_ori, tip_rot)))
            self.move_pos_ori(retreat_pos, place_ori, 0.005)
            self.step_sim_and_render()

        observation = self.get_observation()
        reward = self.get_reward()
        done = True if not unreach_flag else False
        info = {}
        return done

    def step(self, action=None):
        """Do pick and place motion primitive."""
        pick_xyz, place_xyz = action["pick"].copy(), action["place"].copy()

        # Set fixed primitive z-heights.
        hover_xyz = pick_xyz.copy() + np.float32([0, 0, 0.2])
        pick_xyz[2] = 0.03
        place_xyz[2] = 0.15

        # Move to object.
        ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
        while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
            self.movep(hover_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
        while np.linalg.norm(pick_xyz - ee_xyz) > 0.01:
            self.movep(pick_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

        # Pick up object.
        self.gripper.activate()
        for _ in range(240):
            self.step_sim_and_render()
        while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
            self.movep(hover_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

        # Move to place location.
        while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
            self.movep(place_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

        # Place down object.
        while (not self.gripper.detect_contact()) and (place_xyz[2] > 0.03):
            place_xyz[2] -= 0.001
            self.movep(place_xyz)
            for _ in range(3):
                self.step_sim_and_render()
        self.gripper.release()
        for _ in range(240):
            self.step_sim_and_render()
        place_xyz[2] = 0.2
        ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
        while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
            self.movep(place_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
        place_xyz = np.float32([0, -0.5, 0.2])
        while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
            self.movep(place_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

        observation = self.get_observation()
        reward = self.get_reward()
        done = False
        info = {}
        return observation, reward, done, info

    def set_alpha_transparency(self, alpha: float) -> None:
        for id in range(20):
            visual_shape_data = pybullet.getVisualShapeData(id)
            for i in range(len(visual_shape_data)):
                object_id, link_index, _, _, _, _, _, rgba_color = visual_shape_data[i]
                rgba_color = list(rgba_color[0:3]) + [alpha]
                pybullet.changeVisualShape(
                    self.robot_id, linkIndex=i, rgbaColor=rgba_color)
                pybullet.changeVisualShape(
                    self.gripper.body, linkIndex=i, rgbaColor=rgba_color)

    def step_sim_and_render(self):
        pybullet.stepSimulation()
        self.sim_step += 1

        # Render current image at 8 FPS.
        if self.sim_step % 60 == 0:
            self.cache_video.append(self.image_for_video(self.cam_info))

    def image_for_video(self, cam_info):
        cam_info_video = copy.deepcopy(cam_info)
        cam_info_video['height'] = 720
        cam_info_video['width'] = 1280
        # Camera parameters.
        intrinsics = (cam_info_video['width']/2, 0, cam_info_video['width']/2, 0, cam_info_video['height']/2, cam_info_video['height']/2, 0, 0, 1)
        noise = True
        # OpenGL camera settings.
        focal_len = intrinsics[0]
        znear, zfar = (0.01, 10.)
        # change z and y
        cam_info_video['camera_look_at']['up'] = [cam_info_video['camera_look_at']['up'][0], cam_info_video['camera_look_at']['up'][1], -cam_info_video['camera_look_at']['up'][2]]

        render_eye = (cam_info_video['camera_look_at']['eye'][0], -cam_info_video['camera_look_at']['eye'][1]+0.2, cam_info_video['camera_look_at']['eye'][2])
        viewm = pybullet.computeViewMatrix(render_eye,
                                           tuple(cam_info_video['camera_look_at']['at']),
                                           tuple(cam_info_video['camera_look_at']['up']))
        fovh = (cam_info_video['height'] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = cam_info_video['width'] / cam_info_video['height']
        projm = pybullet.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, _, _ = pybullet.getCameraImage(
            width=cam_info_video['width'],
            height=cam_info_video['height'],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        # Get color image.
        color_image_size = (cam_info_video['height'], cam_info_video['width'], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[::-1, ::-1, :3]  # remove alpha channel
        if noise:
            color = np.int32(color)
            color += np.int32(np.random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        return color

    def get_camera_image(self):
        # image_size = (240, 240)
        # intrinsics = (120., 0, 120., 0, 120., 120., 0, 0, 1)
        color, _, _, _, _ = self.render_image(self.cam_info)
        return color

    def get_camera_image_top(self,
                             image_size=(240, 240),
                             intrinsics=(2000., 0, 2000., 0, 2000., 2000., 0, 0, 1),
                             position=(0, -0.5, 5),
                             orientation=(0, np.pi, -np.pi / 2),
                             zrange=(0.01, 1.),
                             set_alpha=True):
        set_alpha and self.set_alpha_transparency(0)
        color, _, _, _, _ = self.render_image_top(self.cam_info)
        set_alpha and self.set_alpha_transparency(1)
        return color

    def get_reward(self):
        return 0  # TODO: check did the robot follow text instructions?

    def get_current_pose(self, id):
        current_matrix = np.eye(4)
        current_pos, current_ori = pybullet.getBasePositionAndOrientation(id)
        current_rot_mat = pybullet.getMatrixFromQuaternion(current_ori)
        current_matrix[:3, :3] = np.reshape(current_rot_mat, (3, 3))
        current_matrix[:3, 3] = current_pos
        return current_matrix

    def get_observation(self, bounds=BOUNDS, pixel_size=PIXEL_SIZE):
        observation = {}

        # Render current image.
        color, depth, segm, cam_to_world, intrinsics = self.render_image(self.cam_info)
        segm[np.isin(segm, [self.pybullet_id_dict['robot'], self.pybullet_id_dict['ghost'],
                            self.pybullet_id_dict['gripper']])] = 0
        pose_dict = {}
        object_ids = np.unique(segm)[1:]
        for object_id in object_ids:
            position, orientation = pybullet.getBasePositionAndOrientation(object_id, 0)
            rotation = np.array(pybullet.getMatrixFromQuaternion(orientation)).reshape(3, 3)
            current_pose = np.eye(4)
            current_pose[:3, :3] = rotation
            current_pose[:3, 3] = position
            pose_dict[object_id] = current_pose
        observation["current_inertial_poses"] = pose_dict
        observation["vis"] = color
        observation["depth"] = depth
        observation["seg"] = segm
        observation["cam_K"] = intrinsics
        # Get heightmaps and colormaps.
        points = self.get_pointcloud(depth, intrinsics)
        observation["points_cam"] = points
        # rotate 180 along x_axis why?
        transform = np.stack((cam_to_world[:, 0], -cam_to_world[:, 1], -cam_to_world[:, 2], cam_to_world[:, 3]), axis=1)
        self.real_cam_to_world = transform

        points = self.transform_pointcloud(points, transform)

        reshaped_points = points.reshape(-1, 3)
        # 创建一个PointCloud对象
        pcd = o3d.geometry.PointCloud()
        # 将numpy数组转换为PointCloud
        pcd.points = o3d.utility.Vector3dVector(reshaped_points)
        # 创建一个坐标框
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0])
        # 可视化点云和坐标框
        # o3d.visualization.draw_geometries([pcd, coordinate_frame])
        heightmap, colormap, xyzmap = self.get_heightmap(points, color, bounds, pixel_size)

        observation["points_w"] = points
        observation["image"] = colormap
        observation["xyzmap"] = xyzmap
        return observation

    def render_image(self, cam_info):

        # Camera parameters.
        cam_to_world = np.array(cam_info['camera_local_to_world_matrix']).reshape(4, 4).T
        intrinsics = (cam_info['intrinsics']['fx'], 0, cam_info['intrinsics']['cx'], 0, cam_info['intrinsics']['fy'],
                      cam_info['intrinsics']['cy'], 0, 0, 1)
        zrange = (0.01, 10.)
        noise = True
        # OpenGL camera settings.
        focal_len = intrinsics[0]
        znear, zfar = (0.01, 10.)
        # change z and y
        cam_info['camera_look_at']['up'] = [cam_info['camera_look_at']['up'][0], cam_info['camera_look_at']['up'][2], cam_info['camera_look_at']['up'][1]]

        viewm = pybullet.computeViewMatrix(tuple(cam_info['camera_look_at']['eye']),
                                           tuple(cam_info['camera_look_at']['at']),
                                           tuple(cam_info['camera_look_at']['up']))
        fovh = (cam_info['height'] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = cam_info['width'] / cam_info['height']
        projm = pybullet.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = pybullet.getCameraImage(
            width=cam_info['width'],
            height=cam_info['height'],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        # Get color image.
        color_image_size = (cam_info['height'], cam_info['width'], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if noise:
            color = np.int32(color)
            color += np.int32(np.random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (cam_info['height'], cam_info['width'])
        zbuffer = np.float32(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2 * zbuffer - 1) * (zfar - znear))
        depth = (2 * znear * zfar) / depth
        if noise:
            depth += np.random.normal(0, 0.003, depth.shape)

        intrinsics = np.float32(intrinsics).reshape(3, 3)

        return color, depth, segm, cam_to_world, intrinsics

    def render_image_top(self, cam_info):

        position = (0, -0.5, 2),
        orientation = (0, np.pi, -np.pi / 2)
        intrinsics = (cam_info['intrinsics']['fx'], 0, cam_info['intrinsics']['cx'], 0, cam_info['intrinsics']['fy'],
                      cam_info['intrinsics']['cy'], 0, 0, 1)
        zrange = (0.01, 10.)

        # Camera parameters.
        orientation = pybullet.getQuaternionFromEuler(orientation)
        noise = True

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pybullet.getMatrixFromQuaternion(orientation)
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = position + lookdir
        focal_len = intrinsics[0]
        znear, zfar = (0.01, 10.)
        viewm = pybullet.computeViewMatrix(position, lookat, updir)
        fovh = (cam_info['height'] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = cam_info['width'] / cam_info['height']
        projm = pybullet.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = pybullet.getCameraImage(
            width=cam_info['width'],
            height=cam_info['height'],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        # Get color image.
        color_image_size = (cam_info['height'], cam_info['width'], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if noise:
            color = np.int32(color)
            color += np.int32(np.random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (cam_info['height'], cam_info['width'])
        zbuffer = np.float32(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2 * zbuffer - 1) * (zfar - znear))
        depth = (2 * znear * zfar) / depth
        if noise:
            depth += np.random.normal(0, 0.003, depth.shape)

        intrinsics = np.float32(intrinsics).reshape(3, 3)
        return color, depth, position, orientation, intrinsics

    def get_pointcloud(self, depth, intrinsics):
        """Get 3D pointcloud from perspective depth image.
    Args:
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.
    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """
        height, width = depth.shape
        xlin = np.linspace(0, width - 1, width)
        ylin = np.linspace(0, height - 1, height)
        px, py = np.meshgrid(xlin, ylin)
        px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
        py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
        points = np.float32([px, py, depth]).transpose(1, 2, 0)
        return points

    def transform_pointcloud(self, points, transform):
        """Apply rigid transformation to 3D pointcloud.
    Args:
      points: HxWx3 float array of 3D points in camera coordinates.
      transform: 4x4 float array representing a rigid transformation matrix.
    Returns:
      points: HxWx3 float array of transformed 3D points.
    """
        padding = ((0, 0), (0, 0), (0, 1))
        homogen_points = np.pad(points.copy(), padding,
                                "constant", constant_values=1)
        for i in range(3):
            points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
        return points

    def get_heightmap(self, points, colors, bounds, pixel_size):
        """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.
    Args:
      points: HxWx3 float array of 3D points in world coordinates.
      colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
      bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
        region in 3D space to generate heightmap in world coordinates.
      pixel_size: float defining size of each pixel in meters.
    Returns:
      heightmap: HxW float array of height (from lower z-bound) in meters.
      colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
      xyzmap: HxWx3 float array of XYZ points in world coordinates.
    """
        width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
        height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
        heightmap = np.zeros((height, width), dtype=np.float32)
        colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)
        xyzmap = np.zeros((height, width, 3), dtype=np.float32)

        # Filter out 3D points that are outside of the predefined bounds.
        ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
        iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
        iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
        valid = ix & iy & iz
        points = points[valid]
        colors = colors[valid]

        # Sort 3D points by z-value, which works with array assignment to simulate
        # z-buffering for rendering the heightmap image.
        iz = np.argsort(points[:, -1])
        points, colors = points[iz], colors[iz]
        px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
        py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)
        heightmap[py, px] = points[:, 2] - bounds[2, 0]
        for c in range(colors.shape[-1]):
            colormap[py, px, c] = colors[:, c]
            xyzmap[py, px, c] = points[:, c]
        colormap = colormap[::-1, :, :]  # Flip up-down.
        xv, yv = np.meshgrid(np.linspace(bounds[0, 0], bounds[0, 1], height),
                             np.linspace(bounds[1, 0], bounds[1, 1], width))
        xyzmap[:, :, 0] = xv
        xyzmap[:, :, 1] = yv
        xyzmap = xyzmap[::-1, :, :]  # Flip up-down.
        heightmap = heightmap[::-1, :]  # Flip up-down.
        return heightmap, colormap, xyzmap


def xyz_to_pix(position, bounds=BOUNDS, pixel_size=PIXEL_SIZE):
    """Convert from 3D position to pixel location on heightmap."""

    u = int(np.round((bounds[1, 1] - position[1]) / pixel_size))
    v = int(np.round((position[0] - bounds[0, 0]) / pixel_size))
    if u >= 224 or v >= 224:
        print(u, v)
    return (u, v)


## **Scripted Expert** Scripted pick and place oracle to collect expert demonstrations.
class ScriptedPolicy():

    def __init__(self, env):
        self.env = env

    def step(self, text, obs, place_sides=None):
        print(f'Input: {text}')
        self.place_sides = place_sides

        # Parse pick and place targets.
        pick_text, place_text = text.split('and', 1)
        pick_target, place_target = None, None
        for name in self.env.pick_targets.keys():
            if name in pick_text:
                pick_target = name
                break
        for name in self.env.place_targets.keys():
            if name in place_text:
                place_target = name
                break
        if self.place_sides != None:
            for name in self.place_sides.keys():
                if name in place_text:
                    place_side = name
                    break

        # Admissable targets only.
        assert pick_target is not None
        assert place_target is not None

        pick_id = self.env.obj_name_to_id[pick_target]
        pick_pose = pybullet.getBasePositionAndOrientation(pick_id)
        pick_position = np.float32(pick_pose[0])

        if place_target in self.env.obj_name_to_id:
            place_id = self.env.obj_name_to_id[place_target]
            place_pose = pybullet.getBasePositionAndOrientation(place_id)
            place_position = np.float32(place_pose[0])
            # 获取物体的 AABB
            aabb_min, aabb_max = pybullet.getAABB(place_id)
            xyz = np.array(aabb_max) - np.array(aabb_min)
        else:
            place_position = np.float32(self.env.place_targets[place_target])

        # Add some noise to pick and place positions.
        # pick_position[:2] += np.random.normal(scale=0.01)
        place_position[:2] += np.random.normal(scale=0.01)
        if self.place_sides != None:
            place_position[:2] += self.place_sides[place_side][:2]
            if place_side == "to the right of and in front of":
                place_position[1] -= xyz[1] / 2
                place_position[0] += xyz[0] / 2
            elif place_side == "to the left of and in front of":
                place_position[1] -= xyz[1] / 2
                place_position[0] -= xyz[0] / 2
            elif place_side == "to the right of and behind":
                place_position[1] += xyz[1] / 2
                place_position[0] += xyz[0] / 2
            elif place_side == "to the left of and behind":
                place_position[1] += xyz[1] / 2
                place_position[0] -= xyz[0] / 2

            elif place_side == 'in front of':
                place_position[1] -= xyz[1] / 2
            elif place_side == 'behind':
                place_position[1] += xyz[1] / 2

            elif place_side == 'to the right of':
                place_position[0] += xyz[0] / 2
            elif place_side == 'to the left of':
                place_position[0] -= xyz[0] / 2

        act = {'pick': pick_position, 'place': place_position}
        return act

## **Scripted Expert** Scripted pick and place oracle to collect expert demonstrations.
class ScriptedRearrangePolicy():

    def __init__(self, env):
        self.env = env

    def step(self, text, obs, place_sides=None):
        print(f'Input: {text}')
        self.place_sides = place_sides

        # Parse pick and place targets.
        pick_text, place_text = text.split('and', 1)
        pick_target, place_target = None, None
        for name in self.env.target_pose_dict.keys():
            if name.split('_')[0] in pick_text:
                pick_target = name
                break
        for name in self.env.target_pose_dict.keys():
            if name.split('_')[0] in place_text:
                place_target = name
                break
        if self.place_sides != None:
            for name in self.place_sides:
                if name in place_text:
                    place_side = name
                    break

        # Admissable targets only.
        assert pick_target is not None
        assert place_target is not None

        pick_id = self.env.pybullet_id_dict[pick_target]
        pick_pose = pybullet.getBasePositionAndOrientation(pick_id)
        pick_position = np.float32(pick_pose[0])


        place_position = np.float32(self.env.target_pose_dict[pick_target][:3,3]) # the same object in the target position

        # Add some noise to pick and place positions.
        # pick_position[:2] += np.random.normal(scale=0.01)
        place_position[:2] += np.random.normal(scale=0.01)
        act = {'pick': pick_position, 'place': place_position}
        return act



def main():
    x = RearrangeEnv()
    robotiq_gripper = x.gripper  # 假设你已经初始化了robot和tool
    list_joint_names(robotiq_gripper.body)

if __name__ == "__main__":
    main()