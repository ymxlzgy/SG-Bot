'''
Author: Dianye Huang
Date: 2022-08-08 00:08:34
LastEditors: Dianye Huang
LastEditTime: 2022-08-08 00:36:28
Description: 
    This python file subscribe the point cloud data from 
    realsense camera and send return pc_full for grasp
    estimator.
    To test the code, run the following commands:
    $ cd ~/grasping_ws
    $ roscore
    $ rosbag -l guangyao.bag
    $ rviz
    Change the fixed Frame to be "camera_link"
    Add rostopic: /camera/depth/color/points
    $ rostopic info /camera/depth/color/points 
        --- Type: sensor_msgs/PointCloud2
    $ rosmsg show sensor_msgs/PointCloud2
    std_msgs/Header header
        uint32 seq
        time stamp
        string frame_id
    uint32 height
    uint32 width
    sensor_msgs/PointField[] fields
        uint8 INT8=1
        uint8 UINT8=2
        uint8 INT16=3
        uint8 UINT16=4
        uint8 INT32=5
        uint8 UINT32=6
        uint8 FLOAT32=7
        uint8 FLOAT64=8
        string name
        uint32 offset
        uint8 datatype
        uint32 count
    bool is_bigendian
    uint32 point_step
    uint32 row_step
    uint8[] data
    bool is_dense
Troube shooting: 
- 1. Not fixed 
QObject::moveToThread: Current thread (0x563f3f743df0) is not the object's thread (0x563f3f961a10).
Cannot move to target thread (0x563f3f743df0)
https://chowdera.com/2022/02/202202100555216474.html
https://stackoverflow.com/questions/46449850/how-to-fix-the-error-qobjectmovetothread-in-opencv-in-python
$ conda uninstall pyqt
$ pip install PyQt5
$ pip install opencv-python
'''
import struct
import ctypes
import numpy as np
import time
import cv2
# from cv_bridge import CvBridge, CvBridgeError

import rospy
from sensor_msgs.msg import (
    Image,
    PointCloud2
)
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
import tf.transformations as t
from sensor_msgs import (
    point_cloud2
)
import tf
from actionlib_msgs.msg import GoalStatusArray


class GraspNetRosInterface(object):
    def __init__(self, node_name = 'graspnet_rosinterface_node',
                    # sub_topic_name_pc = '/camera/depth/color/points',
                    # sub_topic_name_pc = '/depth_to_rgb/points', #'/kinect/depth/color/points',
                    sub_topic_name_pc = '/kinect/depth/color/points',
                    sub_topic_name_img = '/camera/color/image_raw',
                    # pub_topic_name_grasp = '/grasping_ws/grasp_coordinate',
                    pub_topic_name_grasp = '/grasping_ws/grasp_pose_candidates',
                    loop_hz = 30) -> None:
        '''
        Description: 
        @ param : node_name {string} -- ros node name to be initialized
        @ param : sub_topic_name {string} -- topic that publishes pointcloud2 data 
        @ param : loop_hz {float} -- ros rate
        @ return: None: 
        '''        
        rospy.init_node(node_name, anonymous=True)
        
        # an interface connected to grasp_gui
        self.param_trigger_grasp = '/grasp_gui/btn_grasp_get_clicked'
        self.param_update_grasp = '/grasp_gui/flag_update_grasp'
        rospy.set_param(self.param_trigger_grasp, False)
        rospy.set_param(self.param_update_grasp, False)
        
        self.rate = rospy.Rate(loop_hz)

        print('CaliSampler: waiting_for_message: move_group/status')
        rospy.wait_for_message('move_group/status', GoalStatusArray)
        print('CaliSampler: move_group/status okay!')
        self.tf_listener = tf.TransformListener()

        # point cloud and image subscription
        self.pc2_rosmsg = None
        self.pc2_gen = None
        self.pc_full = None
        self.pc_colors = None
        self.pc_info = None
        self.flag_msgparsing = False
        self.flag_msgupdated = False
        self.sub_pc2 = rospy.Subscriber(sub_topic_name_pc, PointCloud2, 
                            self.sub_pc2_cb, queue_size=1, tcp_nodelay=True)
        self.img = None
        # self.cv_bridge = CvBridge()
        # self.sub_img = rospy.Subscriber(sub_topic_name_img, Image, 
        #                     self.sub_img_cb, queue_size=1, tcp_nodelay=True)
        try:
            print('waiting for topic:', sub_topic_name_pc)
            rospy.wait_for_message(sub_topic_name_pc, PointCloud2, timeout=4.0)
            print('topic recevied!')
            # print('waiting for topic:', sub_topic_name_img)
            # rospy.wait_for_message(sub_topic_name_img, Image, timeout=4.0)
            # print('topic recevied!')
        except Exception as e:
            print(e)
        print('All topics recevied!')

        # grasping coordinate publish 
        # self.pub_pose = rospy.Publisher(pub_topic_name_grasp, PoseStamped, queue_size=1)
        self.pub_pose_candidate = rospy.Publisher(pub_topic_name_grasp, PoseArray, queue_size=1)

    def get_trigger_grasp_get(self):
        return rospy.get_param(self.param_trigger_grasp)

    def get_grasp_update(self):
        return rospy.get_param(self.param_update_grasp)
    
    def set_grasp_update(self, flag:bool):
        if flag:
            rospy.set_param(self.param_trigger_grasp, False)
        rospy.set_param(self.param_update_grasp, flag)
    

    def get_posemat_from_tf_tree(self, 
                            child_frame='panda_link8', 
                            parent_frame='panda_link0'):
        '''
        Description: 
            get translation and quaternion of child frame w.r.t. the parent frame
        @ param : child_frame{string} 
        @ param : parent_frame{string} 
        @ return: trans{list} -- x, y, z
        @ return: quat{list}: -- x, y, z, w
        '''    
        self.tf_listener.waitForTransform(parent_frame, 
                                        child_frame, 
                                        rospy.Time(), 
                                        rospy.Duration(2.0))
        (trans,quat) = self.tf_listener.lookupTransform(parent_frame,  # parent frame
                                                        child_frame,   # child frame
                                                        rospy.Time(0))
        posemat = t.quaternion_matrix(quat)
        posemat[:3, 3] = trans
        return posemat

    def get_tfmat_from_trans_quat(self, trans, quat):
        posemat = t.quaternion_matrix(quat)
        posemat[:3, 3] = trans
        return posemat 

    def get_posestamped_msg(self, tfmat, frame_id = "camera_color_optical_frame"):
        posestamped_msg = PoseStamped()
        posestamped_msg.header.frame_id = frame_id
        trans = t.translation_from_matrix(tfmat)
        quat = t.quaternion_from_matrix(tfmat)
        posestamped_msg.pose.position.x = trans[0]
        posestamped_msg.pose.position.y = trans[1]
        posestamped_msg.pose.position.z = trans[2]
        posestamped_msg.pose.orientation.x = quat[0]
        posestamped_msg.pose.orientation.y = quat[1]
        posestamped_msg.pose.orientation.z = quat[2]
        posestamped_msg.pose.orientation.w = quat[3]
        return posestamped_msg
    
    def pose_rotation_rectify(self, tfmat, rotx, roty, rotz):
        return np.dot(tfmat, t.euler_matrix(rotx, roty, rotz))
        

    # def pub_grasping_pose(self, tfmat, frame_id="camera_color_optical_frame"):
    #     msg = self.get_posestamped_msg(tfmat, frame_id)
    #     self.pub_pose.publish(msg)
    
    
    def pub_grasping_pose(self, tfmat, frame_id="panda_link0"):
        # msg = self.get_posestamped_msg(tfmat, frame_id)
        trans = t.translation_from_matrix(tfmat)
        quat = t.quaternion_from_matrix(tfmat)
        pose_list = [list(trans) + list(quat)]
        msg = self.get_posearray_msg(pose_list, frame_id)
        self.pub_pose_candidate.publish(msg)


    def get_posearray_msg(self, pose_list, frame_id = "panda_link0"):
        msg = PoseArray()
        msg.header.frame_id = frame_id
        for p in pose_list:
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            pose.position.z = p[2]
            pose.orientation.x = p[3]
            pose.orientation.y = p[4]
            pose.orientation.z = p[5]
            pose.orientation.w = p[6]
            msg.poses.append(pose)
        return msg


    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()
    
    def parsing_pc2(self, rosmsg:PointCloud2,
                    c2w=np.identity(4),
                    view_filter=1.0,
                    height_filter=0.0,
                    flag_parse_color = False,
                    x_lim=0.2):
        '''
        Description: 
            Parsing point cloud 2 data by recalling point_cloud2.readpoints() function
            To parse rgb info, bit operation on the 4th channel is needed
            -- for c++
            unsigned long rgb = *reinterpret_cast<int*>(&cloud->points[i].rgb);
            int r = (rgb >> 16) & 0x0000ff;
            int g = (rgb >> 8)  & 0x0000ff;
            int b = (rgb)       & 0x0000ff;
        @ param : rosmsg{sensor_msg::PointCloud2}
        @ param : flag_parse_color{bool} --default:False, --to parse color channel or not
        @ return: pc_full{np.array, dtype=float32, Nx3} stored x, y, z info
        @ return: pc_colors{np.array, dtype=uint8, Nx3} stored r, g, b info
        '''        
        pc2_gen = point_cloud2.read_points(rosmsg)
        pc_full = list()
        pc_colors = list()
        for p in pc2_gen:
            # constrain the camere view z-range
            if p[2] > view_filter:
                continue
            if abs(p[0]) > x_lim:  # x_filtered
                continue
            # constrain the height w.r.t the base frame
            p_vec = np.array([*list(p)[:3], 1])
            p_world = c2w.dot(p_vec)
            if p_world[2] < height_filter:
                continue
            pc_full.append(list(p)[:3])
            if flag_parse_color:
                # cast float32 to int so that bitwise operations are possible
                s = struct.pack('>f' ,p[3])
                i = struct.unpack('>l', s)[0]
                intb = ctypes.c_uint32(i).value
                r = (intb & 0x00FF0000)>> 16
                g = (intb & 0x0000FF00)>> 8
                b = (intb & 0x000000FF)
                # prints r,g,b values in the 0-255 range
                pc_colors.append([r, g, b])
        return np.array(pc_full, dtype=np.float32), \
                np.array(pc_colors, dtype=np.uint8), rosmsg

    def get_pc_info(self, c2w=np.identity(4), view_filter=1.0, height_filter=0.0, x_lim=0.2, flag_get_colors=False):
        '''
        Description: return pointcloud2 data
        @ param : flag_get_colors{bool} --default: False
        @ return: point cloud info,
        '''  
        if self.flag_msgupdated:
            self.flag_msgupdated = False
            self.flag_msgparsing = True
            time_start = time.time()
            self.pc_full, self.pc_colors, self.pc_info = self.parsing_pc2(self.pc2_rosmsg, 
                                                                        c2w=c2w,
                                                                        view_filter=view_filter,
                                                                        height_filter=height_filter,
                                                                        flag_parse_color=flag_get_colors,
                                                                        x_lim=0.2)
            print('Time collapse <parsing ros pc2>: {}s'.format(round(time.time()-time_start, 2)))
            self.flag_msgparsing = False
        return self.pc_full, self.pc_colors, self.pc_info
    
    def get_image(self):
        return None
        # return self.img

    def sub_pc2_cb(self, msg:PointCloud2):
        '''
        Description: 
            A callback function subscribing point cloud data
            check received info 
            print('-------')
            print('height:', msg.height) # if height == 1: unordered point cloud
            print('fields:', msg.fields)
            print('is_bigendian:', msg.is_bigendian)
            print('point_step:', msg.point_step)
            print('row_step:', msg.row_step)
            print('is_dense:', msg.is_dense)
        @ param : msg {PointCloud2} 
        @ return: None 
        '''      
        assert isinstance(msg, PointCloud2)
        if not self.flag_msgparsing:
            self.pc2_rosmsg = msg
            self.flag_msgupdated = True
    
    def sub_img_cb(self, msg:Image):
        '''
        Description: 
            A callback function subscribing raw rgb image
        @ param : msg {Image} 
        @ return: None 
        '''      
        assert isinstance(msg, Image)
        '''
        Description: 
            $ rosmsg show sensor_msgs/Image 
            std_msgs/Header header
                uint32 seq
                time stamp
                string frame_id
            uint32 height
            uint32 width
            string encoding
            uint8 is_bigendian
            uint32 step
            uint8[] data
        @ param : {}: 
        @ return: {}: 
        '''        
        if not self.flag_msgparsing:
            pass
            # self.img = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
            # self.img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")


if __name__ == '__main__':
    graspnet_ros = GraspNetRosInterface()
    while not rospy.is_shutdown():
        pc_full, pc_colors, _ = graspnet_ros.get_pc_info(flag_get_colors=False)
        img = graspnet_ros.get_image()
        # cv2.imshow("frame" , img)
        # cv2.waitKey(2)
        graspnet_ros.rate.sleep()

