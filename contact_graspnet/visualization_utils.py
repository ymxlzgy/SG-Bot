import cv2
import numpy as np
import matplotlib.pyplot as plt
import trimesh

# import mayavi.mlab as mlab
from matplotlib import cm

from .mesh_utils import create_gripper

def maximize_score_and_z_alignment(matrices, scores, openings=None, target_z=np.array([0,0,-1]), score_weight=0.5, alignment_weight=0.5, rel_pose=None, post=False):
    # 计算每个矩阵的z轴
    z_axes = matrices[:, :3, 2]

    # post is for cutlery.
    if rel_pose is not None and post==True:
        target_pose = np.matmul(rel_pose, matrices)
        z_axes = target_pose[:, :3, 2]

    # 计算每个矩阵z轴和目标z轴的对齐度
    alignments = np.dot(z_axes, target_z)

    # 计算每个矩阵的总得分
    total_scores = scores*score_weight + alignments*alignment_weight

    # 找到总得分最高的矩阵
    max_total_score_index = np.argmax(total_scores)
    # 返回总得分最高的矩阵
    return matrices[max_total_score_index], openings[max_total_score_index]

def create_panda_marker(color=[0, 0, 255], tube_radius=0.002, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=tube_radius, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    # z axis to x axis
    # R = np.array([[0,0,1],[1,0,0],[0,1,0]]).reshape(3,3)
    # t =  np.array([0, 0, -1.12169998e-01]).reshape(3,1)
    #
    # T = np.r_[np.c_[np.eye(3), t], [[0, 0, 0, 1]]]
    # tmp.apply_transform(T)

    return tmp

def points2pixels(points, cam_matrix):
    """

    Parameters
    ----------
    points N * 3
    cam_matrix

    Returns
    pixel_coordinates on image N * 2 (WxH)
    -------

    """
    points = points.reshape(-1, 3).transpose()
    points /= points[2]
    projection = cam_matrix.dot(points)
    pixel_coordinates = np.round(projection.T, 0)[:, :2].astype('int')
    return pixel_coordinates

def plot_grasp_img(img, segmap, grasps, scores, K, downsample_2D=False, downsample_size = 300, visual=False):
    grasps_show = 0
    grasps_total = 0
    for key in grasps.keys():
        T = grasps[key]
        if T.shape[0] == 0:
            continue
        grasps_total += len(T)
        if downsample_2D:
            np.random.seed(10)
            slices = np.random.choice(np.arange(T.shape[0]), size=downsample_size, replace=False) if T.shape[0] >= downsample_size else np.arange(T.shape[0])
            T = T[slices]
            grasps_show += len(slices)

        b1 = np.array([0, 0, 0, 1]).reshape(-1, 1)
        b2 = np.array([0, 0, 6.59999996e-02, 1]).reshape(-1, 1)
        lb = np.array([4.10000000e-02, -7.27595772e-12, 6.59999996e-02, 1]).reshape(-1, 1)
        lt = np.array([4.10000000e-02, -7.27595772e-12, 1.12169998e-01, 1]).reshape(-1, 1)
        rb = np.array([-4.100000e-02, -7.27595772e-12, 6.59999996e-02, 1]).reshape(-1, 1)
        rt = np.array([-4.100000e-02, -7.27595772e-12, 1.12169998e-01, 1]).reshape(-1, 1)
        gripper_points = np.c_[b1, b2, lb, lt, rb, rt]  # 4*6
        gripper_points_cam = np.matmul(T, gripper_points)  # N*4*6
        color = tuple(np.array([1-key/max(grasps.keys()), key/max(grasps.keys()), 0])*255)
        for i in range(T.shape[0]):
            pixels = points2pixels(gripper_points_cam[i, 0:3, :].T, K)  # 4*4
            cv2.line(img, pixels[0], pixels[1], color, 1)
            cv2.line(img, pixels[2], pixels[3], color, 1)
            cv2.line(img, pixels[4], pixels[5], color, 1)
            cv2.line(img, pixels[2], pixels[4], color, 1)
            cv2.line(img, pixels[4], pixels[5], color, 1)

        color_best = tuple(np.array([1-key/max(grasps.keys()), 0, key/max(grasps.keys())])*255)

        # best_grasp = maximize_score_and_z_alignment(T, scores[key], target_z=np.array([0,0,-1]), score_weight=0, alignment_weight=1)
        # best_gripper_points_cam = np.matmul(best_grasp, gripper_points)  # N*4*6
        # best_pixels = points2pixels(best_gripper_points_cam[0:3, :].T, K)  # 4*4

        best_gripper_points_cam = gripper_points_cam[np.argmax(scores[key])]
        print(f'best ind: {np.argmax(scores[key])}')
        best_pixels = points2pixels(best_gripper_points_cam[0:3, :].T, K)  # 4*4
        cv2.line(img, best_pixels[0], best_pixels[1], color_best, 2)
        cv2.line(img, best_pixels[2], best_pixels[3], color_best, 2)
        cv2.line(img, best_pixels[4], best_pixels[5], color_best, 2)
        cv2.line(img, best_pixels[2], best_pixels[4], color_best, 2)
        cv2.line(img, best_pixels[4], best_pixels[5], color_best, 2)
    # print("show ", grasps_show, " grasps out of totally ", grasps_total, "grasps.")
    if visual:
        cv2.imshow('2D', img[:,:,::-1])
        cv2.waitKey(0)
        cv2.destroyWindow('2D')
    if img.max() <= 1:
        img = (img * 255).astype(np.float32)
    show_image(img, segmap=None)

    return img

def plot_mesh(mesh, cam_trafo=np.eye(4), mesh_pose=np.eye(4)):
    """
    Plots mesh in mesh_pose from 

    Arguments:
        mesh {trimesh.base.Trimesh} -- input mesh, e.g. gripper

    Keyword Arguments:
        cam_trafo {np.ndarray} -- 4x4 transformation from world to camera coords (default: {np.eye(4)})
        mesh_pose {np.ndarray} -- 4x4 transformation from mesh to world coords (default: {np.eye(4)})
    """
    
    homog_mesh_vert = np.pad(mesh.vertices, (0, 1), 'constant', constant_values=(0, 1))
    mesh_cam = homog_mesh_vert.dot(mesh_pose.T).dot(cam_trafo.T)[:,:3]
    mlab.triangular_mesh(mesh_cam[:, 0],
                         mesh_cam[:, 1],
                         mesh_cam[:, 2],
                         mesh.faces,
                         colormap='Blues',
                         opacity=0.5)

def plot_coordinates(t,r, tube_radius=0.005):
    """
    plots coordinate frame

    Arguments:
        t {np.ndarray} -- translation vector
        r {np.ndarray} -- rotation matrix

    Keyword Arguments:
        tube_radius {float} -- radius of the plotted tubes (default: {0.005})
    """
    mlab.plot3d([t[0],t[0]+0.2*r[0,0]], [t[1],t[1]+0.2*r[1,0]], [t[2],t[2]+0.2*r[2,0]], color=(1,0,0), tube_radius=tube_radius, opacity=1)
    mlab.plot3d([t[0],t[0]+0.2*r[0,1]], [t[1],t[1]+0.2*r[1,1]], [t[2],t[2]+0.2*r[2,1]], color=(0,1,0), tube_radius=tube_radius, opacity=1)
    mlab.plot3d([t[0],t[0]+0.2*r[0,2]], [t[1],t[1]+0.2*r[1,2]], [t[2],t[2]+0.2*r[2,2]], color=(0,0,1), tube_radius=tube_radius, opacity=1)
                
def show_image(rgb, segmap=None):
    """
    Overlay rgb image with segmentation and imshow segment

    Arguments:
        rgb {np.ndarray} -- color image
        segmap {np.ndarray} -- integer segmap of same size as rgb
    """
    plt.figure()
    figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    
    plt.ion()
    plt.show()
    
    if rgb is not None:
        plt.imshow(rgb)
    if segmap is not None:
        cmap = plt.get_cmap('rainbow')
        cmap.set_under(alpha=0.0)   
        plt.imshow(segmap, cmap=cmap, alpha=0.5, vmin=0.0001)
    plt.draw()
    plt.pause(0.001)

def visualize_grasps(full_pc, pred_grasps_cam, scores, plot_opencv_cam=False, pc_colors=None, gripper_openings=None, gripper_width=0.08):
    """Visualizes colored point cloud and predicted grasps. If given, colors grasps by segmap regions. 
    Thick grasp is most confident per segment. For scene point cloud predictions, colors grasps according to confidence.

    Arguments:
        full_pc {np.ndarray} -- Nx3 point cloud of the scene
        pred_grasps_cam {dict[int:np.ndarray]} -- Predicted 4x4 grasp trafos per segment or for whole point cloud
        scores {dict[int:np.ndarray]} -- Confidence scores for grasps

    Keyword Arguments:
        plot_opencv_cam {bool} -- plot camera coordinate frame (default: {False})
        pc_colors {np.ndarray} -- Nx3 point cloud colors (default: {None})
        gripper_openings {dict[int:np.ndarray]} -- Predicted grasp widths (default: {None})
        gripper_width {float} -- If gripper_openings is None, plot grasp widths (default: {0.008})
    """

    print('Visualizing...takes time')
    cm = plt.get_cmap('rainbow')
    cm2 = plt.get_cmap('gist_rainbow')

    fig = mlab.figure('Pred Grasps')
    mlab.view(azimuth=180, elevation=180, distance=0.2)
    draw_pc_with_colors(full_pc, pc_colors)
    colors = [cm(1. * i/len(pred_grasps_cam))[:3] for i in range(len(pred_grasps_cam))]
    colors2 = {k:cm2(0.5*np.max(scores[k]))[:3] for k in pred_grasps_cam if np.any(pred_grasps_cam[k])}
    
    if plot_opencv_cam:
        plot_coordinates(np.zeros(3,),np.eye(3,3))
    for i,k in enumerate(pred_grasps_cam):
        if np.any(pred_grasps_cam[k]):
            gripper_openings_k = np.ones(len(pred_grasps_cam[k]))*gripper_width if gripper_openings is None else gripper_openings[k]
            if len(pred_grasps_cam) >= 1:
                draw_grasps(pred_grasps_cam[k], np.eye(4), color=colors[i], gripper_openings=gripper_openings_k)    
                draw_grasps([pred_grasps_cam[k][np.argmax(scores[k])]], np.eye(4), color=colors2[k], 
                            gripper_openings=[gripper_openings_k[np.argmax(scores[k])]], tube_radius=0.0025)    
            else:
                colors3 = [cm2(0.5*score)[:3] for score in scores[k]]
                draw_grasps(pred_grasps_cam[k], np.eye(4), colors=colors3, gripper_openings=gripper_openings_k)    
    mlab.show()

def draw_pc_with_colors(pc, pc_colors=None, single_color=(0.3,0.3,0.3), mode='2dsquare', scale_factor=0.0018):
    """
    Draws colored point clouds

    Arguments:
        pc {np.ndarray} -- Nx3 point cloud
        pc_colors {np.ndarray} -- Nx3 point cloud colors

    Keyword Arguments:
        single_color {tuple} -- single color for point cloud (default: {(0.3,0.3,0.3)})
        mode {str} -- primitive type to plot (default: {'point'})
        scale_factor {float} -- Scale of primitives. Does not work for points. (default: {0.002})

    """

    if pc_colors is None:
        mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=single_color, scale_factor=scale_factor, mode=mode)
    else:
        #create direct grid as 256**3 x 4 array 
        def create_8bit_rgb_lut():
            xl = np.mgrid[0:256, 0:256, 0:256]
            lut = np.vstack((xl[0].reshape(1, 256**3),
                                xl[1].reshape(1, 256**3),
                                xl[2].reshape(1, 256**3),
                                255 * np.ones((1, 256**3)))).T
            return lut.astype('int32')
        
        scalars = pc_colors[:,0]*256**2 + pc_colors[:,1]*256 + pc_colors[:,2]
        rgb_lut = create_8bit_rgb_lut()
        points_mlab = mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], scalars, mode=mode, scale_factor=.0018)
        points_mlab.glyph.scale_mode = 'scale_by_vector'
        points_mlab.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, rgb_lut.shape[0])
        points_mlab.module_manager.scalar_lut_manager.lut.number_of_colors = rgb_lut.shape[0]
        points_mlab.module_manager.scalar_lut_manager.lut.table = rgb_lut

def draw_grasps(grasps, cam_pose, gripper_openings, color=(0,1.,0), colors=None, show_gripper_mesh=False, tube_radius=0.0008):
    """
    Draws wireframe grasps from given camera pose and with given gripper openings

    Arguments:
        grasps {np.ndarray} -- Nx4x4 grasp pose transformations
        cam_pose {np.ndarray} -- 4x4 camera pose transformation
        gripper_openings {np.ndarray} -- Nx1 gripper openings

    Keyword Arguments:
        color {tuple} -- color of all grasps (default: {(0,1.,0)})
        colors {np.ndarray} -- Nx3 color of each grasp (default: {None})
        tube_radius {float} -- Radius of the grasp wireframes (default: {0.0008})
        show_gripper_mesh {bool} -- Renders the gripper mesh for one of the grasp poses (default: {False})
    """

    gripper = create_gripper('panda')
    gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3], 
                                gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])

    if show_gripper_mesh and len(grasps) > 0:
        plot_mesh(gripper.hand, cam_pose, grasps[0])
        
    all_pts = []
    connections = []
    index = 0
    N = 7
    for i,(g,g_opening) in enumerate(zip(grasps, gripper_openings)):
        gripper_control_points_closed = grasp_line_plot.copy()
        gripper_control_points_closed[2:,0] = np.sign(grasp_line_plot[2:,0]) * g_opening/2
        
        pts = np.matmul(gripper_control_points_closed, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)
        pts_homog = np.concatenate((pts, np.ones((7, 1))),axis=1)
        pts = np.dot(pts_homog, cam_pose.T)[:,:3]
        
        color = color if colors is None else colors[i]
        
        all_pts.append(pts)
        connections.append(np.vstack([np.arange(index,   index + N - 1.5),
                                      np.arange(index + 1, index + N - .5)]).T)
        index += N
        # mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2], color=color, tube_radius=tube_radius, opacity=1.0)
    
    # speeds up plot3d because only one vtk object
    all_pts = np.vstack(all_pts)
    connections = np.vstack(connections)
    # src = mlab.pipeline.scalar_scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2])
    # src.mlab_source.dataset.lines = connections
    # src.update()
    # lines =mlab.pipeline.tube(src, tube_radius=tube_radius, tube_sides=12)
    # mlab.pipeline.surface(lines, color=color, opacity=1.0)
    
