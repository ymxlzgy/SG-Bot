import numpy as np
import open3d as o3d
import seaborn as sns
import open3d.visualization.gui as gui
from graphto3d.helpers.util import fit_shapes_to_box, fit_shapes_to_boxv2, params_to_8points, params_to_8points_no_rot
from graphto3d.render.lineMesh import LineMesh


def render_ini_and_goal(goal_bboxes, goal_points, raw_points, ini_points, name_id_dict, cam_to_world, render_raw = False, render_shapes=True, render_boxes=True):
    color_palette = np.array(sns.color_palette('hls', len(list(goal_bboxes.keys()))))

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    ren_opt = vis.get_render_option()
    ren_opt.mesh_show_back_face = True
    ren_opt.line_width = 50.

    if render_raw:
        raw_points = raw_points.reshape(-1,3)
        raw_points = raw_points[np.where(raw_points[:,2]>0.01)]
        # raw_points = ini_points[np.random.choice(raw_points.shape[0], 50000, replace=False), :]
        raw_shape = o3d.geometry.PointCloud()
        raw_shape.points = o3d.utility.Vector3dVector(raw_points)
        raw_shape_colors = [[192/255, 192/255, 192/255] for _ in range(len(raw_points))]
        raw_shape.colors = o3d.utility.Vector3dVector(raw_shape_colors)
        vis.add_geometry(raw_shape)

    edges = [0, 1], [0, 4], [0, 3], [1, 2], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 7], [5, 6], [6, 7]

    for i, k in enumerate(goal_bboxes):

        goal_vertices = goal_points[k]
        box_vertices = params_to_8points_no_rot(goal_bboxes[k])

        goal_shape = o3d.geometry.PointCloud()
        goal_shape.points = o3d.utility.Vector3dVector(goal_vertices)
        goal_shape_colors = [color_palette[i % len(color_palette)] for _ in range(len(goal_vertices))]
        goal_shape.colors = o3d.utility.Vector3dVector(goal_shape_colors)

        if render_shapes:
            vis.add_geometry(goal_shape)
            vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]))

        if render_boxes:
            line_colors = [color_palette[i % len(color_palette)] for _ in range(len(edges))]
            line_mesh = LineMesh(box_vertices, edges, line_colors, radius=0.001)
            line_mesh_geoms = line_mesh.cylinder_segments

            for g in line_mesh_geoms:
                vis.add_geometry(g)

        ini_vertices = ini_points.pop(name_id_dict[k], None)
        if ini_vertices is not None:
            ini_vertices_w = np.matmul(cam_to_world[:3, :3], ini_vertices.T) + cam_to_world[:3, 3].reshape(3, 1)
            ini_shape = o3d.geometry.PointCloud()
            ini_shape.points = o3d.utility.Vector3dVector(ini_vertices_w.T)
            ini_shape_colors = [color_palette[i % len(color_palette)] * 0.5 for _ in range(len(ini_vertices_w.T))]
            ini_shape.colors = o3d.utility.Vector3dVector(ini_shape_colors)
            vis.add_geometry(ini_shape)

    for i,k in enumerate(ini_points):
        unexpected_vertices = ini_points[k]
        unexpected_vertices_w = np.matmul(cam_to_world[:3, :3], unexpected_vertices.T) + cam_to_world[:3, 3].reshape(3, 1)
        unexpected_shape = o3d.geometry.PointCloud()
        unexpected_shape.points = o3d.utility.Vector3dVector(unexpected_vertices_w.T)
        unexpected_shape_colors = [color_palette[i % len(color_palette)] * 0.5 for _ in range(len(unexpected_vertices_w.T))]
        unexpected_shape.colors = o3d.utility.Vector3dVector(unexpected_shape_colors)
        vis.add_geometry(unexpected_shape)

    # vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 2]))
    vis.poll_events()
    vis.run()
    vis.destroy_window()


def render_comparison(shapes_pred, shape_ini, color_pred=None, color_ini=None):
    """
    param predBoxes: denormalized bounding box param6
    param shapes_pred: predicted point clouds
    """
    color_pred = [1, 0, 0]
    color_ini = [0, 1, 0]

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ren_opt = vis.get_render_option()
    ren_opt.mesh_show_back_face = True
    ren_opt.line_width = 50.

    pcd_shape_ini = o3d.geometry.PointCloud()
    pcd_shape_ini.points = o3d.utility.Vector3dVector(shape_ini)
    pcd_shape_ini_colors = [color_ini for _ in range(len(shape_ini))]
    pcd_shape_ini.colors = o3d.utility.Vector3dVector(pcd_shape_ini_colors)
    vis.add_geometry(pcd_shape_ini)

    pcd_shape_pred = o3d.geometry.PointCloud()
    pcd_shape_pred.points = o3d.utility.Vector3dVector(shapes_pred)
    pcd_shape_pred_colors = [color_pred for _ in range(len(shapes_pred))]
    pcd_shape_pred.colors = o3d.utility.Vector3dVector(pcd_shape_pred_colors)
    vis.add_geometry(pcd_shape_pred)

    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]))
    vis.poll_events()
    vis.run()
    vis.destroy_window()


def render_per_shape(shapes_pred, colors=None):
    """
    param predBoxes: denormalized bounding box param6
    param shapes_pred: predicted point clouds
    """
    if colors is None:
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 默认颜色
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ren_opt = vis.get_render_option()
    ren_opt.mesh_show_back_face = True
    ren_opt.line_width = 50.
    denorm_shape = shapes_pred
    pcd_shape = o3d.geometry.PointCloud()
    pcd_shape.points = o3d.utility.Vector3dVector(denorm_shape)
    pcd_shape_colors = [colors for _ in range(len(denorm_shape))]
    pcd_shape.colors = o3d.utility.Vector3dVector(pcd_shape_colors)
    vis.add_geometry(pcd_shape)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]))
    vis.update_geometry(pcd_shape)
    vis.poll_events()
    vis.run()
    vis.destroy_window()

def render_result(shapes_pred, predBoxes=None, render_shapes=True, render_boxes=False, colors=None):
    """
    param predBoxes: denormalized bounding box param6
    param shapes_pred: predicted point clouds
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    ren_opt = vis.get_render_option()
    ren_opt.mesh_show_back_face = True
    ren_opt.line_width = 50.

    edges = [0, 1], [0, 4], [0, 3], [1, 2], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 7], [5, 6], [6, 7]

    for i in range(len(shapes_pred)):

        vertices = shapes_pred[i]

        pcd_shape = o3d.geometry.PointCloud()
        pcd_shape.points = o3d.utility.Vector3dVector(vertices)
        pcd_shape_colors = [colors[i % len(colors)] for _ in range(len(vertices))]
        pcd_shape.colors = o3d.utility.Vector3dVector(pcd_shape_colors)

        if render_shapes:
            vis.add_geometry(pcd_shape)
            # vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]))

        if render_boxes:
            box_vertices = params_to_8points_no_rot(predBoxes[i])
            line_colors = [colors[i % len(colors)] for _ in range(len(edges))]
            line_mesh = LineMesh(box_vertices, edges, line_colors, radius=0.001)
            line_mesh_geoms = line_mesh.cylinder_segments

            for g in line_mesh_geoms:
                vis.add_geometry(g)

    # vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 2]))
    vis.poll_events()
    vis.run()
    vis.destroy_window()

def render(predBoxes, shapes_pred, render_shapes=True, render_boxes=False, colors=None):
    """ 
    param predBoxes: denormalized bounding box param6
    param shapes_pred: predicted point clouds
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    ren_opt = vis.get_render_option()
    ren_opt.mesh_show_back_face = True
    ren_opt.line_width = 50.

    edges = [0, 1], [0, 4], [0, 3], [1, 2], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 7], [5, 6], [6, 7]

    for i in range(len(predBoxes) - 1):  # ! -1为了最后去掉的属于0的预测

        vertices = shapes_pred[i]  # ! 点云中的顶点
        box_vertices = params_to_8points_no_rot(predBoxes[i])
        denorm_shape = fit_shapes_to_boxv2(predBoxes[i], vertices, withangle=False)

        pcd_shape = o3d.geometry.PointCloud()
        pcd_shape.points = o3d.utility.Vector3dVector(denorm_shape)
        pcd_shape_colors = [colors[i % len(colors)] for _ in range(len(denorm_shape))]
        pcd_shape.colors = o3d.utility.Vector3dVector(pcd_shape_colors)

        if render_shapes:
            vis.add_geometry(pcd_shape)
            vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]))

        if render_boxes:
            line_colors = [colors[i % len(colors)] for _ in range(len(edges))]
            line_mesh = LineMesh(box_vertices, edges, line_colors, radius=0.001)
            line_mesh_geoms = line_mesh.cylinder_segments

            for g in line_mesh_geoms:
                vis.add_geometry(g)

    # vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 2]))
    vis.poll_events()
    vis.run()
    vis.destroy_window()

