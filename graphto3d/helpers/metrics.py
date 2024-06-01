#!/usr/bin/python

import numpy as np
import torch
from scipy.spatial import ConvexHull
from helpers.util import denormalize_box_params
from helpers.util import fit_shapes_to_box
from cmath import rect, phase


def angular_distance(a, b):
    a %= 360.
    b %= 360.

    va = np.matmul(rot2d(a), [1, 0])
    vb = np.matmul(rot2d(b), [1, 0])
    return anglebetween2vecs(va, vb) % 360.


def anglebetween2vecs(va, vb):
    rad = np.arccos(np.clip(np.dot(va, vb), -1, 1))
    return np.rad2deg(rad)


def rot2d(degrees):
    rad = np.deg2rad(degrees)
    return np.asarray([[np.cos(rad), -np.sin(rad)],
                       [np.sin(rad), np.cos(rad)]])


def estimate_angular_mean(deg):
    return np.rad2deg(phase(np.sum(rect(1, np.deg2rad(d)) for d in deg)/len(deg))) % 360.


def estimate_angular_std(degs):
    m = estimate_angular_mean(degs)
    std = np.sqrt(np.sum([angular_distance(d, m)**2 for d in degs]) / len(degs))
    return std


def denormalize(box_params, with_norm=True):
    if with_norm:
        return denormalize_box_params(box_params, params=box_params.shape[0])
    else:
        return box_params

def validate_constrains(triples, pred_boxes, gt_boxes, keep, vocab, accuracy, with_norm=True,
                        strict=True, overlap_threshold=0.5):

    param6 = pred_boxes.shape[1] == 6
    layout_boxes = pred_boxes

    for [s, p, o] in triples:
        if keep is None:
            box_s = denormalize(layout_boxes[s.item()].cpu().detach().numpy(), with_norm)
            box_o = denormalize(layout_boxes[o.item()].cpu().detach().numpy(), with_norm)
        else:
            if keep[s.item()] == 1 and keep[o.item()] == 1: # if both are unchanged we evaluate the normal constraints
                box_s = denormalize(layout_boxes[s.item()].cpu().detach().numpy(), with_norm)
                box_o = denormalize(layout_boxes[o.item()].cpu().detach().numpy(), with_norm)
            else:
                continue
        if vocab["pred_idx_to_name"][p.item()][:-1] == "left":
            if check_if_has_rel(box_s, box_o, type="left"):
                accuracy['left'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['left'].append(0)
                accuracy['total'].append(0)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "right":
            if check_if_has_rel(box_s, box_o, type="right"):
                accuracy['right'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['right'].append(0)
                accuracy['total'].append(0)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "front":
            if check_if_has_rel(box_s, box_o, type="front"):
                accuracy['front'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['front'].append(0)
                accuracy['total'].append(0)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "behind":
            if check_if_has_rel(box_s, box_o, type="behind"):
                accuracy['behind'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['behind'].append(0)
                accuracy['total'].append(0)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "close by":
            if check_if_has_rel(box_s, box_o, type="close by"):
                accuracy['close by'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['close by'].append(0)
                accuracy['total'].append(0)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "symmetrical to":
            if check_if_has_rel(box_s, box_o, type="symmetrical to"):
                accuracy['symmetrical to'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['symmetrical to'].append(0)
                accuracy['total'].append(0)
    return accuracy


def validate_constrains_changes(triples, pred_boxes, gt_boxes, keep, vocab, accuracy, with_norm=True,
                                strict=True, overlap_threshold=0.5):

    layout_boxes = pred_boxes

    for [s, p, o] in triples:
        if keep is None:
            box_s = denormalize(layout_boxes[s.item()].cpu().detach().numpy(), with_norm)#! 主体 和 客体
            box_o = denormalize(layout_boxes[o.item()].cpu().detach().numpy(), with_norm)
        else:
            if keep[s.item()] == 0 or keep[o.item()] == 0: # if any node is change we evaluate the changes
                box_s = denormalize(layout_boxes[s.item()].cpu().detach().numpy(), with_norm)
                box_o = denormalize(layout_boxes[o.item()].cpu().detach().numpy(), with_norm)
            else:
                continue
        
        if vocab["pred_idx_to_name"][p.item()][:-1] == "left":
            if check_if_has_rel(box_s, box_o, type="left"):
                accuracy['left'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['left'].append(0)
                accuracy['total'].append(0)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "right":
            if check_if_has_rel(box_s, box_o, type="right"):
                accuracy['right'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['right'].append(0)
                accuracy['total'].append(0)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "front":
            if check_if_has_rel(box_s, box_o, type="front"):
                accuracy['front'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['front'].append(0)
                accuracy['total'].append(0)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "behind":
            if check_if_has_rel(box_s, box_o, type="behind"):
                accuracy['behind'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['behind'].append(0)
                accuracy['total'].append(0)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "close by":
            if check_if_has_rel(box_s, box_o, type="close by"):
                accuracy['close by'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['close by'].append(0)
                accuracy['total'].append(0)
        if vocab["pred_idx_to_name"][p.item()][:-1] == "symmetrical to":
            if check_if_has_rel(box_s, box_o, type="symmetrical to"):
                accuracy['symmetrical to'].append(1)
                accuracy['total'].append(1)
            else:
                accuracy['symmetrical to'].append(0)
                accuracy['total'].append(0)

    return accuracy


def corners_from_box(box, param6=True, with_translation=False):
    """
    按照一个预定义的顺序生成8个点的三维坐标（x_corners、y_corners、z_corners），并将它们存储在一个3x8的numpy数组corners_3d中。
    生成8个点的方法是：先按照边框的长、宽、高生成一个八面体，然后将其顶点沿着不同的坐标轴平移一定的距离，最终得到一个八个顶点的立方体。
    根据输入参数with_translation判断是否需要对坐标进行平移。如果需要平移，则将中心点坐标(tx, ty, tz)设置为(cx, cy, cz)，否则设置为(0, 0, 0)。

    根据边框的长l、宽w和高h生成8个顶点的x、y、z坐标，分别存储在x_corners、y_corners和z_corners数组中。这里顶点的顺序遵循一个预定义的规则，用于确保不同的边框生成的顶点顺序相同。
    将三个数组x_corners、y_corners、z_corners竖直堆叠起来，得到一个3x8的numpy数组corners_3d。
    根据需要进行平移操作，将corners_3d数组的每个元素的x、y、z坐标分别加上中心点坐标的tx、ty、tz值。具体来说，通过更改数组corners_3d中的第0、1、2行，实现对x、y、z坐标的平移。
    将corners_3d数组进行转置操作，得到8个点的坐标按照(x, y, z)的顺序排列，存储在一个8x3的numpy数组中，作为函数的返回值。
    """

    # box given as: [w, l, h, cx, cy, cz, z]
    if param6:
        l, w, h, cx, cy, cz = box
    else:
        l, w, h, cx, cy, cz, _ = box

    (tx, ty, tz) = (cx, cy, cz) if with_translation else (0,0,0)

    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d = np.dot(np.eye(3), np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + ty
    corners_3d[1,:] = corners_3d[1,:] + tz
    corners_3d[2,:] = corners_3d[2,:] + tx
    corners_3d = np.transpose(corners_3d)

    return corners_3d

def corners_from_box_up_z_axis(box):
    corner_offsets = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
            
    l,w,h,c_x,c_y,c_z = box
    half_l, half_w, half_h = l / 2, w / 2, h / 2

    corners = []
    for offset in corner_offsets:
        corner_x = c_x + offset[0] * half_l
        corner_y = c_y + offset[1] * half_w
        corner_z = c_z + offset[2] * half_h
        corners.append((corner_x, corner_y, corner_z))
    corners = np.array(corners)

    bbox_min = corners[0]
    bbox_max = corners[6]

    return corners
import math
def get_custom_radius_xy(min_corner,max_corner,aabb_center):
    min_x,min_y,min_z = min_corner
    max_x,max_y,max_z = max_corner
    corners = [(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]

    radius = 0
    for corner in corners:
        distance = math.sqrt((corner[0] - aabb_center[0])**2 + (corner[1] - aabb_center[1])**2)
        radius = max(radius, distance)
    return radius

def cal_l2_distance(point_1, point_2):
    return math.sqrt((point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2)

def check_if_has_rel(box1, box2, type, threshold_rad=0.05):
    #input of box: l, w, h, x, y, z
    corners1 = corners_from_box_up_z_axis(box1)
    corners2 = corners_from_box_up_z_axis(box2)

    if type=="left": #check if box1 - left - box2
        box1_center_x = box1[3]
        box2_min_x = corners2[0][0]
        return True if box1_center_x < box2_min_x else False
    elif type=="right":
        box1_center_x = box1[3]
        box2_max_x = corners2[6][0]
        return True if box1_center_x > box2_max_x else False
    elif type=="front":
        box1_center_y = box1[4]
        box2_min_y = corners2[0][1]
        return True if box1_center_y < box2_min_y else False
    elif type=="behind":
        box1_center_y = box1[4]
        box2_max_y = corners2[6][1]
        return True if box1_center_y > box2_max_y else False
    elif type=="close by":
        box1_center = box1[3:]
        box2_center = box2[3:]
        box2_min_corner = corners2[0]
        box2_max_corner = corners2[6]
        box2_custom_radius_xy = get_custom_radius_xy(box2_min_corner, box2_max_corner, box2_center)
        distance = cal_l2_distance(box1_center, box2_center)
        return True if distance <= box2_custom_radius_xy + threshold_rad else False
    elif type=="symmetrical to":  
        box1_center = box1[3:]
        box2_center = box2[3:]
        def round_float(num):
            return float("{:.1f}".format(num))

        return True if round_float(box1_center[0]) == round_float(box2_center[0]) or round_float(box1_center[1]) == round_float(box2_center[1]) else False 

def box3d_iou(box1, box2, param6=True, with_translation=False):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    '''
    # corner points are in counter clockwise order
    corners1 = corners_from_box(box1, param6, with_translation)
    corners2 = corners_from_box(box2, param6, with_translation)
    #print(f"shape of corners1 and corners2: {corners1.shape}, {corners2.shape}")

    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)] #两个边界框, 鸟瞰图
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] #

    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])

    inter, inter_area = convex_hull_intersection(rect1, rect2)
    #print(f"inter_area: {inter_area}, {area1}, {area2}")
    #这个警告通常是由于在计算IOU时遇到了除以0的情况或者计算出的IOU的值为NaN导致的。
    #可能是因为输入的边界框面积为0，或者两个边界框没有交集。建议在计算IOU前先检查输入的边界框是否存在问题，
    #例如是否有负值或零值的宽度、长度或高度。如果检查后仍无法解决问题，请检查IOU计算公式是否正确，或者使用其他的IOU计算方法来替代当前的计算方法。
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)

    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)

    volmin = min(vol1, vol2)

    iou = inter_vol / volmin #(vol1 + vol2 - inter_vol)

    return iou, iou_2d


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**
    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """
    def inside(p):
        return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

    def computeIntersection():
        dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
        dp = [ s[0] - e[0], s[1] - e[1] ]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return(outputList)


def pointcloud_overlap(pclouds, objs, boxes, angles, triples, vocab, overlap_metric):

    obj_classes = vocab['object_idx_to_name']
    pred_classes = vocab['pred_idx_to_name']
    pair = [(t[0].item(),t[2].item()) for t in triples]
    pred = [t[1].item() for t in triples]
    pair2pred = dict(zip(pair, pred))
    structural = ['floor', 'wall', 'ceiling', '_scene_']
    touching = ['none', 'inside', 'attached to', 'part of', 'cover', 'belonging to', 'build in', 'connected to']
    boxes = torch.cat([boxes.float(), angles.view(-1,1).float()], 1)

    for i in range(len(pclouds) - 1):
        for j in range(i+1, len(pclouds)):
            if obj_classes[objs[i]].split('\n')[0] in structural or \
                    obj_classes[objs[j]].split('\n')[0] in structural:
                # do not consider structural objects
                continue
            if (i, j) in pair2pred.keys() and pred_classes[pair2pred[(i,j)]].split('\n')[0] in touching:
                # naturally expected overlap
                continue
            if (j, i) in pair2pred.keys() and pred_classes[pair2pred[(j,i)]].split('\n')[0] in touching:
                # naturally expected overlap
                continue
            pc1 = fit_shapes_to_box(boxes[i].clone(), pclouds[i].clone())
            pc2 = fit_shapes_to_box(boxes[j].clone(), pclouds[j].clone())
            result = pointcloud_overlap_pair(pc1, pc2)
            overlap_metric.append(result)
    return overlap_metric


def pointcloud_overlap_pair(pc1, pc2):
    from sklearn.neighbors import NearestNeighbors
    all_pc = np.concatenate([pc1, pc2], 0)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree')
    nbrs.fit(all_pc)
    distances, indices = nbrs.kneighbors(pc1)
    # first neighbour will likely be itself other neighbour is a point from the same pc or the other pc
    # two point clouds are overlaping, when the nearest neighbours of one set are from the other set
    overlap = np.sum(indices >= len(pc1))
    return overlap
