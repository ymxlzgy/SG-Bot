import torch
import numpy as np


def get_cross_prod_mat(pVec_Arr):
    """ Convert pVec_Arr of shape (3) to its cross product matrix
    """
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def params_to_8points(box, degrees=False):
    """ Given bounding box as 7 parameters: w, l, h, cx, cy, cz, z, compute the 8 corners of the box
    """
    w, l, h, cx, cy, cz, z = box
    points = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                points.append([w.item()/2 * i, l.item()/2 * j, h.item()/2 * k])
    points = np.asarray(points)
    points = (get_rotation(z.item(), degree=degrees) @ points.T).T
    points += np.expand_dims(np.array([cx.item(), cy.item(), cz.item()]), 0)
    return points


def params_to_8points_no_rot(box):
    """ Given bounding box as 6 parameters (without rotation): w, l, h, cx, cy, cz, compute the 8 corners of the box.
        Works when the box is axis aligned
    """
    """     w, l, h, cx, cy, cz = box
        points = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    points.append([w.item()/2 * i, l.item()/2 * j, h.item()/2 * k]) """
    corner_offsets = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    l,w,h, c_x,c_y,c_z = box
    half_l, half_w, half_h = l / 2, w / 2, h / 2

    corners = []

    for offset in corner_offsets:
        corner_x = c_x + offset[0] * half_l
        corner_y = c_y + offset[1] * half_w
        corner_z = c_z + offset[2] * half_h
        corners.append((corner_x, corner_y, corner_z))
    #corners = corners.cpu().detach().numpy()
    #corners = np.array(corners)

    #points = np.asarray(points)
    #points += np.expand_dims(np.array([cx.item(), cy.item(), cz.item()]), 0)
    corners = np.array([[x for x in row] for row in corners])
    return corners


def fit_shapes_to_box(box, shape, withangle=False):
    """ Given normalized shape, transform it to fit the input bounding box.
        Expects denormalized bounding box with optional angle channel in degrees
        :param box: tensor
        :param shape: tensor
        :param withangle: boolean
        :return: transformed shape
    #!目的是将给定的归一化形状 shape 转换为适应输入边界框 box(反归一化后) 的形状。
    """
    box = box.detach().cpu().numpy()#! 将输入张量 box 从计算图中分离（detach()），并将其从 GPU 上移到 CPU 上（如果可用），然后转换为 NumPy 数组。
    shape = shape.detach().cpu().numpy()#! 将输入张量 shape 分离并转换为 NumPy 数组。
    if withangle:
        l, w, h, cx, cy, cz, z = box
    else:
        l, w, h, cx, cy, cz = box
    # scale
    shape_size = np.max(shape, axis=0) - np.min(shape, axis=0)#! 计算输入形状 shape 的尺寸（即最大坐标值和最小坐标值之差）。
    shape = shape / shape_size#! 将输入形状 shape 除以其尺寸以归一化
    shape *= box[:3] #! 将归一化的形状 shape 乘以输入边界框 box 的前三个元素, 从而将 shape 缩放为适应 box 的大小。
    #if withangle:#!X
        # rotate
    #    shape = (get_rotation(z, degree=True).astype("float32") @ shape.T).T
    # translate
    shape += [cx, cy, cz]

    return shape

def fit_shapes_to_boxv2(box, shape, withangle=False):
    """ Given normalized shape, transform it to fit the input bounding box.
        Expects denormalized bounding box with optional angle channel in degrees
        :param box: tensor
        :param shape: tensor
        :param withangle: boolean
        :return: transformed shape
    #!目的是将给定的归一化形状 shape 转换为适应输入边界框 box(反归一化后) 的形状。
    """
    if isinstance(box, torch.Tensor):
        box = box.detach().cpu().numpy()#! 将输入张量 box 从计算图中分离（detach()），并将其从 GPU 上移到 CPU 上（如果可用），然后转换为 NumPy 数组。
    if isinstance(shape, torch.Tensor):
        shape = shape.detach().cpu().numpy()#! 将输入张量 shape 分离并转换为 NumPy 数组。
    if withangle:
        l, w, h, cx, cy, cz, z = box
    else:
        l, w, h, cx, cy, cz = box
    # scale
    shape_size = np.max(shape, axis=0) - np.min(shape, axis=0)#! 计算输入形状 shape 的尺寸（即最大坐标值和最小坐标值之差）。
    max_axis = np.argmax(shape_size)
    max_value = shape_size[max_axis]
    rate = box[max_axis] / max_value
    shape *= rate
    #if withangle:#!X
        # rotate
    #    shape = (get_rotation(z, degree=True).astype("float32") @ shape.T).T
    # translate
    shape += [cx, cy, cz]

    return shape

def refineBoxes(boxes, objs, triples, relationships, vocab):
    for idx in range(len(boxes)):
      child_box = boxes[idx]
      w, l, h, cx, cy, cz = child_box
      for t in triples:
         if idx == t[0] and relationships[t[1]] in ["supported by", "lying on", "standing on"]:
            parent_idx = t[2]
            cat = vocab['object_idx_to_name'][objs[parent_idx]].replace('\n', '')
            if cat != 'floor':
                continue
            parent_box = boxes[parent_idx]
            base = parent_box[5] + 0.0125

            new_bottom = base
            # new_h = cz + h / 2 - new_bottom
            new_cz = new_bottom + h / 2
            shift = new_cz - cz
            boxes[idx][:] = [w, l, h, cx, cy, new_cz]

            # fix adjusmets
            for t_ in triples:
                if t_[2] == t[0] and relationships[t_[1]] in ["supported by", "lying on", "standing on"]:
                    cat = vocab['object_idx_to_name'][t_[2]].replace('\n', '')
                    if cat != 'floor':
                        continue

                    w_, l_, h_, cx_, cy_, cz_ = boxes[t_[0]]
                    boxes[t_[0]][:] = [w_, l_, h_, cx_, cy_, cz_ + shift]
    return boxes


def get_rotation(z, degree=True):
    """ Get rotation matrix given rotation angle along the z axis.
    :param z: angle of z axos rotation
    :param degree: boolean, if true angle is given in degrees, else in radians
    :return: rotation matrix as np array of shape[3,3]
    """
    if degree:
        z = np.deg2rad(z)
    rot = np.array([[np.cos(z), -np.sin(z),  0],
                    [np.sin(z),  np.cos(z),  0],
                    [        0,          0,  1]])
    return rot


def normalize_box_params(box_params, scale=3):
    """ Normalize the box parameters for more stable learning utilizing the accumulated dataset statistics

    :param box_params: float array of shape [7] containing the box parameters
    :param scale: float scalar that scales the parameter distribution
    :return: normalized box parameters array of shape [7]
    """
    mean = np.array([ 0.2610482 ,  0.22473196,  0.14623462,  0.0010283 , -0.02288815 , 0.20876316])#np.array([ 2.42144732e-01,  2.35105852e-01,  1.53590141e-01, -1.54968627e-04, -2.68763962e-02,  2.23784580e-01 ])
    std = np.array([0.31285113, 0.21937416, 0.17070778, 0.14874465, 0.1200992,  0.11501499])#np.array([ 0.27346058, 0.23751527, 0.18529049, 0.12504842, 0.13313938 ,0.12407406 ])

    return scale * ((box_params - mean) / std)


def denormalize_box_params(box_params, scale=3, params=6):
    """ Denormalize the box parameters utilizing the accumulated dataset statistics

    :param box_params: float array of shape [params] containing the box parameters
    :param scale: float scalar that scales the parameter distribution
    :param params: number of bounding box parameters. Expects values of either 6 or 7. 6 omits the angle
    :return: denormalized box parameters array of shape [params]
    """
    if params == 6:
        mean = np.array([ 0.2610482 ,  0.22473196,  0.14623462,  0.0010283 , -0.02288815 , 0.20876316])#np.array([ 2.42144732e-01,  2.35105852e-01,  1.53590141e-01, -1.54968627e-04, -2.68763962e-02,  2.23784580e-01 ])
        std = np.array([0.31285113, 0.21937416, 0.17070778, 0.14874465, 0.1200992,  0.11501499])#np.array([ 0.27346058, 0.23751527, 0.18529049, 0.12504842, 0.13313938 ,0.12407406 ])
    else:
        raise NotImplementedError
    return (box_params * std) / scale + mean


def denormalize_initial_box_params(box_params, scale=3, params=6):
    """ Denormalize the box parameters utilizing the accumulated dataset statistics

    :param box_params: float array of shape [params] containing the box parameters
    :param scale: float scalar that scales the parameter distribution
    :param params: number of bounding box parameters. Expects values of either 6 or 7. 6 omits the angle
    :return: denormalized box parameters array of shape [params]
    """
    if params == 6:
        mean = np.array([ 0.2610482 ,  0.22473196,  0.14623462,  0.0010283 , -0.02288815 , 0.20876316])#np.array([ 2.42144732e-01,  2.35105852e-01,  1.53590141e-01, -1.54968627e-04, -2.68763962e-02,  2.23784580e-01 ])
        std = np.array([0.31285113, 0.21937416, 0.17070778, 0.14874465, 0.1200992,  0.11501499])#np.array([ 0.27346058, 0.23751527, 0.18529049, 0.12504842, 0.13313938 ,0.12407406 ])
    else:
        raise NotImplementedError
    return (box_params * std) / scale + mean


def batch_torch_denormalize_box_params(box_params, scale=3):
    """ Denormalize the box parameters utilizing the accumulated dateaset statistics

    :param box_params: float tensor of shape [N, 6] containing the 6 box parameters, where N is the number of boxes
    :param scale: float scalar that scales the parameter distribution
    :return: float tensor of shape [N, 6], the denormalized box parameters
    """

    mean = torch.tensor([ 0.2610482 ,  0.22473196,  0.14623462,  0.0010283 , -0.02288815 , 0.20876316]).reshape(1,-1).float().cuda()#torch.tensor([ 2.42144732e-01,  2.35105852e-01,  1.53590141e-01, -1.54968627e-04, -2.68763962e-02,  2.23784580e-01 ]).reshape(1,-1).float().cuda()
    std = torch.tensor([0.31285113, 0.21937416, 0.17070778, 0.14874465, 0.1200992,  0.11501499]).reshape(1,-1).float().cuda()#torch.tensor([ 0.27346058, 0.23751527, 0.18529049, 0.12504842, 0.13313938 ,0.12407406 ]).reshape(1,-1).float().cuda()

    return (box_params * std) / scale + mean


def bool_flag(s):
    """Helper function to make argparse work with the input True and False.
    Otherwise it reads it as a string and is always set to True.

    :param s: string input
    :return: the boolean equivalent of the string s
    """
    if s == '1' or s == 'True':
      return True
    elif s == '0' or s == 'False':
      return False
    msg = 'Invalid value "%s" for bool flag (should be 0, False or 1, True)'
    raise ValueError(msg % s)
