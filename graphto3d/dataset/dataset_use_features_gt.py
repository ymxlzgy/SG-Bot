from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import copy
import graphto3d.dataset.util as util
from tqdm import tqdm
import json
from graphto3d.helpers.psutil import FreeMemLinux
from graphto3d.helpers.util import normalize_box_params
import random
import pickle

class RIODatasetSceneGraph(data.Dataset):
    def __init__(self, root, root_raw,
                 label_file, npoints=2500, 
                 split='train', shuffle_objs=False,
                 use_points=False, use_scene_rels=True, with_changes=True, vae_baseline=False,
                 scale_func='diag', eval=False, eval_type='addition',
                 atlas=None, atlas2=None, path2atlas2=None, path2atlas=None, with_feats=False,
                 seed=True, recompute_feats=False, features_gt=None):

        self.seed = seed
        self.with_feats = with_feats
        self.atlas = atlas
        self.path2atlas = path2atlas
        self.atlas2 = atlas2
        self.path2atlas2 = path2atlas2
        self.recompute_feats = recompute_feats

        if eval and seed:
            np.random.seed(47)
            torch.manual_seed(47)
            random.seed(47)

        path2atlas, path2atlas_trained_model = os.path.split(self.path2atlas)
        path2atlas2, path2atlas2_trained_model = os.path.split(self.path2atlas2)
        self.features_gt_json_path = os.path.join(path2atlas, features_gt)
  
        self.scale_func = scale_func
        self.with_changes = with_changes
        self.npoints = npoints
        self.use_points = use_points
        self.root = root
        # list of class categories
        self.catfile = os.path.join(self.root, 'classes.txt')
        self.cat = {}
        self.scans = []
        self.vae_baseline = vae_baseline
        self.use_scene_rels = use_scene_rels

        self.fm = FreeMemLinux('GB')
        self.vocab = {}
        with open(os.path.join(self.root, 'classes.txt'), "r") as f:#! 包含所有的class, 包括_scene_
            self.vocab['object_idx_to_name'] = f.readlines()
        with open(os.path.join(self.root, 'relationships.txt'), "r") as f:#! 包含所有的relationship, 包括none
            self.vocab['pred_idx_to_name'] = f.readlines()
        #! {'object_idx_to_name': ['_scene_\n', 'fork\n', 'knife\n', 'tablespoon\n', 'teaspoon\n', 'plate\n', 'bowl\n', 'cup\n', 'teapot\n', 'pitcher\n', 'can\n', 'box\n', 'obstacle\n', 'support_table'], 
        #! 'pred_idx_to_name': ['none\n', 'left\n', 'right\n', 'front\n', 'behind\n', 'close by\n', 'standing on\n', 'lying on\n', 'lying in\n', 'symmetrical to']}

        splitfile = os.path.join(self.root, '{}.txt'.format(split))

        filelist = open(splitfile, "r").read().splitlines()
        self.filelist = [file.rstrip() for file in filelist] #!包含对应type(train/validation)的所有scene_names
        # list of relationship categories
        self.relationships = self.read_relationships(os.path.join(self.root, 'relationships.txt'))#!所有的relationship名称, 包括none

        # uses scene sections of up to 9 objects (from 3DSSG) if true, and full scenes otherwise
        if split == 'train_scenes': # training set
            splits_fname = 'relationships_train'
            self.rel_json_file = os.path.join(self.root, '{}.json'.format(splits_fname))
            self.box_json_file = os.path.join(self.root, 'obj_boxes_train.json')

        else: # validation set
            splits_fname = 'relationships_validation'
            self.rel_json_file = os.path.join(self.root, '{}.json'.format(splits_fname))
            self.box_json_file = os.path.join(self.root, 'obj_boxes_validation.json')

        self.relationship_json, self.objs_json, self.tight_boxes_json = \
                self.read_relationship_json(self.rel_json_file, self.box_json_file)

        self.label_file = label_file

        self.padding = 0.2
        self.eval = eval

        self.shuffle_objs = shuffle_objs

        self.root_raw = root_raw
        if self.root_raw == '':
            self.root_raw = os.path.join(self.root, "raw")

        with open(self.catfile, 'r') as f:
            for line in f:
                category = line.rstrip()
                self.cat[category] = category
        #! cat: {'_scene_': '_scene_', 'bowl': 'bowl', 'box': 'box', 'can': 'can', 'cup': 'cup', 'fork': 'fork', 'knife': 'knife', 'pitcher': 'pitcher', 'plate': 'plate', 'support_table': 'support_table', 'tablespoon': 'tablespoon', 'teapot': 'teapot', 'teaspoon': 'teaspoon', 'obstacle': 'obstacle'}
        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        #! classes: {'_scene_': 0, 'bowl': 1, 'box': 2, 'can': 3, 'cup': 4, 'fork': 5, 'knife': 6, 'obstacle': 7, 'pitcher': 8, 'plate': 9, 'support_table': 10, 'tablespoon': 11, 'teapot': 12, 'teaspoon': 13}
        
        points_classes = ['bowl','box', 'can', 'cup', 'fork', 'knife', 'pitcher', 'plate', 'support_table', 'tablespoon', 'teapot', 'teaspoon']

        points_classes_idx = []
        for pc in points_classes:
            points_classes_idx.append(self.classes[pc])#! 所选的特定的id
        
        self.point_classes_idx = points_classes_idx + [0] #! [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 0]

        self.sorted_cat_list =  sorted(self.cat) 
        self.files = {}
        self.eval_type = eval_type

        # check if all shape features exist. If not they get generated here (once)
        if with_feats:
            print('Checking for missing feats. This can be slow the first time.\nThis process needs to be only run once!')
            for index in tqdm(range(len(self))):
                self.__getitem__(index)
            self.recompute_feats = False

    def read_relationship_json(self, json_file, box_json_file):
        """ Reads from json files the relationship labels, objects and bounding boxes

        :param json_file: file that stores the objects and relationships
        :param box_json_file: file that stores the oriented 3D bounding box parameters
        :return: three dicts, relationships, objects and boxes
        """
        rel = {}
        objs = {}
        tight_boxes = {}

        with open(box_json_file, "r") as read_file:
            box_data = json.load(read_file)

        with open(json_file, "r") as read_file:
            data = json.load(read_file)
            for scene in data['scenes']:

                relationships = []
                for realationship in scene["relationships"]:
                    realationship[2] -= 1 
                    relationships.append(realationship)

                # for every scene in rel json, we append the scene id
                rel[scene["scene_id"]] = relationships
                self.scans.append(scene["scene_id"]) #mid/goal只有一个视角

                objects = {}
                boxes = {}
                for k, v in scene["objects"].items():
                    objects[int(k)] = v
                    try:
                        boxes[int(k)] = {}
                        boxes[int(k)]['param6'] = box_data[scene["scene_id"]][k]["param6"]
                    except:
                        # probably box was not saved because there were 0 points in the instance!
                        continue
                objs[scene["scene_id"]] = objects
                tight_boxes[scene["scene_id"]] = boxes
        return rel, objs, tight_boxes

    def read_relationships(self, read_file):
        """load list of relationship labels

        :param read_file: path of relationship list txt file
        """
        relationships = []
        with open(read_file, 'r') as f:
            for line in f:
                relationship = line.rstrip().lower()
                relationships.append(relationship)
        return relationships

    def __getitem__(self, index):
        scene_id = self.scans[index]

        file = os.path.join(self.root_raw, scene_id.split('_')[0], scene_id+self.label_file)
        if not os.path.exists(file):
            raise FileNotFoundError(f"Cannot find {scene_id}.obj file. Path: {file}")

        #!labelName2InstanceId, e.g. {'plate_6': 10, '2362ec480b3e9baa4fd5721982c508ad_support_table': 109, 'fork_1': 1, 'knife_1': 2}
        #!instanceId2class {10: 'plate', 109: 'support_table', 1: 'fork', 2: 'knife'}
        labelName2InstanceId, instanceId2class = util.get_label_name_to_global_id(file)
        selected_instances = list(self.objs_json[scene_id].keys())#! all instance_ids 与下行的keys相同 [10,109,1,2]
        keys = list(instanceId2class.keys()) #! all instance_ids

        if self.shuffle_objs:
            random.shuffle(keys) #![1,2,109,10]
        feats_in = None
        shape_priors_in = None
        # If true, expected paths to saved atlasnet features will be set here
        if self.with_feats and self.path2atlas is not None:
            _, atlasname = os.path.split(self.path2atlas)
            atlasname = atlasname.split('.')[0]

            feats_path = os.path.join(self.root_raw, scene_id.split('_')[0], '{}_{}_{}.pkl'.format(atlasname,'features',scene_id))
        # Load points if with features but features cannot be found or are forced to be recomputed
        # Loads points if use_points is set to true
        if (self.with_feats and (not os.path.exists(feats_path) or self.recompute_feats)) or self.use_points:
            if file in self.files: # Caching
                (points, instances) = self.files[file]
            else:#! 获得所有的点 和 点所属的instances_id, 检查是否对准
                atlas2folder, atlas2name = os.path.split(self.path2atlas2)
                atlas2folder = os.path.dirname(os.path.dirname(atlas2folder))
                points, instances = util.get_partial_point_cloud_of_initial_from_goal(file, labelName2InstanceId, atlas2folder)#! points: 所有的vertices, instances: 对应的instance_id(global_id)
                #! 这里的points已经归一化
                if self.fm.user_free > 5:
                    self.files[file] = (points, instances)

        instance2mask = {}
        instance2mask[0] = 0 

        cat = [] #! 存储当前scene, 存在的class_id [5, 6, 10, 9]
        tight_boxes = []

        counter = 0

        instances_order = []#! 添加instance_id#! [1,2,109,10] 与key顺序相同 instanceId2class {10: 'plate', 109: 'support_table', 1: 'fork', 2: 'knife'}
        selected_shapes = []

        #!self.classes: {'_scene_': 0, 'bowl': 1, 'box': 2, 'can': 3, 'cup': 4, 'fork': 5, 'knife': 6, 'obstacle': 7, 'pitcher': 8, 'plate': 9, 'support_table': 10, 'tablespoon': 11, 'teapot': 12, 'teaspoon': 13}
        
        #! instance2mask {0: 0, 1: 1, 2: 2, 109: 3, 10: 4}
        for key in keys: #! 所有的instance_ids e.g., [1,2,109,10]
            # get objects from the selected list of classes of 3dssg
            scene_instance_id = key #!1
            scene_instance_class = instanceId2class[key] #!fork
            scene_class_id = -1
            if scene_instance_class in self.classes:
                scene_class_id = self.classes[scene_instance_class] #!fork的clas_id -> 5
            else:
                raise ValueError(f"{scene_instance_class} must be in {self.classes}")
            
            if scene_class_id != -1 and key in selected_instances: #selected_instances与keys相同, 所以一定进入
                instance2mask[scene_instance_id] = counter + 1 #
                counter += 1
            else:#! scene_class_id == -1 -> 不存在self.classes中
                #print(key, selected_instances)
                instance2mask[scene_instance_id] = 0
                raise ValueError("scene_class_id should not be -1.")
            
            # mask to cat:
            if (scene_class_id >= 0) and (scene_instance_id > 0) and (key in selected_instances):#! 一定进入

                cat.append(scene_class_id)
                bbox = self.tight_boxes_json[scene_id][key]['param6'].copy()

                instances_order.append(key)
                if not self.vae_baseline:#!network_type==shared!=sln则进入
                    bbox = normalize_box_params(bbox)
                tight_boxes.append(bbox)#! 按照instances_order的顺序
            else:
                raise ValueError(f'{scene_class_id}<0? or {scene_instance_id}<=0?')

        if self.with_feats:
            # If precomputed features exist, we simply load them
            if os.path.exists(feats_path):
                feats_dic = pickle.load(open(feats_path, 'rb'))

                feats_in = feats_dic['feats']
                shape_priors_in = feats_dic['shape_priors']
                feats_order = np.asarray(feats_dic['instance_order'])
                ordered_feats = []
                ordered_shape_priors = []
                for inst in instances_order:
                    feats_in_instance = inst == feats_order
                    ordered_feats.append(feats_in[:-1][feats_in_instance]) 
                    ordered_shape_priors.append(shape_priors_in[:-1][feats_in_instance])
                ordered_feats.append(np.zeros([1, feats_in.shape[1]]))
                ordered_shape_priors.append(np.zeros([1, shape_priors_in.shape[1]]))
                feats_in = list(np.concatenate(ordered_feats, axis=0))
                shape_priors_in = list(np.concatenate(ordered_shape_priors, axis=0))

        # Sampling of points from object if they are loaded
        #! instance2mask {0: 0, 61: 1, 75: 2, 12: 3, 118: 4, 41: 5} instance_id:instance在场景中的编号/索引
        if (self.with_feats and (not os.path.exists(feats_path) or feats_in is None or shape_priors_in is None)) or self.use_points:
            #!为了shape_prior做准备
            masks = np.array(list(map(lambda l: instance2mask[l] if l in instance2mask.keys() else 0, instances)),
                             dtype=np.int32) #! instanceid 映射到 maskid
            #!  masks - [3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 4 4 4 4 4 4 4 4 4 4]
            num_pointsets = len(cat) + int(self.use_scene_rels)  # add zeros for the scene node
            obj_points = torch.zeros([num_pointsets, self.npoints, 3])
            
            assert(len(cat)==len(np.unique(masks)))
            
            #初始场景有些物体被遮挡住
            for i in range(len(cat)):#! 当前scene中存在的class_id [5, 6, 10, 9]
                obj_pointset = points[np.where(masks == i + 1)[0], :] #! 根据mask_id获取对应class的point

                if len(obj_pointset) <= 0:
                    assert ValueError("len of obj_pointset should not be 0")

                if len(obj_pointset) >= self.npoints:
                    choice = np.random.choice(len(obj_pointset), self.npoints, replace=False)
                else:
                    choice = np.arange(len(obj_pointset))
                    # use repetitions to fill some more points
                    choice2 = np.random.choice(len(obj_pointset), self.npoints - choice.shape[0], replace=True)
                    choice = np.concatenate([choice, choice2], 0)
                    random.shuffle(choice)
                    raise ValueError("len(obj_pointset) should not smaller than npoints")

                obj_pointset = obj_pointset[choice, :]
                obj_pointset = torch.from_numpy(obj_pointset.astype(np.float32))

                obj_points[i] = obj_pointset 
            
            #! 为了feats做准备
            instanceId2LabelName = {v: k for k, v in labelName2InstanceId.items()}
            #! {10: 'plate_6', 109: '2362ec480b3e9baa4fd5721982c508ad_support_table', 1: 'fork_1', 2: 'knife_1'}
            with open(self.features_gt_json_path) as json_file:
                objs_features_gt_dict = json.load(json_file)
            objs_features_gt = []

            if_rotate_z_90_degree = False
            if_rotate_z_minus_180_degree = True
            keywords = ['plate', 'spoon']
            keep_cup_pose_keywords = ['plate', 'cup']
            if all( any(key in value for value in instanceId2LabelName.values()) for key in keywords):
                if_rotate_z_90_degree = True
            if all( any(key in value for value in instanceId2LabelName.values()) for key in keep_cup_pose_keywords) \
                and all(['bowl' not in value for value in instanceId2LabelName.values()]):#只有plate+cup, 没有bowl -> 不转
                if_rotate_z_minus_180_degree = False

            for instance_id in instances_order:
                labelname = instanceId2LabelName[instance_id]
                if 'support_table' in labelname:
                    labelname = labelname.split('_')[0]
                if 'spoon' in labelname and if_rotate_z_90_degree:
                    labelname = labelname+'_rotate_z_90'
                if 'cup' in labelname and if_rotate_z_minus_180_degree and 'mid' not in scene_id:
                    labelname = labelname+'_rotate_z_minus_180'
                if 'cup' in labelname and 'mid' in scene_id:
                    labelname = instanceId2LabelName[instance_id]
                objs_features_gt.append(objs_features_gt_dict[labelname][0])
        else:
            obj_points = None

        triples = []
        rel_json = self.relationship_json[scene_id]

        for r in rel_json: # create relationship triplets from data
            if r[0] in instance2mask.keys() and r[1] in instance2mask.keys():  #r[0], r[1] -> instance_id
                subject = instance2mask[r[0]]-1 #! instance2mask[r[0] -> 实例在场景中的编号/索引 - 1, 最后一个node '_scene_' 放最后
                object = instance2mask[r[1]]-1
                predicate = r[2] +1
                if subject >= 0 and object >= 0:
                    triples.append([subject, predicate, object])
            else:
                continue     
        if self.use_scene_rels:
            # add _scene_ object and _in_scene_ connections
            scene_idx = len(cat) #4  
            for i, ob in enumerate(cat): #!当前scene中的class_id(从0开始), 都和scene_idx(_scene_ object),有predicate-0: _in_scene_
                triples.append([i, 0, scene_idx])  #_in_scene_ 关系是从0开始的
            cat.append(0)
            # dummy scene box
            tight_boxes.append([-1, -1, -1, -1, -1, -1])
        output = {}

        # if features are requested but the files don't exist, we run all loaded pointclouds through atlasnet
        # to compute them and then save them for future usage
        if self.with_feats and (not os.path.exists(feats_path) or feats_in is None or shape_priors_in is None) and (self.atlas is not None) and (self.atlas2 is not None):
            #! 获得shape_prior -> 从atlasnet2得到的
            pf = torch.from_numpy(np.array(list(obj_points.numpy()), dtype=np.float32)).float().cuda().transpose(1,2)
            with torch.no_grad():
                shape_priors = self.atlas2.encoder(pf).detach().cpu().numpy()

            #! 获得feats -> 从atlasnet1得到的
            feats = np.array(objs_features_gt)
            feats_single = torch.zeros(1, 128)#添加最后一个(__node__)的feat
            feats = np.vstack((feats, feats_single))#(num_obj+1,128)
            
            feats_out = {}
            feats_out['feats'] = feats
            feats_out['shape_priors'] = shape_priors #! 存储为cache
            feats_out['instance_order'] = instances_order#! instances_id
            assert(shape_priors.shape == shape_priors.shape)
            feats_in = list(feats)
            shape_priors_in = list(shape_priors)

            assert self.path2atlas2 is not None
            assert self.path2atlas is not None
            path = os.path.join(feats_path)

            pickle.dump(feats_out, open(path, 'wb'))
        # prepare outputs
        output['encoder'] = {}
        output['encoder']['objs'] = cat#! 当前scene所有的class_id + _scene_(0)
        output['encoder']['triples'] = triples
        output['encoder']['boxes'] = tight_boxes

        if self.with_feats:
            output['encoder']['feats'] = feats_in
            output['encoder']['shape_priors'] = shape_priors_in

        output['manipulate'] = {}
        if not self.with_changes:
            output['manipulate']['type'] = 'none'
            output['decoder'] = copy.deepcopy(output['encoder'])
        else:#! 进入
            if not self.eval:#! eval=False进入
                if self.with_changes:
                    output['manipulate']['type'] = ['relationship', 'addition', 'none'][np.random.randint(3)]  # removal is trivial - so only addition and rel change
                else:
                    output['manipulate']['type'] = 'none'
                output['decoder'] = copy.deepcopy(output['encoder'])
                if output['manipulate']['type'] == 'addition':
                    node_id,pop_node_id_value = self.remove_node_and_relationship(output['encoder'])
                    if node_id >= 0:
                        output['manipulate']['added'] = node_id
                        output['manipulate']['added_shape_prior']  = pop_node_id_value

                    else:
                        output['manipulate']['type'] = 'none'
                elif output['manipulate']['type'] == 'relationship':
                    rel, pair, suc = self.modify_relship(output['decoder'], interpretable=True)
                    if suc:
                        output['manipulate']['relship'] = (rel, pair)
                    else:
                        output['manipulate']['type'] = 'none'
            else:
                output['manipulate']['type'] = self.eval_type
                output['decoder'] = copy.deepcopy(output['encoder'])
                if output['manipulate']['type'] == 'addition':
                    node_id,pop_node_id_value = self.remove_node_and_relationship(output['encoder'])
                    if node_id >= 0:
                        output['manipulate']['added'] = node_id
                        output['manipulate']['added_shape_prior']  = pop_node_id_value

                    else:
                        return -1
                elif output['manipulate']['type'] == 'relationship':
                    rel, pair, suc = self.modify_relship(output['decoder'], interpretable=True)
                    if suc:
                        output['manipulate']['relship'] = (rel, pair)
                    else:
                        return -1
        # torchify
        output['encoder']['objs'] = torch.from_numpy(np.array(output['encoder']['objs'], dtype=np.int64))
        output['encoder']['triples'] = torch.from_numpy(np.array(output['encoder']['triples'], dtype=np.int64))
        output['encoder']['boxes'] = torch.from_numpy(np.array(output['encoder']['boxes'], dtype=np.float32))
        if self.use_points:#!X
            output['encoder']['points'] = torch.from_numpy(np.array(output['encoder']['points'], dtype=np.float32))
        if self.with_feats:#!进入
            output['encoder']['feats'] = torch.from_numpy(np.array(output['encoder']['feats'], dtype=np.float32))
            output['encoder']['shape_priors'] = torch.from_numpy(np.array(output['encoder']['shape_priors'], dtype=np.float32))

        output['decoder']['objs'] = torch.from_numpy(np.array(output['decoder']['objs'], dtype=np.int64))
        output['decoder']['triples'] = torch.from_numpy(np.array(output['decoder']['triples'], dtype=np.int64))
        output['decoder']['boxes'] = torch.from_numpy(np.array(output['decoder']['boxes'], dtype=np.float32))
        if self.use_points:#!X
            output['decoder']['points'] = torch.from_numpy(np.array(output['decoder']['points'], dtype=np.float32))
        if self.with_feats:#!进入
            output['decoder']['feats'] = torch.from_numpy(np.array(output['decoder']['feats'], dtype=np.float32))
            output['decoder']['shape_priors'] = torch.from_numpy(np.array(output['decoder']['shape_priors'], dtype=np.float32))

        output['scene_id'] = scene_id
        output['instance_id'] = instances_order

        return output


    def remove_node_and_relationship(self, graph):
        """ Automatic random removal of certain nodes at training time to enable training with changes. In that case
        also the connecting relationships of that node are removed

        :param graph: dict containing objects, features, boxes, points and relationship triplets
        :return: index of the removed node
        具体实现是在函数中随机选择一个节点，并检查是否为特定的一些节点（在excluded列表中定义），
        如果是则重新选择。然后从图中删除该节点以及与它相连的关系，以及与节点相关的信息，如包围盒、特征和点云。
        最后，更新剩余节点的编号以确保它们在图中的顺序正确。函数返回被删除节点的索引。
        这个函数的目的是模拟模型在遇到图中物体被遮挡、丢失或者错误的情况下的行为，从而提高模型的鲁棒性和泛化能力。
        """
        node_id = -1
        # dont remove layout components, like floor. those are essential -> #? 在我们的情况应该是support_table
        excluded = [13]
        trials = 0
        while node_id < 0 or graph['objs'][node_id] in excluded:
            if trials > 100:
                return -1
            trials += 1
            node_id = np.random.randint(len(graph['objs']) - 1)#不包含最后一个0

        graph['objs'].pop(node_id) #node_id 在场景中的编号/索引，不是Global_id
        if self.use_points:
            graph['points'].pop(node_id)
        if self.with_feats:
            graph['feats'].pop(node_id)
            pop_node_id_value = graph['shape_priors'].pop(node_id)
            assert(len(pop_node_id_value.shape)==1)
        graph['boxes'].pop(node_id)

        to_rm = []
        for x in graph['triples']:
            sub, pred, obj = x
            if sub == node_id or obj == node_id:
                to_rm.append(x)

        while len(to_rm) > 0:
            graph['triples'].remove(to_rm.pop(0))

        for i in range(len(graph['triples'])):
            if graph['triples'][i][0] > node_id:
                graph['triples'][i][0] -= 1

            if graph['triples'][i][2] > node_id:
                graph['triples'][i][2] -= 1

        return node_id,pop_node_id_value

    def modify_relship(self, graph, interpretable=True):
        """ Change a relationship type in a graph

        :param graph: dict containing objects, features, boxes, points and relationship triplets
        :param interpretable: boolean, if true choose a subset of easy to interpret relations for the changes
        :return: index of changed triplet, a tuple of affected subject & object, and a boolean indicating if change happened
        """
        interpretable_rels = [1,2,3,4] 

        did_change = False
        trials = 0
        excluded = [13]#! 修改 不动桌子

        while not did_change and trials < 1000:
            idx = np.random.randint(len(graph['triples']))
            sub, pred, obj = graph['triples'][idx]
            trials += 1

            if pred == 0:
                continue
            if graph['objs'][obj] in excluded or graph['objs'][sub] in excluded:
                continue
            new_pred = interpretable_rels[np.random.randint(len(interpretable_rels))]

            graph['triples'][idx][1] = new_pred
            did_change = True
        return idx, (sub, obj), did_change

    def __len__(self):
        return len(self.scans)


def collate_fn_vaegan(batch, use_points=False):
    """
    Collate function to be used when wrapping a RIODatasetSceneGraph in a
    DataLoader. Returns a dictionary
    """

    out = {}

    out['scene_points'] = []
    out['scene_id'] = []
    out['instance_id'] = []

    out['missing_nodes'] = []
    out['missing_nodes_decoder'] = []
    out['manipulated_nodes'] = []
    out['missing_nodes_shape_priors'] = []
    global_node_id = 0
    global_dec_id = 0

    for i in range(len(batch)):
        if batch[i] == -1:
            return -1
        # notice only works with single batches
        out['scene_id'].append(batch[i]['scene_id'])
        out['instance_id'].append(batch[i]['instance_id'])

        if batch[i]['manipulate']['type'] == 'addition':
            out['missing_nodes'].append(global_node_id + batch[i]['manipulate']['added'])#! output['manipulate']['added'] = node_id
            out['missing_nodes_decoder'].append(global_dec_id + batch[i]['manipulate']['added'])
            out['missing_nodes_shape_priors'].append(batch[i]['manipulate']['added_shape_prior'])

        elif batch[i]['manipulate']['type'] == 'relationship':
            rel, (sub, obj) = batch[i]['manipulate']['relship']
            out['manipulated_nodes'].append(global_dec_id + sub)
            out['manipulated_nodes'].append(global_dec_id + obj)

        if 'scene' in batch[i]:
            out['scene_points'].append(batch[i]['scene'])

        global_node_id += len(batch[i]['encoder']['objs'])
        global_dec_id += len(batch[i]['decoder']['objs'])

    for key in ['encoder', 'decoder']: 
        all_objs, all_boxes, all_triples = [], [], []
        all_obj_to_scene, all_triple_to_scene = [], []
        all_points = []
        all_feats = []
        all_shape_priors = []

        obj_offset = 0

        for i in range(len(batch)):
            if batch[i] == -1:
                print('this should not happen')
                continue
            (objs, triples, boxes) = batch[i][key]['objs'], batch[i][key]['triples'], batch[i][key]['boxes']

            if 'points' in batch[i][key]:
                all_points.append(batch[i][key]['points'])
            if 'feats' in batch[i][key]:
                all_feats.append(batch[i][key]['feats'])
            if 'shape_priors' in batch[i][key]:
                all_shape_priors.append(batch[i][key]['shape_priors'])
            else:
                raise ValueError(f"should exist shape_priors in {key}")

            num_objs, num_triples = objs.size(0), triples.size(0)

            all_objs.append(objs)
            all_boxes.append(boxes)

            if triples.dim() > 1:
                triples = triples.clone()
                triples[:, 0] += obj_offset
                triples[:, 2] += obj_offset

                all_triples.append(triples)#! 每个关系三元组所属的场景索引。
                all_triple_to_scene.append(torch.LongTensor(num_triples).fill_(i))#! 创建一个torch.LongTensor对象，其长度等于num_triples，并将所有元素填充为整数i。

            all_obj_to_scene.append(torch.LongTensor(num_objs).fill_(i))#!每个对象所属的场景索引。

            obj_offset += num_objs

        all_objs = torch.cat(all_objs)
        all_boxes = torch.cat(all_boxes)

        all_obj_to_scene = torch.cat(all_obj_to_scene)

        if len(all_triples) > 0:
            all_triples = torch.cat(all_triples)
            all_triple_to_scene = torch.cat(all_triple_to_scene)
        else:
            return -1

        outputs = {'objs': all_objs,
                   'triples': all_triples,
                   'boxes': all_boxes,
                   'obj_to_scene': all_obj_to_scene,
                   'tiple_to_scene': all_triple_to_scene}

        if len(all_points) > 0:
            all_points = torch.cat(all_points)
            outputs['points'] = all_points

        if len(all_feats) > 0:
            all_feats = torch.cat(all_feats)
            outputs['feats'] = all_feats
        
        if len(all_shape_priors) > 0:
            all_shape_priors = torch.cat(all_shape_priors)
            outputs['shape_priors'] = all_shape_priors
            
        out[key] = outputs

    return out


def collate_fn_vaegan_points(batch):
    """ Wrapper of the function collate_fn_vaegan to make it also return points
    """
    return collate_fn_vaegan(batch, use_points=True)
