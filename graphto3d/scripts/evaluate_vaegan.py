from __future__ import print_function
import open3d as o3d # open3d needs to be imported before other packages!
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import sys
# 获取当前文件的路径
current_path = os.path.abspath(__file__)  # .../scripts/evaluate_vaegan.py

# 获取上一级路径
parent_path = os.path.dirname(current_path)# .../scripts
sys.path.append(os.path.dirname(parent_path))  #.../graphto3d/

from graphto3d_2.model.VAE import VAE
from graphto3d_2.dataset.dataset_use_features_gt import RIODatasetSceneGraph, collate_fn_vaegan, collate_fn_vaegan_points
from graphto3d_2.helpers.util import bool_flag, batch_torch_denormalize_box_params
from graphto3d_2.helpers.metrics import validate_constrains, validate_constrains_changes
from graphto3d_2.helpers.visualize_graph import run as vis_graph
from graphto3d_2.helpers.visualize_scene import render, render_per_shape
from graphto3d_2.model.atlasnet import AE_AtlasNet

import graphto3d_2.extension.dist_chamfer as ext
chamfer = ext.chamferDist()
import json

parser = argparse.ArgumentParser()
parser.add_argument('--num_points', type=int, default=5625, help='number of points in the shape')

parser.add_argument('--dataset', required=False, type=str, default="/media/ymxlzgy/Data/Dataset/rearrange_dataset", help="dataset path")
parser.add_argument('--dataset_raw', type=str, default='/media/ymxlzgy/Data/Dataset/rearrange_dataset/raw', help="dataset path of raw dataset")
parser.add_argument('--label_file', required=False, type=str, default='.obj', help="label file name")

parser.add_argument('--with_points', type=bool_flag, default=False, help="if false, only predicts layout")
parser.add_argument('--with_feats', type=bool_flag, default=True, help="Load Feats directly instead of points.")

parser.add_argument('--manipulate', default=True, type=bool_flag)
parser.add_argument('--path2atlas', required=False, default="../../AtlasNet/log/atlasnet_separate_cultery/network2.pth", type=str)
parser.add_argument('--path2atlas2', required=False, default="../../AtlasNet2/log/AE_AtlasNet2_20230408T2110/atlasnet2.pth", type=str)
parser.add_argument('--objs_features_gt', default="objs_features_gt_atlasnet_separate_cultery.json", type=str)
parser.add_argument('--exp', default='/home/ymxlzgy/code/sgbot/graphto3d_2/experiments/test_shape_prior', help='experiment name')
parser.add_argument('--epoch', type=str, default='600', help='saved epoch')
parser.add_argument('--recompute_stats', type=bool_flag, default=False, help='Recomputes statistics of evaluated networks')
parser.add_argument('--visualize', default=True, type=bool_flag)
parser.add_argument('--export_3d', default=True, type=bool_flag, help='Export the generated shapes and boxes in json files for future use')
args = parser.parse_args()


def evaluate():
    print(torch.__version__)

    random.seed(48)
    torch.manual_seed(48)

    argsJson = os.path.join(args.exp, 'args.json')
    assert os.path.exists(argsJson), 'Could not find args.json for experiment {}'.format(args.exp)
    with open(argsJson) as j:
        modelArgs = json.load(j)

    saved_atlasnet_model = torch.load(args.path2atlas)
    point_ae = AE_AtlasNet(num_points=5625, bottleneck_size=128, nb_primitives=25)
    point_ae.load_state_dict(saved_atlasnet_model, strict=True)
    if torch.cuda.is_available():
        point_ae = point_ae.cuda()
    
    saved_atlasnet2_model = torch.load(args.path2atlas2)
    point_ae2 = AE_AtlasNet(num_points=5625, bottleneck_size=128, nb_primitives=25)
    point_ae2.load_state_dict(saved_atlasnet2_model, strict=True)
    if torch.cuda.is_available():
        point_ae2 = point_ae2.cuda()

    test_dataset_rels_changes = RIODatasetSceneGraph(
        root=args.dataset,
        atlas=point_ae,
        atlas2=point_ae2,
        path2atlas=args.path2atlas,
        path2atlas2=args.path2atlas2,
        root_raw=args.dataset_raw,
        label_file=args.label_file,
        split='validation_scenes',
        npoints=args.num_points,
        use_points=args.with_points,
        use_scene_rels=modelArgs['use_scene_rels'],
        vae_baseline=modelArgs['network_type']=='sln',
        with_changes=True,
        eval=True,
        eval_type='relationship',
        with_feats=args.with_feats,
        features_gt=args.objs_features_gt)

    test_dataset_addition_changes = RIODatasetSceneGraph(
        root=args.dataset,
        atlas=point_ae,
        atlas2=point_ae2,
        path2atlas=args.path2atlas,
        path2atlas2=args.path2atlas2,
        root_raw=args.dataset_raw,
        label_file=args.label_file,
        split='validation_scenes',
        npoints=args.num_points,
        use_points=args.with_points,
        use_scene_rels=modelArgs['use_scene_rels'],
        vae_baseline=modelArgs['network_type']=='sln',
        with_changes=True,
        eval=True,
        eval_type='addition',
        with_feats=args.with_feats,
        features_gt=args.objs_features_gt)

    # used to collect train statistics
    stats_dataset = RIODatasetSceneGraph(
        root=args.dataset,
        atlas=point_ae,
        atlas2=point_ae2,
        path2atlas=args.path2atlas,
        path2atlas2=args.path2atlas2,
        root_raw=args.dataset_raw,
        label_file=args.label_file,
        npoints=args.num_points,
        split='train_scenes',
        use_points=args.with_points,
        use_scene_rels=modelArgs['use_scene_rels'],
        with_changes=False,
        vae_baseline=modelArgs['network_type']=='sln',
        eval=False,
        with_feats=args.with_feats,
        features_gt=args.objs_features_gt)

    test_dataset_no_changes = RIODatasetSceneGraph(
        root=args.dataset,
        atlas=point_ae,
        atlas2=point_ae2,
        path2atlas=args.path2atlas,
        path2atlas2=args.path2atlas2,
        root_raw=args.dataset_raw,
        label_file=args.label_file,
        split='validation_scenes',
        npoints=args.num_points,
        use_points=args.with_points,
        use_scene_rels=modelArgs['use_scene_rels'],
        vae_baseline=modelArgs['network_type']=='sln',
        with_changes=False,
        eval=True,
        with_feats=args.with_feats,
        features_gt=args.objs_features_gt)

    if args.with_points:
        collate_fn = collate_fn_vaegan_points
    else:
        collate_fn = collate_fn_vaegan

    # test_dataloader_rels_changes = torch.utils.data.DataLoader(
    #     test_dataset_rels_changes,
    #     batch_size=1,
    #     collate_fn=collate_fn,
    #     shuffle=False,
    #     num_workers=0)
    #
    # test_dataloader_add_changes = torch.utils.data.DataLoader(
    #     test_dataset_addition_changes,
    #     batch_size=1,
    #     collate_fn=collate_fn,
    #     shuffle=False,
    #     num_workers=0)

    # dataloader to collect train data statistics
    stats_dataloader = torch.utils.data.DataLoader(
        stats_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0)

    test_dataloader_no_changes = torch.utils.data.DataLoader(
        test_dataset_no_changes,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0)

    modeltype_ = modelArgs['network_type']
    replacelatent_ = modelArgs['replace_latent'] if 'replace_latent' in modelArgs else None
    with_changes_ = modelArgs['with_changes'] if 'with_changes' in modelArgs else None

    model = VAE(type=modeltype_, vocab=test_dataset_no_changes.vocab, replace_latent=replacelatent_,
                with_changes=with_changes_, residual=modelArgs['residual'])
    model.load_networks(exp=args.exp, epoch=args.epoch)
    if torch.cuda.is_available():
        model = model.cuda()

    model = model.eval()
    point_ae = point_ae.eval()

    model.compute_statistics(exp=args.exp, epoch=args.epoch, stats_dataloader=stats_dataloader, force=args.recompute_stats)

    def reseed():
        np.random.seed(47)
        torch.manual_seed(47)
        random.seed(47)

    # print('\nEditing Mode - Additions')
    # reseed()
    # validate_constrains_loop_w_changes(test_dataloader_add_changes, model, atlas=point_ae)
    # reseed()
    # print('\nEditing Mode - Relationship changes')
    # validate_constrains_loop_w_changes(test_dataloader_rels_changes, model, atlas=point_ae)

    reseed()
    print('\nGeneration Mode')
    validate_constrains_loop(test_dataloader_no_changes, model, point_ae=point_ae, export_3d=args.export_3d)


def validate_constrains_loop_w_changes(testdataloader, model, atlas):
    accuracy = {}
    accuracy_unchanged = {}
    accuracy_in_orig_graph = {}

    for k in ['left', 'right', 'front', 'behind', 'close by', 'symmetrical to', 'total']:
        accuracy_in_orig_graph[k] = []
        accuracy_unchanged[k] = []
        accuracy[k] = []

    for i, data in enumerate(testdataloader, 0):
        try:
            enc_objs, enc_triples, enc_tight_boxes, enc_objs_to_scene, enc_triples_to_scene, enc_shape_priros = data['encoder']['objs'], \
                                                                                              data['encoder']['triples'], \
                                                                                              data['encoder']['boxes'], \
                                                                                              data['encoder']['obj_to_scene'], \
                                                                                              data['encoder'][ 'tiple_to_scene'], \
                                                                                              data['encoder']['shape_priors']
            if 'feats' in data['encoder']:
                encoded_enc_points = data['encoder']['feats']
                encoded_enc_points = encoded_enc_points.float().cuda()
                encoded_dec_shape_priors = data['decoder']['shape_priors']
                encoded_dec_shape_priors = encoded_dec_shape_priors.cuda()
            if 'points' in data['encoder']:
                enc_points = data['encoder']['points']
                enc_points = enc_points.cuda()
                with torch.no_grad():
                    encoded_enc_points = atlas.encoder(enc_points.transpose(2,1).contiguous())

            dec_objs, dec_triples, dec_tight_boxes, dec_objs_to_scene, dec_triples_to_scene, dec_shape_priors = data['decoder']['objs'], \
                                                                                              data['decoder']['triples'], \
                                                                                              data['decoder']['boxes'], \
                                                                                              data['decoder']['obj_to_scene'], \
                                                                                              data['decoder']['tiple_to_scene'], \
                                                                                              data['decoder']['shape_priors']

            missing_nodes = data['missing_nodes']
            manipulated_nodes = data['manipulated_nodes']

        except Exception as e:
            # skipping scene
            continue

        enc_objs, enc_triples, enc_tight_boxes, enc_shape_priros = enc_objs.cuda(), enc_triples.cuda(), enc_tight_boxes.cuda(), enc_shape_priros.cuda()
        dec_objs, dec_triples, dec_tight_boxes, dec_shape_priors = dec_objs.cuda(), dec_triples.cuda(), dec_tight_boxes.cuda(), dec_shape_priors.cuda()

        model = model.eval()

        all_pred_boxes = []

        enc_boxes = enc_tight_boxes[:, :6]

        with torch.no_grad():
            (z_box, _), (z_shape, _) = model.encode_box_and_shape(enc_objs, enc_triples, encoded_enc_points, enc_boxes)

            if args.manipulate:
                boxes_pred, points_pred, keep = model.decoder_with_changes_boxes_and_shape(z_box, dec_objs,dec_triples, dec_shape_priors, missing_nodes, manipulated_nodes, atlas)

        bp = []
        for i in range(len(keep)):
            if keep[i] == 0:
                bp.append(boxes_pred[i].cpu().detach())
            else:
                bp.append(dec_tight_boxes[i,:6].cpu().detach())

        all_pred_boxes.append(boxes_pred.cpu().detach())

        # compute relationship constraints accuracy through simple geometric rules
        accuracy = validate_constrains_changes(dec_triples, boxes_pred, dec_tight_boxes, keep, model.vocab, accuracy,
                                               with_norm=model.type_ != 'sln')
        accuracy_in_orig_graph = validate_constrains_changes(dec_triples, torch.stack(bp, 0), dec_tight_boxes, keep,
                                                             model.vocab, accuracy_in_orig_graph, with_norm=model.type_ != 'sln')
        accuracy_unchanged = validate_constrains(dec_triples, boxes_pred, dec_tight_boxes, keep, model.vocab,
                                                 accuracy_unchanged, with_norm=model.type_ != 'sln')

    keys = list(accuracy.keys())
    for dic, typ in [(accuracy, "changed nodes"), (accuracy_unchanged, 'unchanged nodes'),
                     (accuracy_in_orig_graph, 'changed nodes placed in original graph')]:
        # NOTE 'changed nodes placed in original graph' are the results reported in the paper!
        # The unchanged nodes are kept from the original scene, and the accuracy in the new nodes is computed with
        # respect to these original nodes
        print('{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} '.format(typ, np.mean(dic[keys[0]]), np.mean(dic[keys[1]]), np.mean(dic[keys[2]]), np.mean(dic[keys[3]]), np.mean(dic[keys[4]]), np.mean(dic[keys[5]]), np.mean(dic[keys[6]])))
        print('means of mean: {:.2f}'.format(np.mean([np.mean([np.mean(dic[keys[0]]), np.mean(dic[keys[1]])]),
                                                      np.mean([np.mean(dic[keys[2]]), np.mean(dic[keys[3]])]), np.mean(dic[keys[4]]), np.mean(dic[keys[5]]), np.mean(dic[keys[6]]) ])))




def validate_constrains_loop(testdataloader, model, point_ae=None, export_3d=False):

    accuracy = {}
    for k in ['left', 'right', 'front', 'behind', 'close by', 'symmetrical to', 'total']:
        # compute validation for these relation categories
        accuracy[k] = []

    all_pred_shapes_exp = {} # for export
    all_pred_boxes_exp = {}

    visualize_num = 0
    for i, data in enumerate(testdataloader, 0):
        print(f"now processing {data['scene_id']}")
        try:
            dec_objs, dec_triples, dec_shape_priors = data['decoder']['objs'], data['decoder']['triples'], data['decoder']['shape_priors']
            instances = data['instance_id'][0]
            scan = data['scene_id'][0]

        except Exception as e:
            continue

        dec_objs, dec_triples, dec_shape_priors = dec_objs.cuda(), dec_triples.cuda(), dec_shape_priors.cuda()

        all_pred_boxes = []

        with torch.no_grad():
            boxes_pred, (points_pred, shape_enc_pred) = model.sample_box_and_shape(point_ae, dec_objs, dec_triples, dec_shape_priors)
            """ temp = np.array(points_pred.tolist())
            max_vals = np.max(temp, axis=1)
            min_vals = np.min(temp, axis=1)
            print(min_vals, max_vals)

            result_path = args.exp
            name = 'pred_point_cloud_'+str(np.random.randint(0,100))
            shape_filename = os.path.join(result_path, name + '.json')
            json.dump(points_pred.tolist(), open(shape_filename, 'w')) """

        if model.type_ != 'sln':
            boxes_pred_den = batch_torch_denormalize_box_params(boxes_pred)
            #! 反归一化: 将它们从归一化的参数空间转换回原始参数空间。这对于在处理场景中的物体边界框时很有用，因为归一化处理可以使训练和优化过程更加稳定和高效。
        else:
            boxes_pred_den = boxes_pred

        if export_3d:
            boxes_pred_exp = boxes_pred_den.detach().cpu().numpy().tolist()
            if model.type_ != 'sln':
                # save point encodings
                shapes_pred_exp = shape_enc_pred.detach().cpu().numpy().tolist()

            for i in range(len(shapes_pred_exp)):
                if dec_objs[i] not in testdataloader.dataset.point_classes_idx:
                    shapes_pred_exp[i] = []
                    raise ValueError(f"{dec_objs[i]} not in {testdataloader.dataset.point_classes_idx}")
            shapes_pred_exp = list(shapes_pred_exp)

            if scan not in all_pred_shapes_exp:
                all_pred_boxes_exp[scan] = {}
                all_pred_shapes_exp[scan] = {}

            all_pred_boxes_exp[scan]['objs'] = list(instances)
            all_pred_shapes_exp[scan]['objs'] = list(instances)
            for i in range(len(dec_objs) - 1):
                all_pred_boxes_exp[scan][instances[i]] = list(boxes_pred_exp[i])
                all_pred_shapes_exp[scan][instances[i]] = list(shapes_pred_exp[i])

        if args.visualize:
            # scene graph visualization. saves a picture of each graph to the outfolder
            colormap = vis_graph(scan_id=scan, data_path=args.dataset, outfolder=args.exp + "/vis_graphs/")
            colors = []
            # convert colors to expected format
            def hex_to_rgb(hex):
                hex = hex.lstrip('#')
                hlen = len(hex)
                return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))
            for i in instances:
                h = colormap[str(i)]
                rgb = hex_to_rgb(h)
                colors.append(rgb)
            colors = np.asarray(colors) / 255.

            # layout and shape visualization through open3d
            render_per_shape(points_pred.cpu().detach(), colors=colors)
            render(boxes_pred_den, shapes_pred=points_pred.cpu().detach(), colors=colors, render_boxes=True)
        visualize_num+=1

        all_pred_boxes.append(boxes_pred_den.cpu().detach())

        # compute constraints accuracy through simple geometric rules
        accuracy = validate_constrains(dec_triples, boxes_pred, None, None, model.vocab, accuracy, with_norm=model.type_ != 'sln')

    if export_3d: #! 保存预测的shapes,box
        # export box and shape predictions for future evaluation
        result_path = os.path.join(args.exp, 'results')
        if not os.path.exists(result_path):
            # Create a new directory for results
            os.makedirs(result_path)
        shape_filename = os.path.join(result_path, 'shapes' + '.json')
        box_filename = os.path.join(result_path, 'boxes' + '.json')
        json.dump(all_pred_boxes_exp, open(box_filename, 'w')) # 'dis_nomani_boxes_large.json'
        json.dump(all_pred_shapes_exp, open(shape_filename, 'w'))

    keys = list(accuracy.keys())
    for dic, typ in [(accuracy, "acc")]:

        print('{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} '.format(typ, np.mean(dic[keys[0]]), np.mean(dic[keys[1]]), np.mean(dic[keys[2]]), np.mean(dic[keys[3]]), np.mean(dic[keys[4]]), np.mean(dic[keys[5]]), np.mean(dic[keys[6]])))
        print('means of mean: {:.2f}'.format(np.mean([np.mean([np.mean(dic[keys[0]]), np.mean(dic[keys[1]])]),
                                                      np.mean([np.mean(dic[keys[2]]), np.mean(dic[keys[3]])]), np.mean(dic[keys[4]]), np.mean(dic[keys[5]]), np.mean(dic[keys[6]]) ])))



if __name__ == "__main__": evaluate()
