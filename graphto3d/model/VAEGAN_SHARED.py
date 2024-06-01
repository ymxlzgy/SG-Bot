import torch
import torch.nn as nn
import torch.nn.functional as F
from graphto3d.model.graph import GraphTripleConvNet, _init_weights, make_mlp
import numpy as np


class Sg2ScVAEModel(nn.Module):
    """
    VAE-based network for scene generation and manipulation from a scene graph.
    It has a shared embedding of shape and bounding box latents.
    """
    def __init__(self, vocab,
                 embedding_dim=128,
                 batch_size=32,
                 train_3d=True,
                 decoder_cat=False,
                 Nangle=24,
                 gconv_pooling='avg',
                 gconv_num_layers=5,
                 gconv_num_shared_layer=3,
                 distribution_before=True,
                 mlp_normalization='none',
                 vec_noise_dim=0,
                 use_AE=False,
                 with_changes=True,
                 replace_latent=True,
                 num_box_params=6,
                 residual=False):

        super(Sg2ScVAEModel, self).__init__()
        self.replace_latent = replace_latent
        self.with_changes = with_changes
        self.dist_before = distribution_before

        gconv_dim = embedding_dim
        gconv_hidden_dim = gconv_dim * 4

        box_embedding_dim = int(embedding_dim)
        shape_embedding_dim = int(embedding_dim)

        obj_embedding_dim = embedding_dim

        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.train_3d = train_3d
        self.decoder_cat = decoder_cat
        self.vocab = vocab
        self.vec_noise_dim = vec_noise_dim
        self.use_AE = use_AE

        num_objs = len(vocab['object_idx_to_name'])
        num_preds = len(vocab['pred_idx_to_name'])

        # build network components for encoder and decoder
        self.obj_embeddings_ec_box = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.obj_embeddings_ec_shape = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.pred_embeddings_ec_box = nn.Embedding(num_preds, 2 * embedding_dim)
        self.pred_embeddings_ec_shape = nn.Embedding(num_preds, 2 * embedding_dim)

        self.obj_embeddings_dc_box = nn.Embedding(num_objs + 1, 2*obj_embedding_dim)
        self.obj_embeddings_dc_man = nn.Embedding(num_objs + 1, 2*obj_embedding_dim)
        self.obj_embeddings_dc_shape = nn.Embedding(num_objs + 1, 2*obj_embedding_dim)
        self.pred_embeddings_dc_box = nn.Embedding(num_preds, 5*embedding_dim) #!
        self.pred_embeddings_dc_shape = nn.Embedding(num_preds, 5*embedding_dim)#!

        if self.decoder_cat:
            self.pred_embeddings_dc = nn.Embedding(num_preds, embedding_dim * 2+128)#!
            self.pred_embeddings_man_dc = nn.Embedding(num_preds, embedding_dim * 6+128)#!
        if self.train_3d:
            self.box_embeddings = nn.Linear(num_box_params, box_embedding_dim)
            self.shape_embeddings = nn.Linear(128, shape_embedding_dim)
        else:
            self.box_embeddings = nn.Linear(4, box_embedding_dim)

        # weight sharing of mean and var
        self.box_mean_var = make_mlp([embedding_dim * 2, gconv_hidden_dim, embedding_dim * 2],
                                     batch_norm=mlp_normalization)
        self.box_mean = make_mlp([embedding_dim * 2, box_embedding_dim], batch_norm=mlp_normalization, norelu=True)
        self.box_var = make_mlp([embedding_dim * 2, box_embedding_dim], batch_norm=mlp_normalization, norelu=True)

        self.shape_mean_var = make_mlp([embedding_dim * 2, gconv_hidden_dim, embedding_dim * 2],
                                     batch_norm=mlp_normalization)
        self.shape_mean = make_mlp([embedding_dim * 2, shape_embedding_dim], batch_norm=mlp_normalization, norelu=True)
        self.shape_var = make_mlp([embedding_dim * 2, shape_embedding_dim], batch_norm=mlp_normalization, norelu=True)

        self.gconv_net_ec = None
        self.gconv_net_dc = None

        gconv_kwargs_ec = {
            'input_dim_obj': gconv_dim * 2,
            'input_dim_pred': gconv_dim * 2,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers,
            'mlp_normalization': mlp_normalization,
            'residual': residual,
        }
        gconv_kwargs_dc = {
            'input_dim_obj': gconv_dim * 2,
            'input_dim_pred': gconv_dim * 2,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers,
            'mlp_normalization': mlp_normalization,
            'residual': residual
        }
        gconv_kwargs_shared = {
            'input_dim_obj': gconv_hidden_dim,
            'input_dim_pred': gconv_hidden_dim,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_shared_layer,
            'mlp_normalization': mlp_normalization,
            'residual': residual
        }
        if self.with_changes:
            gconv_kwargs_manipulation = {
                'input_dim_obj': embedding_dim*6 + 128, #! 加入shape_piror 后将6改成7 embedding_dim * 6,
                'input_dim_pred': embedding_dim*6 + 128, #! embedding_dim * 6,
                'hidden_dim': gconv_hidden_dim * 2,
                'output_dim': embedding_dim * 2, 
                'pooling': gconv_pooling,
                'num_layers': gconv_num_layers,
                'mlp_normalization': mlp_normalization,
                'residual': residual,
            }
        if self.decoder_cat:
            gconv_kwargs_dc['input_dim_obj'] = gconv_dim * 4 + 128#!
            gconv_kwargs_dc['input_dim_pred'] = gconv_dim * 4 + 128#!

        self.gconv_net_ec_box = GraphTripleConvNet(**gconv_kwargs_ec)
        self.gconv_net_ec_shape = GraphTripleConvNet(**gconv_kwargs_ec)

        self.gconv_net_dec_box = GraphTripleConvNet(**gconv_kwargs_dc)
        self.gconv_net_dec_shape = GraphTripleConvNet(**gconv_kwargs_dc)
        self.gconv_net_shared = GraphTripleConvNet(**gconv_kwargs_shared)

        if self.with_changes:
            self.gconv_net_manipulation = GraphTripleConvNet(**gconv_kwargs_manipulation)

        # box prediction net
        if self.train_3d:
            box_net_dim = num_box_params
        else:
            box_net_dim = 4
        #box_net_layers = [gconv_dim * 4, gconv_hidden_dim, box_net_dim]
        box_net_layers = [gconv_dim * 5, gconv_hidden_dim, box_net_dim]#!

        self.box_net = make_mlp(box_net_layers, batch_norm=mlp_normalization, norelu=True)

        #shape_net_layers = [gconv_dim * 4, gconv_hidden_dim, 128]
        shape_net_layers = [gconv_dim * 5, gconv_hidden_dim, 128]#!
        self.shape_net = make_mlp(shape_net_layers, batch_norm=mlp_normalization, norelu=True)

        # initialization
        self.box_embeddings.apply(_init_weights)
        self.box_mean_var.apply(_init_weights)
        self.box_mean.apply(_init_weights)
        self.box_var.apply(_init_weights)
        self.shape_mean_var.apply(_init_weights)
        self.shape_mean.apply(_init_weights)
        self.shape_var.apply(_init_weights)
        self.shape_net.apply(_init_weights)
        self.box_net.apply(_init_weights)

    def encoder(self, objs, triples, boxes_gt, shapes_gt):
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1) #! 分别得到主体s, 谓词p, 客体o
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2) #! 主体和客体堆叠在一起, 得到二维张量edges

        #! 将对象objs, 形状shape, 谓词p进行嵌入表示(转换为连续的向量表示)
        obj_vecs_box = self.obj_embeddings_ec_box(objs) #128
        obj_vecs_shape = self.obj_embeddings_ec_shape(objs)#128

        shape_vecs = self.shape_embeddings(shapes_gt)  #128
        pred_vecs_box = self.pred_embeddings_ec_box(p) #256
        pred_vecs_shape = self.pred_embeddings_ec_shape(p) #256

        boxes_vecs = self.box_embeddings(boxes_gt) #128

        #! 将嵌入表示的obj(所有的class_id) 和 嵌入表示的boxes_gt(位置)拼接
        obj_vecs_box = torch.cat([obj_vecs_box, boxes_vecs], dim=1)
        #! 将嵌入表示的shape 和 嵌入表示的shape_gt拼接
        obj_vecs_shape = torch.cat([obj_vecs_shape, shape_vecs], dim=1)

        if self.gconv_net_ec_box is not None:#! 使用图卷积网络GCN处理嵌入表示的obj+boxes_gt 和 嵌入表示的谓词(关系)
            obj_vecs_box, pred_vecs_box = self.gconv_net_ec_box(obj_vecs_box, pred_vecs_box, edges)
            obj_vecs_shape, pred_vecs_shapes = self.gconv_net_ec_shape(obj_vecs_shape, pred_vecs_shape, edges)#! #! 使用图卷积网络处理嵌入表示的shape+shape_gt 和 嵌入表示的谓词(关系)

        if self.dist_before:#! 进入
            #!图卷积网络GCN的输出 obj_vecs_box 和 obj_vecs_shape拼接
            #!图卷积网络GCN的输出 pred_vecs_box 和 pred_vecs_shapes拼接
            obj_vecs_shared = torch.cat([obj_vecs_box, obj_vecs_shape], dim=1)
            pred_vecs_shared = torch.cat([pred_vecs_box, pred_vecs_shapes], dim=1)
            #!在计算均值和方差之前，将对象的位置和形状嵌入表示进行拼接。接着，将它们传递到共享的图卷积网络 gconv_net_shared 中。
            obj_vecs_shared, pred_vecs_shared =self.gconv_net_shared(obj_vecs_shared, pred_vecs_shared, edges)

            obj_vecs_box, obj_vecs_shape = obj_vecs_shared[:, :obj_vecs_box.shape[1]], obj_vecs_shared[:, obj_vecs_box.shape[1]:]
            obj_vecs_box_norot = self.box_mean_var(obj_vecs_box)
            mu_box = self.box_mean(obj_vecs_box_norot)
            logvar_box = self.box_var(obj_vecs_box_norot)

            obj_vecs_shape = self.shape_mean_var(obj_vecs_shape)
            mu_shape = self.shape_mean(obj_vecs_shape)
            logvar_shape = self.shape_var(obj_vecs_shape)

        else:
            obj_vecs_box_norot = self.box_mean_var(obj_vecs_box)
            mu_box = self.box_mean(obj_vecs_box_norot)
            logvar_box = self.box_var(obj_vecs_box_norot)

            obj_vecs_shape = self.shape_mean_var(obj_vecs_shape)
            mu_shape = self.shape_mean(obj_vecs_shape)
            logvar_shape = self.shape_var(obj_vecs_shape)

        #! 然后，根据位置和形状嵌入表示计算均值（mu）和对数方差（logvar）
        mu = torch.cat([mu_box, mu_shape], dim=1)
        logvar = torch.cat([logvar_box, logvar_shape], dim=1)
        return mu, logvar

    def manipulate(self, z, objs, triples, shape_priors):
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs = self.obj_embeddings_dc_man(objs) #
        pred_vecs = self.pred_embeddings_man_dc(p)  #

        obj_vecs = torch.cat([obj_vecs, shape_priors], dim=1)#256+128

        man_z = torch.cat([z, obj_vecs], dim=1) #[n, 256] + [n, 256+128] = [n, 512+128]
        man_z, _ = self.gconv_net_manipulation(man_z, pred_vecs, edges)

        return man_z

    def decoder(self, z, objs, triples, shape_priors):
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs_box = self.obj_embeddings_dc_box(objs)
        obj_vecs_shape = self.obj_embeddings_dc_shape(objs)#!objs: 所有的class -> 输出128*2的向量
        pred_vecs_box = self.pred_embeddings_dc_box(p)
        pred_vecs_shape = self.pred_embeddings_dc_shape(p)

        obj_vecs_box = torch.cat([obj_vecs_box, shape_priors], dim=1) #!
        obj_vecs_shape = torch.cat([obj_vecs_shape, shape_priors], dim=1)#!
        
        if self.decoder_cat:#! 进入
            if self.dist_before:
                obj_vecs_box = torch.cat([obj_vecs_box, z], dim=1)   #! [256+128  + 256]
                obj_vecs_shape = torch.cat([obj_vecs_shape, z], dim=1)

            #! 加入shape_prior之前, GCN的输入shape为 torch.Size([81, 512]), torch.Size([81, 512])
            #! 加入shape_prior之后, GCN的输入obj_vecs_box为[_,640], pred_vecs_box也需要改为[_,640] 
            #! 将self.pred_embeddings_dc_box = nn.Embedding(num_preds, 4*embedding_dim) 改为[5*embedding_dim]
            obj_vecs_box, pred_vecs_box = self.gconv_net_dec_box(obj_vecs_box, pred_vecs_box, edges)#! 图卷积网络
            obj_vecs_shape, pred_vecs_shape = self.gconv_net_dec_shape(obj_vecs_shape, pred_vecs_shape, edges)
            #! 加入shape_prior之前, GCN的输出shape为 torch.Size([81, 512]), torch.Size([81, 512])
            #! 加入shape_prior之后, 输出为[_640, _640]
        else:
            raise NotImplementedError

        #! 加入shape_prior之后需要将box_net和shape_net的make_mlp修改, shape_net_layers[0]=gconv_dim*4 -> gconv_dim*5
        boxes_pred = self.box_net(obj_vecs_box)
        shapes_pred = self.shape_net(obj_vecs_shape)#! make_mlp(shape_net_layers, batch_norm=mlp_normalization, norelu=True)

        return boxes_pred, shapes_pred

    def decoder_with_changes(self, z, dec_objs, dec_triples, shape_priors, missing_nodes, manipulated_nodes):
        # append zero nodes
        nodes_added = []
        for i in range(len(missing_nodes)):
          #! 函数会在潜在空间编码 z 中插入表示缺失节点的零向量。这些零向量将用于生成新的对象。
          ad_id = missing_nodes[i] + i
          nodes_added.append(ad_id)
          noise = np.zeros(self.embedding_dim* 2) # 128*2=256 np.random.normal(0, 1, 64)
          zeros = torch.from_numpy(noise.reshape(1, self.embedding_dim* 2))#shape: (1,256)
          zeros.requires_grad = True
          zeros = zeros.float().cuda()
          z = torch.cat([z[:ad_id], zeros, z[ad_id:]], dim=0)

        # mark changes in nodes
        change_repr = []
        for i in range(len(z)):
        #! 用于表示场景中的变化。对于添加的节点和操作的节点，它们的变化表示为从正态分布采样的随机噪声；对于其他未修改的节点，它们的变化表示为零向量。
            if i not in nodes_added and i not in manipulated_nodes:
                noisechange = np.zeros(self.embedding_dim* 2)
            else:
                noisechange = np.random.normal(0, 1, self.embedding_dim* 2)
            change_repr.append(torch.from_numpy(noisechange).float().cuda())
        change_repr = torch.stack(change_repr, dim=0)
        z_prime = torch.cat([z, change_repr], dim=1)
        z_prime = self.manipulate(z_prime, dec_objs, dec_triples, shape_priors)
        #! 接下来，将 change_repr 与原始的潜在空间编码 z 拼接起来，形成一个新的潜在空间编码 z_prime。

        if self.replace_latent:
            # take original nodes when untouched
            touched_nodes = torch.tensor(sorted(nodes_added + manipulated_nodes)).long()
            for touched_node in touched_nodes:
                z = torch.cat([z[:touched_node], z_prime[touched_node:touched_node + 1], z[touched_node + 1:]], dim=0)
        else:
            z = z_prime

        #! 修改 加入dec_man_enc_shapes_pred
        dec_man_enc_boxes_pred, dec_man_enc_shapes_pred = self.decoder(z, dec_objs, dec_triples, shape_priors)
        num_dec_objs = len(dec_man_enc_boxes_pred)

        keep = []
        for i in range(num_dec_objs):
          if i not in nodes_added and i not in manipulated_nodes:
            keep.append(1)
          else:
            keep.append(0)

        keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()

        return dec_man_enc_boxes_pred, dec_man_enc_shapes_pred, keep

    def forward(self, enc_objs, enc_triples, enc_boxes, enc_shapes, enc_objs_to_scene, \
                dec_objs, dec_triples, dec_boxes, dec_shapes, dec_shape_priors, dec_objs_to_scene, missing_nodes, manipulated_nodes):

        mu, logvar = self.encoder(enc_objs, enc_triples, enc_boxes, enc_shapes)#! 根据位置和形状嵌入表示计算均值（mu）和对数方差（logvar）

        if self.use_AE:
            z = mu
            raise ValueError("use_AE should be False")
        else:#! 重要！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            # reparameterization#! 从给定的均值（mu）和对数方差（logvar）中采样随机变量。
            std = torch.exp(0.5*logvar)#!从对数方差（logvar）计算标准差（std）。这是通过取对数方差的一半并计算每个元素的指数来实现的。
            # standard sampling
            eps = torch.randn_like(std)#! 生成一个与std具有相同形状的随机张量，其元素符合标准正态分布（均值为0，标准差为1）
            z = eps.mul(std).add_(mu)#!执行重参数化技巧。将随机张量eps与std进行元素级别相乘，然后将结果与mu相加。这样得到的z是一个新的随机张量，其元素服从由给定的均值（mu）和对数方差（logvar）定义的正态分布。
            #! 重参数化技巧的目的是允许我们在梯度下降优化过程中反向传播梯度，从而使神经网络能够学习生成随机变量的参数。
            #! z表示隐变量（latent variable），即从输入数据中学习到的一个潜在表示。在训练VAE时，模型试图在隐空间（latent space）中找到能够有效表示数据的低维表示。
            #! 解码器则从z恢复原始数据。在训练过程中，模型试图最小化输入数据和解码器输出数据之间的差异。
        #! z-shape: (n_objs, 256)

        if self.with_changes:
            #首先，在缺失的节点之后添加零节点，并将新添加的节点的ID存储在nodes_added列表中。
            #然后，对于所有节点，如果节点既不是新添加的节点，也不是已经被修改的节点，则将noisechange设置为全零向量。
            #added/manipulated的点 -> 将noisechange设置为从正态分布中随机抽样的向量。这些节点的变化信息存储在change_repr列表中。
            #接下来，将原始的节点向量z和变化向量change_repr在第二维度上拼接成新的节点向量z_prime。
            #最后，使用manipulate函数对新的节点向量进行操作，得到修改后的节点向量z_prime。

            #如果self.replace_latent为真，则将z_prime中的修改后的节点向量替换掉原来的节点向量，以得到最终的节点向量集合z。
            #如果self.replace_latent为假，则直接将z_prime作为最终的节点向量集合返回。
            
            # append zero nodes
            nodes_added = []#该列表将用于记录添加到z张量中的新节点的索引。
            for i in range(len(missing_nodes)):#包含缺失节点的索引的列表。
              ad_id = missing_nodes[i] + i #添加到张量z中的新节点的索引ad_id，该索引等于该节点的索引加上已添加节点的数量。这是因为在向z中添加新节点之后，其索引将会改变。
              nodes_added.append(ad_id)
              noise = np.zeros(self.embedding_dim * 2) # np.random.normal(0, 1, 64)
              zeros = torch.from_numpy(noise.reshape(1, self.embedding_dim * 2))#! z shape: (1, self.embedding_dim * 2)=(1, 256)
              zeros.requires_grad = True
              zeros = zeros.float().cuda()

              z = torch.cat([z[:ad_id], zeros, z[ad_id:]], dim=0) 

            # mark changes in node
            change_repr = []
            for i in range(len(z)):#! 对于所有节点，如果节点既不是新添加的节点，也不是已经被修改的节点，则将noisechange设置为全零向量。
                if i not in nodes_added and i not in manipulated_nodes:
                    noisechange = np.zeros(self.embedding_dim * 2)
                else:
                    noisechange = np.random.normal(0, 1, self.embedding_dim * 2)
                change_repr.append(torch.from_numpy(noisechange).float().cuda())
                #!遍历z中的每个节点：如果节点既不在nodes_added列表中，也不在manipulated_nodes列表中，则将noisechange设置为全零向量。
                #! 否则，将noisechange设置为从正态分布生成的随机噪声向量。
                #! 将noisechange张量转换为浮点类型并移动到GPU（如果使用GPU进行计算），然后将其添加到change_repr列表中。
            change_repr = torch.stack(change_repr, dim=0) #![25,256]
            z_prime = torch.cat([z, change_repr], dim=1)#!原来是等于z(25,256)+(25,256)=[25,512]
            
            z_prime = self.manipulate(z_prime, dec_objs, dec_triples, dec_shape_priors)
            if self.replace_latent:#! 则将z_prime中的修改后的节点向量替换掉原来的节点向量，以得到最终的节点向量集合z。
                # take original nodes when untouched
                touched_nodes = torch.tensor(sorted(nodes_added + manipulated_nodes)).long()
                for touched_node in touched_nodes:
                    z = torch.cat([z[:touched_node], z_prime[touched_node:touched_node + 1], z[touched_node + 1:]],
                                  dim=0)
            else:
                z = z_prime

        dec_man_enc_boxes_pred, dec_man_enc_shapes_pred = self.decoder(z, dec_objs, dec_triples, dec_shape_priors)#! 预测结果

        if not self.with_changes:
            return mu, logvar, dec_man_enc_boxes_pred, dec_man_enc_shapes_pred
        else:#! 进入 将保持不变的节点（即未添加或修改的节点）的预测结果与实际数据进行比较，并将这些结果返回。
            orig_boxes = []
            orig_shapes = []
            orig_gt_boxes = []
            orig_gt_shapes = []
            keep = []
            # keep becomes 1 in the unchanged nodes and 0 otherwise
            for i in range(len(dec_man_enc_boxes_pred)):#! 遍历预测的bbox
              if i not in nodes_added and i not in manipulated_nodes:
                #!如果当前节点不是新添加的节点且不是被修改的节点，将该节点的预测结果（物体位置和形状）以及实际数据添加到相应的列表中。
                #!同时，将keep列表的当前元素设置为1，表示该节点保持不变。否则，将keep列表的当前元素设置为0。
                orig_boxes.append(dec_man_enc_boxes_pred[i:i+1])
                orig_shapes.append(dec_man_enc_shapes_pred[i:i+1])
                orig_gt_boxes.append(dec_boxes[i:i+1])
                orig_gt_shapes.append(dec_shapes[i:i+1])
                keep.append(1)
              else:
                keep.append(0)

            orig_boxes = torch.cat(orig_boxes, dim=0)
            orig_shapes = torch.cat(orig_shapes, dim=0)
            orig_gt_boxes = torch.cat(orig_gt_boxes, dim=0)
            orig_gt_shapes = torch.cat(orig_gt_shapes, dim=0)
            keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()

            return mu, logvar, orig_gt_boxes, orig_gt_shapes, orig_boxes, \
                   orig_shapes, dec_man_enc_boxes_pred, dec_man_enc_shapes_pred, keep

    def sample(self, point_ae, mean_est, cov_est, dec_objs, dec_triples, dec_shape_priors):#! 它从已经训练好的模型中生成样本。
        with torch.no_grad():
            #! 使用随机多元正态分布生成潜在空间向量 z。分布的均值为 mean_est，协方差为 cov_est(从train中获得)。dec_objs.size(0) 决定了生成的样本数量。
            z = torch.from_numpy(np.random.multivariate_normal(mean_est, cov_est, dec_objs.size(0))).float().cuda()
            #! 修改 将生成的潜在空间向量 z 传递给解码器（self.decoder）
            boxes_pred, shapes_pred = self.decoder(z, dec_objs, dec_triples, dec_shape_priors)
            #! 使用点自编码器（point_ae）将解码器输出的 shapes_pred 转换为点云数据。
            #! 这是通过将 shapes_pred 和点自编码器的网格（point_ae.get_grid()）传递给 point_ae.forward_inference_from_latent_space 函数来实现的。
            points = point_ae.forward_inference_from_latent_space(shapes_pred, point_ae.get_grid())
            #! points: 点云 shapes_pred 解码器生成的形状编码
            return boxes_pred, (points, shapes_pred)
        
    def collect_train_statistics(self, train_loader):
        # model = model.eval()
        mean_cat = None

        for idx, data in enumerate(train_loader):
            if data == -1:
                continue

            try:
                dec_objs, dec_triples, dec_tight_boxes, dec_objs_to_scene, dec_triples_to_scene = data['decoder']['objs'], \
                                                                                                  data['decoder'][
                                                                                                      'triples'], \
                                                                                                  data['decoder']['boxes'], \
                                                                                                  data['decoder'][
                                                                                                      'obj_to_scene'], \
                                                                                                  data['decoder'][
                                                                                                      'tiple_to_scene']

                encoded_dec_points = data['decoder']['feats']
                encoded_dec_points = encoded_dec_points.cuda()

            except Exception as e:
                print('Exception', str(e), " evaluate wrong")
                continue

            dec_objs, dec_triples, dec_tight_boxes = dec_objs.cuda(), dec_triples.cuda(), dec_tight_boxes.cuda()
            #print(f"shape of dec_tight_boxes: {dec_tight_boxes.shape}")
            dec_boxes = dec_tight_boxes[:, :6]

            mean, logvar = self.encoder(dec_objs, dec_triples, dec_boxes, encoded_dec_points)

            mean = mean.data.cpu().clone()
            if mean_cat is None:
                    mean_cat = mean.numpy()
            else:
                mean_cat = np.concatenate([mean_cat, mean.numpy()], axis=0)

        mean_est = np.mean(mean_cat, axis=0, keepdims=True)  # size 1*embed_dim
        mean_cat = mean_cat - mean_est
        n = mean_cat.shape[0]
        d = mean_cat.shape[1]
        cov_est = np.zeros((d, d))
        for i in range(n):
            x = mean_cat[i]
            cov_est += 1.0 / (n - 1.0) * np.outer(x, x)
        mean_est = mean_est[0]

        return mean_est, cov_est
