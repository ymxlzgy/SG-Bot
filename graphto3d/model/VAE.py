import torch
import torch.nn as nn

import pickle
import os

from graphto3d.model.VAEGAN_DIS import Sg2ScVAEModel as Dis
from graphto3d.model.VAEGAN_SLN import Sg2ScVAEModel as SLN
from graphto3d.model.VAEGAN_SHARED import Sg2ScVAEModel as Shared
from graphto3d.model.shapeMlp import ShapeMLP


class VAE(nn.Module):   
    def __init__(self, type='dis', vocab=None, replace_latent=False, with_changes=True, distribution_before=True,
                 residual=False, num_box_params=6):
        super().__init__()
        assert type in ['dis', 'sln', 'shared', 'mlp'], '{} is not included in [dis, sln, shared, mlp]'.format(type)

        self.type_ = type
        self.vocab = vocab

        if self.type_ == 'shared':#!进入
            assert distribution_before is not None and replace_latent is not None and with_changes is not None
            self.vae = Shared(vocab, embedding_dim=128, decoder_cat=True, mlp_normalization="batch",
                              gconv_num_layers=5, gconv_num_shared_layer=5, with_changes=with_changes, 
                              distribution_before=distribution_before, replace_latent=replace_latent,
                              num_box_params=num_box_params, residual=residual)

    def forward_mani(self, enc_objs, enc_triples, enc_boxes, enc_shapes, enc_objs_to_scene, \
                     dec_objs, dec_triples, dec_boxes, dec_shapes, dec_shape_priors, dec_objs_to_scene, missing_nodes,
                     manipulated_nodes):

        if self.type_ == 'shared':
            mu, logvar, orig_gt_boxes, orig_gt_shapes, orig_boxes, orig_shapes, boxes, shapes, keep = \
                self.vae.forward(enc_objs, enc_triples, enc_boxes, enc_shapes, enc_objs_to_scene,
                                 dec_objs, dec_triples, dec_boxes, dec_shapes, dec_shape_priors, dec_objs_to_scene,
                                 missing_nodes, manipulated_nodes)
    
            return mu, logvar, mu, logvar, orig_gt_boxes, orig_gt_shapes, orig_boxes, orig_shapes, boxes, shapes, keep

    def load_networks(self, exp, epoch, strict=True):
        if self.type_ == 'shared':
            ckpt = torch.load(os.path.join(exp, 'checkpoint', 'model{}.pth'.format(epoch))).state_dict()
            self.vae.load_state_dict(
                ckpt,
                strict=strict
            )

    def compute_statistics(self, exp, epoch, stats_dataloader, force=False):
        if self.type_ == 'shared':
            stats_f = os.path.join(exp, 'checkpoint', 'model_stats_{}.pkl'.format(epoch))
            if os.path.exists(stats_f) and not force:
                stats = pickle.load(open(stats_f, 'rb'))
                self.mean_est, self.cov_est = stats[0], stats[1]
            else:
                self.mean_est, self.cov_est = self.vae.collect_train_statistics(stats_dataloader)
                pickle.dump([self.mean_est, self.cov_est], open(stats_f, 'wb'))

    def decoder_with_changes_boxes_and_shape(self, z_box, objs, triples, shape_priors, missing_nodes, manipulated_nodes, atlas):
        if self.type_ == 'shared':
            boxes, feats, keep = self.vae.decoder_with_changes(z_box, objs, triples, shape_priors, missing_nodes, manipulated_nodes)
            points = atlas.forward_inference_from_latent_space(feats, atlas.get_grid())

        return boxes, points, keep

    def encode_box_and_shape(self, objs, triples, feats, boxes):
        if self.type_ == 'shared':
            with torch.no_grad():
                z, log_var = self.vae.encoder(objs, triples, boxes, feats)
                return (z, log_var), (z, log_var)

    def sample_box_and_shape(self, point_ae, dec_objs, dec_triplets, dec_shape_priors):
        if self.type_ == 'shared':#! 进入
            return self.vae.sample(point_ae, self.mean_est, self.cov_est, dec_objs,  dec_triplets, dec_shape_priors)

    def save(self, exp, outf, epoch):
        if self.type_ == 'shared':
            torch.save(self.vae, os.path.join(exp, outf, 'model{}.pth'.format(epoch)))
