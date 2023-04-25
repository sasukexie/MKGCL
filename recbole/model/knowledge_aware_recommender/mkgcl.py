# -*- coding: utf-8 -*-
# @Time   : 2020/9/15
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
MKGCL
Reference:

Reference code:
    https://github.com/sasukexie/MKGCL
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
import random
from recbole.utils.align_uniform_util import AverageMeter, CalAlignAndUniformLoss
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Aggregator(nn.Module):
    """ GNN Aggregator layer
    """

    def __init__(self, input_dim, output_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if self.aggregator_type == 'gcn':
            self.W = nn.Linear(self.input_dim, self.output_dim)
        elif self.aggregator_type == 'graphsage':
            self.W = nn.Linear(self.input_dim * 2, self.output_dim)
        elif self.aggregator_type == 'bi':
            self.W1 = nn.Linear(self.input_dim, self.output_dim)
            self.W2 = nn.Linear(self.input_dim, self.output_dim)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()

    def forward(self, norm_matrix, ego_embeddings):
        side_embeddings = torch.sparse.mm(norm_matrix, ego_embeddings)

        if self.aggregator_type == 'gcn':
            ego_embeddings = self.activation(self.W(ego_embeddings + side_embeddings))
        elif self.aggregator_type == 'graphsage':
            ego_embeddings = self.activation(self.W(torch.cat([ego_embeddings, side_embeddings], dim=1)))
        elif self.aggregator_type == 'bi':
            add_embeddings = ego_embeddings + side_embeddings
            sum_embeddings = self.activation(self.W1(add_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = self.activation(self.W2(bi_embeddings))
            ego_embeddings = bi_embeddings + sum_embeddings
        else:
            raise NotImplementedError

        ego_embeddings = self.message_dropout(ego_embeddings)

        return ego_embeddings


class MKGCL(KnowledgeRecommender):
    r"""MKGCAL: Enhancing Alignment and Uniformity of Contrastive Learning for Knowledge-based Recommendation
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MKGCL, self).__init__(config, dataset)
        self.config = config

        # load dataset info
        self.ckg = dataset.ckg_graph(form='dgl', value_field='relation_id')
        self.all_hs = torch.LongTensor(dataset.ckg_graph(form='coo', value_field='relation_id').row).to(self.device)
        self.all_ts = torch.LongTensor(dataset.ckg_graph(form='coo', value_field='relation_id').col).to(self.device)
        self.all_rs = torch.LongTensor(dataset.ckg_graph(form='coo', value_field='relation_id').data).to(self.device)
        self.matrix_size = torch.Size([self.n_users + self.n_entities, self.n_users + self.n_entities])

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.kg_embedding_size = config['kg_embedding_size']
        self.layers = [self.embedding_size] + config['layers']
        self.aggregator_type = config['aggregator_type']
        self.mess_dropout = config['mess_dropout']
        self.reg_weight = config['reg_weight']

        # generate intermediate data
        self.A_in = self.init_graph()  # init the attention matrix by the structure of ckg
        self.A_in_1 = self.A_in
        self.A_in_2 = self.A_in

        # bn
        affine = True
        self.projection_head = torch.nn.ModuleList()
        inner_size = self.layers[-1] * 2
        print("inner size:", inner_size)
        self.projection_head.append(torch.nn.Linear(inner_size, inner_size * 4, bias=False))
        self.projection_head.append(torch.nn.BatchNorm1d(inner_size * 4, eps=1e-12, affine=affine))
        self.projection_head.append(torch.nn.Linear(inner_size * 4, inner_size, bias=False))
        self.projection_head.append(torch.nn.BatchNorm1d(inner_size, eps=1e-12, affine=affine))
        self.mode = 0

        self.projection_head1 = torch.nn.ModuleList()
        inner_size = self.layers[-1]
        print("inner size:", inner_size)
        self.projection_head1.append(torch.nn.Linear(inner_size, inner_size * 4, bias=False))
        self.projection_head1.append(torch.nn.BatchNorm1d(inner_size * 4, eps=1e-12, affine=affine))
        self.projection_head1.append(torch.nn.Linear(inner_size * 4, inner_size, bias=False))
        self.projection_head1.append(torch.nn.BatchNorm1d(inner_size, eps=1e-12, affine=affine))
        self.mode1 = 0

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.trans_w = nn.Embedding(self.n_relations, self.embedding_size * self.kg_embedding_size)
        self.aggregator_layers = nn.ModuleList()
        for idx, (input_dim, output_dim) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.aggregator_layers.append(Aggregator(input_dim, output_dim, self.mess_dropout, self.aggregator_type))
        self.act_func = nn.Tanh()  # GELU,ReLU,swish,Tanh(),Sigmoid()
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        # ce
        self.ce_loss = nn.CrossEntropyLoss()
        self.restore_user_e = None
        self.restore_entity_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_entity_e']
        self.train_batch_size = int(config['train_batch_size'])

        # method
        self.migcl_data_aug = config['migcl_data_aug']  # sen
        self.mulcl_data_aug = config['mulcl_data_aug']  # sen
        self.r_data_aug = config['r_data_aug']  # sen
        self.open_migcl = config['open_migcl']  # True/False
        self.open_mulcl = config['open_mulcl']  # True/False
        self.open_r = config['open_r']  # True/False

        # mulcl
        self.temperature1 = config['temperature1']  # config['T'],0.007,1.0,0.5,0.1,2
        self.temperature2 = config['temperature2']  # config['T'],0.007,1.0,0.5,0.1,2

        self.mini_batch_size_mul = config['mini_batch_size_mul']  # 64,768
        self.m = config['m']  # 0.9

        self.n_layers_q = config['n_layers_q']  # 5
        self.n_layers_k = config['n_layers_k']  # 5

        self.noise_base_a = config['noise_base_a']  # 0.001
        self.noise_base_m = config['noise_base_m']  # 0.001
        self.u_noise_base = config['u_noise_base']  # 0.01
        self.e_noise_base = config['e_noise_base']  # 0.01
        self.r_noise_base = config['r_noise_base']  # 0.01
        self.r_s_len = config['r_s_len']  # 0.01

        # gen random noise
        self.u_noise_q = None
        self.u_noise_k = None
        self.e_noise_q = None
        self.e_noise_k = None

        # align_uniform
        self.align_meter = AverageMeter('align_loss')
        self.unif_meter = AverageMeter('uniform_loss')
        self.loss_meter = AverageMeter('total_loss')
        # self.it_time_meter = AverageMeter('iter_time')
        # representation
        # import pandas as pd
        # self.data_u = pd.read_csv(f'dataset/{config["dataset"]}/{config["dataset"]}.u', sep="\t")
        # self.data_i = pd.read_csv(f'dataset/{config["dataset"]}/{config["dataset"]}.i', sep="\t")
        self.INDEX = 0

        if self.open_r:
            self.r_x = self.r_noise_base + torch.zeros(self.train_batch_size // 8, 64, dtype=torch.float32,
                                                       device=self.device)  # 0.01

        if self.open_mulcl:
            self.u_x = self.u_noise_base + torch.zeros(self.train_batch_size // 8, 128, dtype=torch.float32,
                                                       device=self.device)  # 0.01
            self.e_x = self.e_noise_base + torch.zeros(self.train_batch_size, 128, dtype=torch.float32,
                                                       device=self.device)  # 0.01

            self.u_mini_batch_size = 128  # 128
            self.u_K = self.train_batch_size // 8 * self.mini_batch_size_mul  # 24,32
            self.register_buffer("u_queue", torch.randn(self.u_mini_batch_size, self.u_K))
            self.u_queue = nn.functional.normalize(self.u_queue, dim=0)
            self.register_buffer("u_queue_ptr", torch.zeros(1, dtype=torch.long))

            self.e_mini_batch_size = 128  # 128
            self.e_K = self.train_batch_size * self.mini_batch_size_mul // 8  # 8,32 # not //8
            # self.e_K = self.train_batch_size * 12 # 8,32
            self.register_buffer("e_queue", torch.randn(self.e_mini_batch_size, self.e_K))
            self.e_queue = nn.functional.normalize(self.e_queue, dim=0)
            self.register_buffer("e_queue_ptr", torch.zeros(1, dtype=torch.long))

        if self.open_migcl:
            self.u_x1 = self.u_noise_base + torch.zeros(self.train_batch_size // 8, 128, dtype=torch.float32,
                                                        device=self.device)  # 0.01
            self.e_x1 = self.e_noise_base + torch.zeros(self.train_batch_size, 128, dtype=torch.float32,
                                                        device=self.device)  # 0.01

            self.u_mini_batch_size = 128  # 128
            self.u_K1 = self.train_batch_size // 8 * self.mini_batch_size_mul  # 24,32
            # self.u_K = self.train_batch_size // 8 * 24 # 24,32
            self.register_buffer("u_queue1", torch.randn(self.u_mini_batch_size, self.u_K1))
            self.u_queue1 = nn.functional.normalize(self.u_queue1, dim=0)
            self.register_buffer("u_queue1_ptr", torch.zeros(1, dtype=torch.long))

            self.e_mini_batch_size = 128  # 128
            self.e_K1 = self.train_batch_size * self.mini_batch_size_mul // 8  # 8,32 # not //8
            # self.e_K1 = self.train_batch_size * 12 # 8,32
            self.register_buffer("e_queue1", torch.randn(self.e_mini_batch_size, self.e_K1))
            self.e_queue1 = nn.functional.normalize(self.e_queue1, dim=0)
            self.register_buffer("e_queue1_ptr", torch.zeros(1, dtype=torch.long))

    def init_graph(self):
        r"""Get the initial attention matrix through the collaborative knowledge graph

        Returns:
            torch.sparse.FloatTensor: Sparse tensor of the attention matrix
        """
        import dgl
        adj_list = []
        for rel_type in range(1, self.n_relations, 1):
            edge_idxs = self.ckg.filter_edges(lambda edge: edge.data['relation_id'] == rel_type)
            sub_graph = dgl.edge_subgraph(self.ckg, edge_idxs, preserve_nodes=True). \
                adjacency_matrix(transpose=False, scipy_fmt='coo').astype('float')
            rowsum = np.array(sub_graph.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(sub_graph).tocoo()
            adj_list.append(norm_adj)

        final_adj_matrix = sum(adj_list).tocoo()
        indices = torch.LongTensor([final_adj_matrix.row, final_adj_matrix.col])
        values = torch.FloatTensor(final_adj_matrix.data)
        adj_matrix_tensor = torch.sparse.FloatTensor(indices, values, self.matrix_size)
        return adj_matrix_tensor.to(self.device)

    def _get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, entity_embeddings], dim=0)
        return ego_embeddings

    def forward(self, a_in=None):
        if a_in is None:
            a_in = self.A_in

        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]

        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(a_in, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])
        return user_all_embeddings, entity_all_embeddings

    def _get_kg_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h).unsqueeze(1)
        pos_t_e = self.entity_embedding(pos_t).unsqueeze(1)
        neg_t_e = self.entity_embedding(neg_t).unsqueeze(1)
        r_e = self.relation_embedding(r)
        r_trans_w = self.trans_w(r).view(r.size(0), self.embedding_size, self.kg_embedding_size)

        h_e = torch.bmm(h_e, r_trans_w).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze(1)

        return h_e, r_e, pos_t_e, neg_t_e

    def calculate_loss(self, interaction, batch_idx):
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training rs
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, entity_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        pos_i_embeddings = entity_all_embeddings[pos_item]
        neg_i_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_i_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_i_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_i_embeddings, neg_i_embeddings)
        # reg_loss += self.algn_uniform_loss(batch_idx, 'u-i:', u_embeddings, neg_i_embeddings)
        loss = mf_loss + self.reg_weight * reg_loss

        # migcl loss
        if self.open_migcl:
            loss += self.calculate_migcl_loss(interaction, batch_idx, user, pos_item, u_embeddings, pos_i_embeddings)
        # mulcl loss
        if self.open_mulcl:
            if str.startswith(self.mulcl_data_aug, 'sen'):
                loss += self.calculate_mulcl_loss_by_sen(batch_idx, user, u_embeddings, pos_i_embeddings)
            elif str.startswith(self.mulcl_data_aug, 'gen'):
                loss += self.calculate_mulcl_loss_by_gen(batch_idx, user, u_embeddings, pos_i_embeddings, interaction,
                                                         pos_item, user_all_embeddings, entity_all_embeddings)

        return loss

    def calculate_migcl_loss(self, interaction, batch_idx, user, pos_item, u_embeddings, pos_i_embeddings):
        # calculate migcl loss

        if str.startswith(self.migcl_data_aug, 'sen'):
            # A_in
            user_all_embeddings, entity_all_embeddings = self.forward(self.A_in)

            # migcl_loss build u-u interaction #random sampling # Negative sample
            user_rand_samples = self.rand_sample(user_all_embeddings.shape[0], size=user.shape[0] // 8, replace=False)
            # e_migcl_loss build e-e interaction
            entity_rand_samples = self.rand_sample(entity_all_embeddings.shape[0], size=user.shape[0], replace=False)

            # ui_migcl_loss build u-i interaction
            u_embedding = user_all_embeddings[torch.LongTensor(user_rand_samples)]
            e_embedding = entity_all_embeddings[torch.LongTensor(entity_rand_samples)]

            u_embedding = self.projection_head_map(u_embedding, self.mode)
            e_embedding = self.projection_head_map(e_embedding, self.mode)

            u_embedding_1 = u_embedding
            u_embedding_2 = u_embedding.clone()
            e_embedding_1 = e_embedding
            e_embedding_2 = e_embedding.clone()

            # build small sen
            self.build_sen_migcl()

            # data aug
            try:
                if self.migcl_data_aug == 'sen_a':
                    u_embedding_1 += self.u_noise_q1
                    u_embedding_2 += self.u_noise_k1
                    e_embedding_1 += self.e_noise_q1
                    e_embedding_2 += self.e_noise_k1
                elif self.migcl_data_aug == 'sen_m':
                    u_embedding_1 = torch.mul(u_embedding_1, 1 + self.u_noise_q1)
                    u_embedding_2 = torch.mul(u_embedding_2, 1 + self.u_noise_k1)
                    e_embedding_1 = torch.mul(e_embedding_1, 1 + self.e_noise_q1)
                    e_embedding_2 = torch.mul(e_embedding_2, 1 + self.e_noise_k1)
            except Exception as e:
                return 0

            u_embedding_1 = u_embedding_1.detach()
            u_embedding_2 = u_embedding_2.detach()
            e_embedding_1 = e_embedding_1.detach()
            e_embedding_2 = e_embedding_2.detach()

        else:
            # A_in_1
            user_all_embeddings_1, entity_all_embeddings_1 = self.forward(self.A_in_1)
            # A_in_2
            user_all_embeddings_2, entity_all_embeddings_2 = self.forward(self.A_in_2)

            # migcl_loss build u-u interaction #random sampling # Negative sample
            user_rand_samples = self.rand_sample(user_all_embeddings_1.shape[0], size=user.shape[0] // 8, replace=False)
            # e_migcl_loss build e-e interaction
            entity_rand_samples = self.rand_sample(entity_all_embeddings_1.shape[0], size=user.shape[0], replace=False)

            # ui_migcl_loss build u-i interaction
            u_embedding_1 = user_all_embeddings_1[torch.LongTensor(user_rand_samples)]
            u_embedding_2 = user_all_embeddings_2[torch.LongTensor(user_rand_samples)]

            e_embedding_1 = entity_all_embeddings_1[torch.LongTensor(entity_rand_samples)]
            e_embedding_2 = entity_all_embeddings_2[torch.LongTensor(entity_rand_samples)]

            u_embedding_1 = self.projection_head_map(u_embedding_1, self.mode)
            u_embedding_2 = self.projection_head_map(u_embedding_2, 1 - self.mode)
            e_embedding_1 = self.projection_head_map(e_embedding_1, self.mode)
            e_embedding_2 = self.projection_head_map(e_embedding_2, 1 - self.mode)

        self.mode = 1 - self.mode
        # calculate migcl loss
        u_embeddings = self.projection_head_map(u_embeddings, self.mode)
        pos_i_embeddings = self.projection_head_map(pos_i_embeddings, 1 - self.mode)

        loss = 0

        u_migcl_loss = self.migcl_loss(u_embedding_1, u_embedding_2, batch_size=u_embedding_1.shape[0])
        # u_migcl_loss += self.algn_uniform_loss(batch_idx, 'migcl:u-u', u_embedding_1, u_embedding_2)
        e_migcl_loss = self.migcl_loss(e_embedding_1, e_embedding_2, batch_size=e_embedding_1.shape[0])
        # e_migcl_loss += self.algn_uniform_loss(batch_idx, 'migcl:e-e', e_embedding_1, e_embedding_2)
        ui_migcl_loss = self.migcl_loss(u_embeddings, pos_i_embeddings, batch_size=u_embeddings.shape[0])
        # ui_migcl_loss += self.algn_uniform_loss(batch_idx, 'migcl:u-i', u_embeddings, pos_i_embeddings)
        loss += 0.01 * (u_migcl_loss + e_migcl_loss + ui_migcl_loss)

        # self.representation(batch_idx, interaction, user, pos_item, user_all_embeddings, entity_all_embeddings)

        return loss

    def representation(self, batch_idx, interaction, user, pos_item, user_all_embeddings, entity_all_embeddings):
        if not self.config['open_represent']:
            return 0

        if batch_idx % 50 != 0:
            return

        ins_ = interaction['in_count']
        ins_ = ins_ * 100 / ins_.max()

        xs1, ys1 = [], []
        xs2, ys2 = [], []
        xs3, ys3 = [], []
        xs4, ys4 = [], []

        for idx in range(len(pos_item)):
            i = pos_item[idx]
            in_ = ins_[idx]
            i_e = entity_all_embeddings[i]
            i_e = i_e.view(2, -1).detach()
            if in_ <= 25:
                xs1.extend(i_e[0].cpu().numpy().tolist())
                ys1.extend(i_e[1].cpu().numpy().tolist())
            elif in_ <= 50:
                xs2.extend(i_e[0].cpu().numpy().tolist())
                ys2.extend(i_e[1].cpu().numpy().tolist())
            elif in_ <= 75:
                xs3.extend(i_e[0].cpu().numpy().tolist())
                ys3.extend(i_e[1].cpu().numpy().tolist())
            elif in_ <= 100:
                xs4.extend(i_e[0].cpu().numpy().tolist())
                ys4.extend(i_e[1].cpu().numpy().tolist())

        g = sns.JointGrid()
        sns.set_style("darkgrid")  # darkgrid, whitegrid, dark, white, ticks
        # sns.set_context("notebook")
        sns.set_context("poster", font_scale=1.5, rc={"lines.linewidth": 3})

        color = '#fde725'  # yellow
        # df = pd.DataFrame({"x": xs1, "y": ys1})
        sns.scatterplot(x=xs1, y=ys1, ax=g.ax_joint, color=color)
        sns.kdeplot(x=xs1, ax=g.ax_marg_x, color=color)
        sns.kdeplot(y=ys1, ax=g.ax_marg_y, color=color)

        color = '#35b779'  # green
        sns.scatterplot(x=xs2, y=ys2, ax=g.ax_joint, color=color)
        sns.kdeplot(x=xs2, ax=g.ax_marg_x, color=color)
        sns.kdeplot(y=ys2, ax=g.ax_marg_y, color=color)

        color = '#440154'  # violet
        sns.scatterplot(x=xs3, y=ys3, ax=g.ax_joint, color=color)
        sns.kdeplot(x=xs3, ax=g.ax_marg_x, color=color)
        sns.kdeplot(y=ys3, ax=g.ax_marg_y, color=color)

        color = '#31688e'  # blue
        sns.scatterplot(x=xs4, y=ys4, ax=g.ax_joint, color=color)
        sns.kdeplot(x=xs4, ax=g.ax_marg_x, color=color)
        sns.kdeplot(y=ys4, ax=g.ax_marg_y, color=color)

        # plt.colorbar()
        g.set_axis_labels('', '')
        pic_name = str(self.INDEX)
        if self.config['open_migcl']:
            pic_name = 'migcl_' + pic_name
        if self.config['open_mulcl']:
            pic_name = 'mulcl_' + pic_name
        plt.savefig(r'/data/temp/{}.png'.format(pic_name))
        plt.close()
        self.INDEX += 1

    def algn_uniform_loss(self, batch_idx, name, x, y):
        if not self.config['open_ali_uni']:
            return 0
        align_loss_val = CalAlignAndUniformLoss.align_loss(x, y, alpha=self.config['align_alpha'])
        unif_loss_val = (CalAlignAndUniformLoss.uniform_loss(x, t=self.config[
            'unif_t']) + CalAlignAndUniformLoss.uniform_loss(y, t=self.config['unif_t'])) / 2
        align_uniform_loss = align_loss_val * self.config['align_w'] + unif_loss_val * self.config['unif_w']
        # loss.detach()
        self.align_meter.update(align_loss_val, x.shape[0])
        self.unif_meter.update(unif_loss_val)
        self.loss_meter.update(align_uniform_loss, x.shape[0])

        if batch_idx % self.config['log_interval'] == 0:
            self.logger.info(f"name: {name}\t{self.align_meter}\t{self.unif_meter}\t{self.loss_meter}")
        # loss.backward()
        return align_uniform_loss

    def migcl_loss(self, z_i, z_j, batch_size):  # B * D    B * D

        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)  # 2B * D  ,z*z

        sim = torch.mm(z, z.T) / self.temperature1  # 2B * 2B

        sim_i_j = torch.diag(sim, batch_size)  # B*1
        sim_j_i = torch.diag(sim, -batch_size)  # B*1

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        mask = self.mask_correlated_samples(batch_size)

        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # N * C
        loss = self.ce_loss(logits, labels)
        return loss

    def calculate_mulcl_loss_by_sen(self, batch_idx, user, u_embeddings, pos_i_embeddings):
        # calculate mulcl loss

        # A_in
        user_all_embeddings, entity_all_embeddings = self.forward(self.A_in)

        # migcl_loss build u-u interaction #random sampling # Negative sample
        user_rand_samples = self.rand_sample(user_all_embeddings.shape[0], size=user.shape[0] // 8, replace=False)
        # e_migcl_loss build e-e interaction
        entity_rand_samples = self.rand_sample(entity_all_embeddings.shape[0], size=user.shape[0], replace=False)

        # ui_migcl_loss build u-i interaction
        u_embedding = user_all_embeddings[torch.LongTensor(user_rand_samples)]
        e_embedding = entity_all_embeddings[torch.LongTensor(entity_rand_samples)]

        u_embedding = self.projection_head_map(u_embedding, self.mode)
        e_embedding = self.projection_head_map(e_embedding, self.mode)

        loss = 0

        self.mode = 1 - self.mode
        u_embeddings = self.projection_head_map(u_embeddings, self.mode)
        pos_i_embeddings = self.projection_head_map(pos_i_embeddings, 1 - self.mode)

        # build small sen
        self.build_sen()
        u_mulcl_loss = self.mulcl_loss(u_embedding, self.u_noise_q, self.u_noise_k, queue=self.u_queue,
                                       queue_ptr=self.u_queue_ptr, K=self.u_K)
        e_mulcl_loss = self.mulcl_loss(e_embedding, self.e_noise_q, self.e_noise_k, queue=self.e_queue,
                                       queue_ptr=self.e_queue_ptr, K=self.e_K)
        ui_migcl_loss = self.migcl_loss(u_embeddings, pos_i_embeddings, batch_size=u_embeddings.shape[0])
        loss += 0.01 * (u_mulcl_loss + e_mulcl_loss + ui_migcl_loss)  # 0.01
        return loss

    def calculate_mulcl_loss_by_gen(self, batch_idx, user, u_embeddings, pos_i_embeddings, interaction, pos_item,
                                    user_all_embeddings, entity_all_embeddings):
        # calculate mulcl loss

        # A_in_1
        user_all_embeddings_1, entity_all_embeddings_1 = self.forward(self.A_in_1)
        # A_in_2
        user_all_embeddings_2, entity_all_embeddings_2 = self.forward(self.A_in_2)

        # build u-u interaction #random sampling # Negative sample
        user_rand_samples = self.rand_sample(user_all_embeddings_1.shape[0], size=user.shape[0] // 8, replace=False)
        # build e-e interaction
        entity_rand_samples = self.rand_sample(entity_all_embeddings_1.shape[0], size=user.shape[0], replace=False)

        # build u-i interaction
        u_embedding_1 = user_all_embeddings_1[torch.LongTensor(user_rand_samples)]
        u_embedding_2 = user_all_embeddings_2[torch.LongTensor(user_rand_samples)]

        e_embedding_1 = entity_all_embeddings_1[torch.LongTensor(entity_rand_samples)]
        e_embedding_2 = entity_all_embeddings_2[torch.LongTensor(entity_rand_samples)]

        u_embedding_1 = self.projection_head_map(u_embedding_1, self.mode)
        u_embedding_2 = self.projection_head_map(u_embedding_2, 1 - self.mode)
        e_embedding_1 = self.projection_head_map(e_embedding_1, self.mode)
        e_embedding_2 = self.projection_head_map(e_embedding_2, 1 - self.mode)

        loss = 0

        self.mode = 1 - self.mode
        u_embeddings = self.projection_head_map(u_embeddings, self.mode)
        pos_i_embeddings = self.projection_head_map(pos_i_embeddings, 1 - self.mode)

        u_mulcl_loss = self.mulcl_loss_by_gen(u_embedding_1, u_embedding_2, queue=self.u_queue,
                                              queue_ptr=self.u_queue_ptr, K=self.u_K)
        # u_mulcl_loss += self.algn_uniform_loss(batch_idx, 'mulcl:u-u', u_embedding_1, u_embedding_2)
        e_mulcl_loss = self.mulcl_loss_by_gen(e_embedding_1, e_embedding_2, queue=self.e_queue,
                                              queue_ptr=self.e_queue_ptr, K=self.e_K)
        # e_mulcl_loss += self.algn_uniform_loss(batch_idx, 'mulcl:e-e', e_embedding_1, e_embedding_2)
        ui_migcl_loss = self.migcl_loss(u_embeddings, pos_i_embeddings, batch_size=u_embeddings.shape[0])
        ui_migcl_loss += self.algn_uniform_loss(batch_idx, 'mulcl:u-i', u_embeddings, pos_i_embeddings)
        loss += 0.01 * (u_mulcl_loss + e_mulcl_loss + ui_migcl_loss)  # 0.01

        self.representation(batch_idx, interaction, user, pos_item, user_all_embeddings, entity_all_embeddings)
        return loss

    def mulcl_loss(self, x, noise_q, noise_k, queue, queue_ptr, K):  # B * D    B * D
        # data aug
        try:
            if self.mulcl_data_aug == 'sen_a':
                q = x + noise_q
                k = x + noise_k
            elif self.mulcl_data_aug == 'sen_m':
                q = torch.mul(x, 1 + noise_q)
                k = torch.mul(x, 1 + noise_k)
        except Exception as e:
            return 0

        q = nn.functional.normalize(q, dim=1)
        k = k.detach()
        k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.temperature2

        # labels
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        self._dequeue_and_enqueue(k, queue, queue_ptr, K)

        # CrossEntropyLoss
        loss = self.ce_loss(logits, labels)
        return loss

    def mulcl_loss_by_gen(self, q, k, queue, queue_ptr, K):  # B * D    B * D
        q = nn.functional.normalize(q, dim=1)
        k = k.detach()
        k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.temperature2

        # labels
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        self._dequeue_and_enqueue(k, queue, queue_ptr, K)

        # CrossEntropyLoss
        loss = self.ce_loss(logits, labels)
        return loss

    def build_sen(self):
        # build spacy noise
        self.u_noise_q = None
        self.u_noise_k = None
        self.e_noise_q = None
        self.e_noise_k = None
        for i in range(self.n_layers_q):
            if self.u_noise_q == None:
                self.u_noise_q = torch.normal(mean=torch.tensor([0.0]).to(self.device), std=self.u_x).to(self.device)
                self.e_noise_q = torch.normal(mean=torch.tensor([0.0]).to(self.device), std=self.e_x).to(self.device)
            else:
                self.u_noise_q += torch.normal(mean=torch.tensor([0.0]).to(self.device), std=self.u_x).to(self.device)
                self.e_noise_q += torch.normal(mean=torch.tensor([0.0]).to(self.device), std=self.e_x).to(self.device)

        for i in range(self.n_layers_k):
            if self.u_noise_k == None:
                self.u_noise_k = torch.normal(mean=torch.tensor([0.0]).to(self.device), std=self.u_x).to(self.device)
                self.e_noise_k = torch.normal(mean=torch.tensor([0.0]).to(self.device), std=self.e_x).to(self.device)
            else:
                self.u_noise_k += torch.normal(mean=torch.tensor([0.0]).to(self.device), std=self.u_x).to(self.device)
                self.e_noise_k += torch.normal(mean=torch.tensor([0.0]).to(self.device), std=self.e_x).to(self.device)

    def build_sen_migcl(self):
        # build Spatial Embedding Noise
        self.u_noise_q1 = None
        self.u_noise_k1 = None
        self.e_noise_q1 = None
        self.e_noise_k1 = None
        for i in range(self.n_layers_q):
            if self.u_noise_q1 == None:
                self.u_noise_q1 = torch.normal(mean=torch.tensor([0.0]).to(self.device), std=self.u_x1).to(self.device)
                self.e_noise_q1 = torch.normal(mean=torch.tensor([0.0]).to(self.device), std=self.e_x1).to(self.device)
            else:
                self.u_noise_q1 += torch.normal(mean=torch.tensor([0.0]).to(self.device), std=self.u_x1).to(self.device)
                self.e_noise_q1 += torch.normal(mean=torch.tensor([0.0]).to(self.device), std=self.e_x1).to(self.device)

        for i in range(self.n_layers_k):
            if self.u_noise_k1 == None:
                self.u_noise_k1 = torch.normal(mean=torch.tensor([0.0]).to(self.device), std=self.u_x1).to(self.device)
                self.e_noise_k1 = torch.normal(mean=torch.tensor([0.0]).to(self.device), std=self.e_x1).to(self.device)
            else:
                self.u_noise_k1 += torch.normal(mean=torch.tensor([0.0]).to(self.device), std=self.u_x1).to(self.device)
                self.e_noise_k1 += torch.normal(mean=torch.tensor([0.0]).to(self.device), std=self.e_x1).to(self.device)

    def build_sen1(self):
        # build Spatial Embedding Noise1
        self.u_noise_q = None
        self.u_noise_k = None
        self.e_noise_q = None
        self.e_noise_k = None
        for i in range(self.n_layers_q):
            if self.u_noise_q == None:
                self.u_noise_q = (self.u_noise_base * 2) * torch.rand(self.u_x.size()).to(
                    self.device) - self.u_noise_base
                self.e_noise_q = (self.e_noise_base * 2) * torch.rand(self.e_x.size()).to(
                    self.device) - self.e_noise_base
            else:
                self.u_noise_q += (self.u_noise_base * 2) * torch.rand(self.u_x.size()).to(
                    self.device) - self.u_noise_base
                self.e_noise_q += (self.e_noise_base * 2) * torch.rand(self.e_x.size()).to(
                    self.device) - self.e_noise_base

        for i in range(self.n_layers_k):
            if self.u_noise_k == None:
                self.u_noise_k = (self.u_noise_base * 2) * torch.rand(self.u_x.size()).to(
                    self.device) - self.u_noise_base
                self.e_noise_k = (self.e_noise_base * 2) * torch.rand(self.e_x.size()).to(
                    self.device) - self.e_noise_base
            else:
                self.u_noise_k += (self.u_noise_base * 2) * torch.rand(self.u_x.size()).to(
                    self.device) - self.u_noise_base
                self.e_noise_k += (self.e_noise_base * 2) * torch.rand(self.e_x.size()).to(
                    self.device) - self.e_noise_base

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue, queue_ptr, K):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        if K % batch_size != 0:  # for simplicity
            # print("batch_size: ", batch_size)
            return
        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % K  # move pointer

        queue_ptr[0] = ptr

    def rand_sample(self, high, size=None, replace=True):
        r"""Randomly discard some points or edges.
        Args:
            high (int): Upper limit of index value
            size (int): Array size after sampling
        Returns:
            numpy.ndarray: Array index after sampling, shape: [size]
        """
        a = np.arange(high)
        sample = np.random.choice(a, size=size, replace=replace)
        return sample

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def projection_head_map(self, state, mode):
        """
        1.BN is used to realize the scattering of embedding. BN: Bootstrap Your Own Latent A New Approach to Self-Supervised Learning
        2.The same effect as the shuffle bn is achieved by constantly switching the mode of the BN. Shuffle BN: Momentum Contrast for Unsupervised Visual Representation Learning
        """
        for i, l in enumerate(self.projection_head):  # 0: Linear 1: BN (relu)  2: Linear 3:BN (relu)
            if i % 2 != 0:
                if mode == 0:
                    l.train()  # set BN to train mode: use a learned mean and variance.
                else:
                    l.eval()  # set BN to eval mode: use a accumulated mean and variance.
            state = l(state)
            if i % 2 != 0:
                state = F.relu(state)
        return state

    def calculate_kg_loss(self, interaction, r_s=None):
        r"""Calculate the training loss for a batch data of KG.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """

        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training kg
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_embedding(h, r, pos_t, neg_t)

        pos_tail_score = ((h_e + r_e - pos_t_e) ** 2).sum(dim=1)
        neg_tail_score = ((h_e + r_e - neg_t_e) ** 2).sum(dim=1)
        kg_loss = F.softplus(pos_tail_score - neg_tail_score).mean()
        kg_reg_loss = self.reg_loss(h_e, r_e, pos_t_e, neg_t_e)
        loss = kg_loss + self.reg_weight * kg_reg_loss

        if self.open_r:
            r_loss = 0
            if str.startswith(self.r_data_aug, 'sen'):
                r_loss = self.r_e_loss(r, r_e)
            elif self.r_data_aug == 'r_s':
                r_loss = self.r_s_loss(r, r_s)
            loss += r_loss

        return loss

    def r_e_loss(self, r, r_embedding_all):

        r_rand_samples = self.rand_sample(r_embedding_all.shape[0], size=r.shape[0] // 8, replace=False)

        # build u-i interaction
        r_embedding = r_embedding_all[torch.LongTensor(r_rand_samples)]

        r_embedding = self.projection_head_map1(r_embedding, self.mode1)

        r_embedding_1 = self.r_e_data_aug(r_embedding)
        r_embedding_2 = self.r_e_data_aug(r_embedding)

        r_loss = self.migcl_loss(r_embedding_1, r_embedding_2, batch_size=r_embedding_1.shape[0])
        return 0.01 * r_loss

    def r_s_loss(self, r, r_s):
        if r_s == None:
            return 0

        r1 = self.r_s_data_aug(r, r_s)
        r2 = self.r_s_data_aug(r, r_s)
        r_embedding_all_1 = self.relation_embedding(r1)
        r_embedding_all_2 = self.relation_embedding(r2)
        r_rand_samples = self.rand_sample(r_embedding_all_1.shape[0], size=r.shape[0] // 8, replace=False)

        # build u-i interaction
        r_embedding_1 = r_embedding_all_1[torch.LongTensor(r_rand_samples)]
        r_embedding_1 = self.projection_head_map1(r_embedding_1, self.mode1)
        r_embedding_2 = r_embedding_all_2[torch.LongTensor(r_rand_samples)]
        r_embedding_2 = self.projection_head_map1(r_embedding_2, self.mode1)

        r_loss = self.migcl_loss(r_embedding_1, r_embedding_2, batch_size=r_embedding_1.shape[0])
        return 1 * r_loss

    def projection_head_map1(self, state, mode):
        """
        1.BN is used to realize the scattering of embedding. BN: Bootstrap Your Own Latent A New Approach to Self-Supervised Learning
        2.The same effect as the shuffle bn is achieved by constantly switching the mode of the BN. Shuffle BN: Momentum Contrast for Unsupervised Visual Representation Learning
        """
        for i, l in enumerate(self.projection_head1):  # 0: Linear 1: BN (relu)  2: Linear 3:BN (relu)
            if i % 2 != 0:
                if mode == 0:
                    l.train()  # set BN to train mode: use a learned mean and variance.
                else:
                    l.eval()  # set BN to eval mode: use a accumulated mean and variance.
            state = l(state)
            if i % 2 != 0:
                state = F.relu(state)
        return state

    # r_e data aug
    def r_e_data_aug(self, r_e):
        try:
            r_noise = torch.normal(mean=torch.tensor([0.0]).to(self.device), std=self.r_x).to(self.device)
            if self.r_data_aug == 'sen_a':
                r_e = r_e + r_noise
            elif self.r_data_aug == 'sen_m':
                r_e = torch.mul(r_e, 1 + r_noise)
        except Exception as e:
            pass
        return r_e

    # r data aug
    def r_s_data_aug(self, r, r_s):
        r = torch.clone(r)
        r_rand_samples = torch.LongTensor(
            self.rand_sample(r.shape[0], size=int(r.shape[0] * self.r_s_len), replace=False))
        for i in r_rand_samples:
            r_i = r[i]
            for r_s_i in r_s['arr']:
                if r_i in r_s_i:
                    r[i] = random.choice(r_s_i)
                    break
        return r

    def generate_transE_score(self, hs, ts, r):
        r"""Calculating scores for triples in KG.

        Args:
            hs (torch.Tensor): head entities
            ts (torch.Tensor): tail entities
            r (int): the relation id between hs and ts

        Returns:
            torch.Tensor: the scores of (hs, r, ts)
        """

        all_embeddings = self._get_ego_embeddings().cpu()
        h_e = all_embeddings[hs]
        t_e = all_embeddings[ts]
        r_e = self.relation_embedding.weight[r].cpu()
        r_trans_w = self.trans_w.weight[r].view(self.embedding_size, self.kg_embedding_size).cpu()

        h_e = torch.matmul(h_e, r_trans_w)
        t_e = torch.matmul(t_e, r_trans_w)

        kg_score = torch.mul(t_e, self.act_func(h_e + r_e)).sum(dim=1)

        return kg_score

    def update_attentive_A(self):
        r"""
        Update the attention matrix using the updated embedding matrix
        """

        kg_score_list, row_list, col_list = [], [], []
        # To reduce the GPU memory consumption, we calculate the scores of KG triples according to the type of relation
        # gpu memory is insufficient, transfer to cpu
        for rel_idx in range(1, self.n_relations, 1):
            triple_index = torch.where(self.all_rs == rel_idx)
            kg_score = self.generate_transE_score(self.all_hs[triple_index].cpu(), self.all_ts[triple_index].cpu(),
                                                  rel_idx)
            row_list.append(self.all_hs[triple_index].cpu())
            col_list.append(self.all_ts[triple_index].cpu())
            kg_score_list.append(kg_score)
        kg_score = torch.cat(kg_score_list, dim=0)
        row = torch.cat(row_list, dim=0)
        col = torch.cat(col_list, dim=0)
        indices = torch.cat([row, col], dim=0).view(2, -1)
        # Current PyTorch version does not support softmax on SparseCUDA, temporarily move to CPU to calculate softmax
        A_in = torch.sparse.FloatTensor(indices, kg_score, self.matrix_size)  # .cpu()
        A_in = torch.sparse.softmax(A_in, dim=1).to(self.device)
        self.A_in = A_in

        # data augmentation
        A_in_1, A_in_2 = None, None
        if 'ed' in [self.migcl_data_aug, self.mulcl_data_aug]:
            A_in_1 = self.data_aug_by_ed(indices, kg_score)
            A_in_2 = self.data_aug_by_ed(indices, kg_score)
        elif 'gen_a:ed' in [self.migcl_data_aug, self.mulcl_data_aug]:
            A_in_1 = self.data_aug_by_eda(indices, kg_score)
            A_in_2 = self.data_aug_by_eda(indices, kg_score)
        elif 'gen_m:ed' in [self.migcl_data_aug, self.mulcl_data_aug]:
            A_in_1 = self.data_aug_by_edm(indices, kg_score)
            A_in_2 = self.data_aug_by_edm(indices, kg_score)
        elif 'gen_a' in [self.migcl_data_aug, self.mulcl_data_aug]:
            A_in_1 = self.data_aug_by_gen_a(indices, kg_score)
            A_in_2 = self.data_aug_by_gen_a(indices, kg_score)
        elif 'gen_m' in [self.migcl_data_aug, self.mulcl_data_aug]:
            A_in_1 = self.data_aug_by_gen_m(indices, kg_score)
            A_in_2 = self.data_aug_by_gen_m(indices, kg_score)

        self.A_in_1 = A_in_1
        self.A_in_2 = A_in_2

    def data_aug_by_ed(self, indices, kg_score):
        r""" ed: Edge Dropout """
        drop_edge = self.rand_sample(indices.shape[1], size=int(indices.shape[1] * 0.1), replace=False)
        indices = indices.view(-1, 2)[torch.LongTensor(drop_edge)].view(2, -1)
        kg_score = kg_score[torch.LongTensor(drop_edge)]

        # Current PyTorch version does not support softmax on SparseCUDA, temporarily move to CPU to calculate softmax
        A_in = torch.sparse.FloatTensor(indices, kg_score, self.matrix_size).cpu()
        A_in = torch.sparse.softmax(A_in, dim=1).to(self.device)

        return A_in

    def data_aug_by_eda(self, indices, kg_score):
        r""" ed: Edge Dropout """
        indices, kg_score = self.data_aug_by_gen_a(indices, kg_score, flag='ik')

        drop_edge = self.rand_sample(indices.shape[1], size=int(indices.shape[1] * 0.1), replace=False)
        indices = indices.view(-1, 2)[torch.LongTensor(drop_edge)].view(2, -1)
        kg_score = kg_score[torch.LongTensor(drop_edge)]

        # Current PyTorch version does not support softmax on SparseCUDA, temporarily move to CPU to calculate softmax
        A_in = torch.sparse.FloatTensor(indices, kg_score, self.matrix_size).cpu()
        A_in = torch.sparse.softmax(A_in, dim=1).to(self.device)

        return A_in

    def data_aug_by_edm(self, indices, kg_score):
        r""" ed: Edge Dropout """
        indices, kg_score = self.data_aug_by_gen_m(indices, kg_score, flag='ik')

        drop_edge = self.rand_sample(indices.shape[1], size=int(indices.shape[1] * 0.1), replace=False)
        indices = indices.view(-1, 2)[torch.LongTensor(drop_edge)].view(2, -1)
        kg_score = kg_score[torch.LongTensor(drop_edge)]

        # Current PyTorch version does not support softmax on SparseCUDA, temporarily move to CPU to calculate softmax
        A_in = torch.sparse.FloatTensor(indices, kg_score, self.matrix_size).cpu()
        A_in = torch.sparse.softmax(A_in, dim=1).to(self.device)

        return A_in

    def data_aug_by_gen_a(self, indices, kg_score, flag=None):
        r""" Graph Embedding Noise """

        x = self.noise_base_a + torch.zeros(1, indices.shape[1], dtype=torch.float32, device=indices.device)  # 0.01
        noise = torch.normal(mean=torch.tensor([0.0]).to(indices.device), std=x).to(indices.device)
        # all turn positive number
        # noise = torch.abs(noise)

        indices += noise.cpu().type(torch.LongTensor).to(indices.device)
        # all turn positive number
        indices = torch.abs(indices)
        # all less self.matrix_size[0]
        indices = torch.where(indices <= self.matrix_size[0] - 1, indices, self.matrix_size[0] - 1)

        # kg_score += noise[:,0] * 0.1

        if flag == 'ik':
            return indices, kg_score

        A_in = torch.sparse.FloatTensor(indices, kg_score, self.matrix_size).cpu()
        A_in = torch.sparse.softmax(A_in, dim=1).to(self.device)

        return A_in

    def data_aug_by_gen_m(self, indices, kg_score, flag=None):
        r""" Graph Embedding Noise """

        indices = indices.to(self.device)
        kg_score = kg_score.to(self.device)
        x = self.noise_base_m + torch.zeros(1, indices.shape[1], dtype=torch.float32, device=self.device)  # 0.01
        noise = torch.normal(mean=torch.tensor([0.0]).to(self.device), std=x).to(self.device)

        indices = torch.mul(indices, 1 + noise).cpu().type(torch.LongTensor).to(self.device)
        # all turn positive number
        indices = torch.abs(indices)
        # all less self.matrix_size[0]
        indices = torch.where(indices <= self.matrix_size[0] - 1, indices, self.matrix_size[0] - 1)

        # kg_score = torch.mul(kg_score, noise)

        if flag == 'ik':
            return indices, kg_score

        A_in = torch.sparse.FloatTensor(indices, kg_score, self.matrix_size).cpu()
        A_in = torch.sparse.softmax(A_in, dim=1).to(self.device)

        return A_in

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, entity_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = entity_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_entity_e is None:
            self.restore_user_e, self.restore_entity_e = self.forward()
        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_entity_e[:self.n_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))

        return scores.view(-1)
