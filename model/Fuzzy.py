import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from collections import OrderedDict
from utility.evaluate import ranklist_by_heapq, get_performance
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from model.KAN import KANLinear

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

class FuzzyUI(nn.Module):
    def __init__(self, dim, membership_type, num_rules=12, batch_size=512, initial_centers=None, initial_scales=None):
        super(FuzzyUI, self).__init__()
        self.dim = dim
        self.num_rules = num_rules
        self.batch_size = batch_size
        self.dim_rule = int(self.dim / self.num_rules)
        self.membership_type = membership_type
        if initial_centers is None:
            if self.membership_type == 'trapezoidal':
                initial_centers = nn.init.uniform_(torch.empty(self.num_rules * 4, dim), a=-1.0, b=1.0)
                initial_centers = initial_centers.view(-1, 4)
                initial_centers, _ = torch.sort(initial_centers, dim=1)
                offsets = torch.linspace(0.1, 0.4, steps=4).to(initial_centers.device)
                initial_centers = initial_centers + offsets
                initial_centers = initial_centers.view(-1, dim)
            else:
                initial_centers = nn.init.xavier_uniform_(torch.empty(self.num_rules, self.dim))
        if initial_scales is None:
            initial_scales = nn.init.uniform_(torch.empty(self.num_rules, self.dim), a=1.0, b=2.0)
            self.scales = nn.Parameter(initial_scales)
        self.centers = nn.Parameter(initial_centers)

        if self.membership_type == 'bell':
            self.m = nn.Parameter(torch.tensor(1.0))
        self.kan_linears = nn.ModuleList([KANLinear(1, 1) for _ in range(num_rules)])
        self.batch_norm = nn.BatchNorm1d(num_rules)

        self.defuzzifier = nn.Sequential(
            nn.BatchNorm1d(num_rules),
            nn.Linear(num_rules, 1),
            nn.Sigmoid()
        )
        self.scale = nn.Parameter(torch.tensor(2.75))
        self.apply(initialize_weights)

    # gaussian membership
    def gaussian_membership(self, x, fast_weights=None, i=0):
        def checkpointed_gaussian_membership(x, centers, scales):
            diff = x.unsqueeze(1) - centers
            current_batch_size = x.size(0)
            scales = scales[:current_batch_size]
            memberships = torch.exp(-diff** 2 / (2 * scales.unsqueeze(0) ** 2))
            memberships = torch.sum(memberships , dim=2)/self.dim  # [batch_size, num_rules]
            return memberships
        centers,scales = self.centers,self.scales
        if fast_weights is not None:
            centers = fast_weights.get(f'convs.{i}.fuzzy_ui.centers', self.centers)
            scales = fast_weights.get(f'convs.{i}.fuzzy_ui.scales', self.scales)
        return checkpoint(checkpointed_gaussian_membership, x, centers, scales, use_reentrant=False)

    # triangular membership
    def triangular_membership(self, x, fast_weights=None, i=0):
        def checkpointed_triangular_membership(x, centers, scales):
            diff = torch.abs(x.unsqueeze(1) - centers)  # [batch_size, num_rules, dim]
            current_batch_size = x.size(0)
            scales = scales[:current_batch_size]
            scales = torch.clamp(scales, min=1e-6)
            memberships = torch.clamp(1 - diff / scales.unsqueeze(0), min=0.0)
            log_memberships = torch.log(memberships + 1)
            penalty = torch.sum(log_memberships, dim=2)
            adjusted_penalty = (penalty + torch.sum(memberships, dim=2)) / self.dim
            return adjusted_penalty
        centers,scales = self.centers,self.scales
        if fast_weights is not None:
            centers = fast_weights.get(f'convs.{i}.fuzzy_ui.centers', self.centers)
            scales = fast_weights.get(f'convs.{i}.fuzzy_ui.scales', self.scales)
        return checkpoint(checkpointed_triangular_membership, x, centers, scales, use_reentrant=False)

    # trapezoidal membership
    def trapezoidal_membership(self, x, fast_weights=None, i=0):
        def checkpointed_trapezoidal_membership(x, centers):
            a, b, c, d = torch.chunk(centers, 4, dim=0)  # [num_rules*4, dim] -> [num_rules, dim] * 4
            b = torch.max(b, a + 1e-6)  # b > a
            c = torch.max(c, b + 1e-6)  # c > b
            d = torch.max(d, c + 1e-6)  # d > c
            x = x.unsqueeze(1)
            x = torch.zeros_like(x)
            memberships = torch.where(
                x < a, torch.zeros_like(x),
                torch.where(
                    x < b, (x - a) / (b - a),
                    torch.where(
                        x < c, torch.ones_like(x),
                        torch.clamp((d - x) / (d - c), min=0.0)
                    )
                )
            )
            memberships = torch.sum(memberships, dim=2) / self.dim
            return memberships
        centers = self.centers
        if fast_weights is not None:
            centers = fast_weights.get(f'convs.{i}.fuzzy_ui.centers', self.centers)
        return checkpoint(checkpointed_trapezoidal_membership, x, centers, use_reentrant=False)

    # s_shape
    def s_shaped_membership(self, x, fast_weights=None, i=0):
        def checkpointed_s_shaped_membership(x, centers, scales):
            diff = x.unsqueeze(1) - centers  # [batch_size, num_rules, dim]
            current_batch_size = x.size(0)
            scales = scales[:current_batch_size]
            memberships = 1 / (1 + torch.exp(-diff / scales.unsqueeze(0)))
            memberships = torch.sum(memberships, dim=2) / self.dim
            return memberships
        centers,scales = self.centers,self.scales
        if fast_weights is not None:
            centers = fast_weights.get(f'convs.{i}.fuzzy_ui.centers', self.centers)
            scales = fast_weights.get(f'convs.{i}.fuzzy_ui.scales', self.scales)
        return checkpoint(checkpointed_s_shaped_membership, x, centers, scales, use_reentrant=False)

    # bell_shaped_membership
    def bell_shaped_membership(self, x, fast_weights=None, i=0):
        def checkpointed_bell_shaped_membership(x, centers, scales,m):
            diff = x.unsqueeze(1) - centers  # [batch_size, num_rules, dim]
            current_batch_size = x.size(0)
            scales = scales[:current_batch_size]
            scales = torch.clamp(scales, min=1e-6)
            memberships = torch.clamp(1 / (1 + (torch.abs(diff) / scales.unsqueeze(0)) ** (2*m)), min=0.0)
            memberships = torch.sum(memberships, dim=2) / self.dim
            return memberships
        centers,scales,m = self.centers,self.scales,self.m
        if fast_weights is not None:
            centers = fast_weights.get(f'convs.{i}.fuzzy_ui.centers', self.centers)
            scales = fast_weights.get(f'convs.{i}.fuzzy_ui.scales', self.scales)
            m = fast_weights.get(f'convs.{i}.fuzzy_ui.m', self.m)
        return checkpoint(checkpointed_bell_shaped_membership, x, centers, scales, m,use_reentrant=False)

    def forward_fuzzy_rule(self, combined):
        return checkpoint(self.fuzzy_rule, combined, use_reentrant=False)

    def forward_defuzzifier(self, fuzzy_output):
        return checkpoint(self.defuzzifier, fuzzy_output, use_reentrant=False)

    def forward(self, neigh_emb, fast_weights=None, i=0):
        """
        neigh_emb: [n_edges, dim]
        fast_weights: dict,
        return: [n_edges, 1]
        """
        n_edges = neigh_emb.size(0)
        iter_num = (n_edges + self.batch_size - 1) // self.batch_size

        similarity = neigh_emb.new_empty(n_edges, 1)

        with torch.cuda.amp.autocast():
            for i in range(iter_num):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, n_edges)
                neigh = neigh_emb[start_idx:end_idx]

                if self.membership_type == 'gaussian':
                    memberships = self.gaussian_membership(neigh,fast_weights,i)
                elif self.membership_type == 'triangular':
                    memberships = self.triangular_membership(neigh,fast_weights,i)
                elif self.membership_type == 'trapezoidal':
                    memberships = self.trapezoidal_membership(neigh,fast_weights,i)
                elif self.membership_type == 's_shaped':
                    memberships = self.s_shaped_membership(neigh,fast_weights,i)
                elif self.membership_type == 'bell':
                    memberships = self.bell_shaped_membership(neigh, fast_weights,i)
                fuzzy_output = [self.kan_linears[i](memberships[:, i].unsqueeze(1)) for i in range(len(self.kan_linears))]
                fuzzy_output = torch.cat(fuzzy_output, dim=1)
                fuzzy_output = self.batch_norm(fuzzy_output)

                sim = self.forward_defuzzifier(fuzzy_output)
                sim = sim * self.scale  # [batch, 1]
                similarity[start_idx:end_idx] = sim
                del neigh, memberships, fuzzy_output, sim
        return similarity


class Aggregator(nn.Module):
    def __init__(self, n_users, n_items, channel,membership_type,num_rules):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.sigmoid = nn.Sigmoid()
        self.fuzzy_ui = FuzzyUI(channel,membership_type,num_rules)

    def forward(self, entity_emb, user_emb, edge_index, edge_type, interact_mat, weight, fast_weights=None, i=0):
        # """item-tags"""
        n_entities = entity_emb.shape[0]
        head, tail = edge_index
        edge_relation_emb = weight[edge_type]
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        del edge_relation_emb, neigh_relation_emb, head, tail, edge_index
        torch.cuda.empty_cache()

        # """"user-items"""
        item_res_emb = entity_agg[:self.n_items]
        mat_row = interact_mat._indices()[0, :]
        mat_col = interact_mat._indices()[1, :]
        neigh_emb = item_res_emb[mat_col] * weight[0]
        del item_res_emb
        combined_emb = user_emb[mat_row]
        user_neigh_emb = (neigh_emb+combined_emb)/2
        del neigh_emb,combined_emb

        if fast_weights == None:
            fuzzy_weights = self.fuzzy_ui(user_neigh_emb)
        else:
            fuzzy_weights_ui = {k: v for k, v in fast_weights.items() if (('{}.fuzzy_ui.scales'.format(str(i)) in k) 
                                                                        and ('{}.fuzzy_ui.centers'.format(str(i)) in k) 
                                                                        and ('{}.fuzzy_ui.m'.format(str(i)) in k))}
            fuzzy_weights = self.fuzzy_ui(user_neigh_emb, fast_weights=fuzzy_weights_ui, i=i)
            del fuzzy_weights_ui
        weighted_user_emb = user_neigh_emb * fuzzy_weights
        user_agg = scatter_mean(src=weighted_user_emb, index=mat_row, dim_size=self.n_users,
                                dim=0)  # [n_users, dim]
        user_emb = user_emb + user_agg
        del user_neigh_emb, fuzzy_weights, weighted_user_emb

        items_u = user_agg[mat_row]
        item_emb = scatter_mean(src=items_u, index=mat_col, dim_size=self.n_items, dim=0)
        entity_emb = torch.cat((item_emb, entity_emb[self.n_items:]), dim=0)

        del mat_row, mat_col, user_agg, items_u
        torch.cuda.empty_cache()
        return entity_emb, user_emb


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, channel, n_hops, n_users, n_relations, n_items, n_entities,membership_type,num_rules,
                 node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.sigmoid = nn.Sigmoid()
        weight = nn.init.xavier_uniform_(torch.empty(n_relations, channel))  # not include interact
        scaling_factor = 1e1
        weight *= scaling_factor
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

        self.n_layers = n_hops
        for i in range(self.n_layers):
            self.convs.append(Aggregator(n_users=n_users, n_items=n_items, channel=channel,
                                         membership_type=membership_type,num_rules=num_rules))

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_emb, entity_emb, edge_index, edge_type,
                interact_mat, fast_weights=None, mess_dropout=True, node_dropout=True):

        """node dropout"""

        if node_dropout:
            n_edges = edge_index.shape[1]
            rate = 0.5
            random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
            edge_index, edge_type = edge_index[:, random_indices], edge_type[random_indices]

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        for i in range(self.n_layers):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb, edge_index, edge_type, interact_mat,
                                                 self.weight, fast_weights, i=i)
            
            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
        return entity_res_emb, user_res_emb


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, user_pre_embed, item_pre_embed):
        super(Recommender, self).__init__()

        '''load info'''
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        # inner meta-learning update
        self.num_inner_update = args_config.num_inner_update
        self.meta_update_lr = args_config.meta_update_lr
        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.membership_type = args_config.membership_type
        self.num_rules = args_config.num_rules
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda else torch.device("cpu")
        self.user_pre_embed = user_pre_embed
        self.item_pre_embed = item_pre_embed
        
        '''get edge'''
        self.edge_index, self.edge_type = self._get_edges(graph)

        '''init weight'''
        self.all_embed = nn.init.xavier_uniform_(torch.empty(self.n_nodes, self.emb_size))

        if self.user_pre_embed != None and self.item_pre_embed != None:
            entity_emb = self.all_embed[(self.n_users + self.n_items):, :]
            self.all_embed = torch.cat([self.user_pre_embed, self.item_pre_embed, entity_emb])

        self.all_embed = nn.Parameter(self.all_embed)

        '''graph'''
        self.gcn = GraphConv(channel=self.emb_size,
                             n_hops=self.context_hops,
                             n_users=self.n_users,
                             n_relations=self.n_relations,
                             n_items=self.n_items,
                             n_entities=self.n_entities,
                             membership_type=self.membership_type,
                             num_rules=self.num_rules,
                             node_dropout_rate=self.node_dropout_rate,
                             mess_dropout_rate=self.mess_dropout_rate)

        self.interact_mat = None

    def _get_edges(self, graph):
        src, dst = graph.edges()
        type = graph.edata['type']
        return torch.stack([src, dst]).long().to(self.device), type.long().to(self.device)

    def get_parameter(self):
        param_dict = OrderedDict()

        for name, param in self.gcn.named_parameters():
            if 'fuzzy_ui.scales' in name or 'fuzzy_ui.centers' in name or 'fuzzy_ui.m' in name:
                param_dict[name] = param

        return param_dict

    def forward_kg(self, h, r, pos_t, neg_t):
        entity_emb = self.all_embed[self.n_users:, :]
        h_emb = entity_emb[h]
        r_emb = entity_emb[r]
        pos_t_emb = entity_emb[pos_t]
        neg_t_emb = entity_emb[neg_t]

        r_t_pos = pos_t_emb * r_emb
        r_t_neg = neg_t_emb * r_emb

        pos_score = torch.sum(torch.pow(r_t_pos - h_emb, 2), dim=1)
        neg_score = torch.sum(torch.pow(r_t_neg - h_emb, 2), dim=1)

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        return kg_loss

    def forward_meta(self, support, query, fast_weights=None):
        user_s = support[0]
        pos_item_s = support[1]
        neg_item_s = support[2]
        user_q = query[0]
        pos_item_q = query[1]
        neg_item_q = query[2]

        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]

        if fast_weights == None:
            fast_weights = self.get_parameter()

        def BPR_Loss(user, pos_item, neg_item):
            entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                    entity_emb,
                                                    self.edge_index,
                                                    self.edge_type,
                                                    self.interact_mat,
                                                    fast_weights=fast_weights,
                                                    mess_dropout=self.mess_dropout,
                                                    node_dropout=self.node_dropout)
            u = user_gcn_emb[user]
            pos, neg = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
            batch_size = u.shape[0]
            pos_scores = torch.sum(torch.mul(u, pos), axis=1)
            neg_scores = torch.sum(torch.mul(u, neg), axis=1)
            mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
            # cul regularizer
            regularizer = (torch.norm(u) ** 2
                           + torch.norm(pos) ** 2
                           + torch.norm(neg) ** 2) / 2
            emb_loss = self.decay * regularizer / batch_size
            return mf_loss + emb_loss

        loss = BPR_Loss(user_s, pos_item_s, neg_item_s)

        gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=False)

        fast_weights = OrderedDict(
            (name, param - self.meta_update_lr * grad)
            for ((name, param), grad) in zip(fast_weights.items(), gradients)
        )

        loss = BPR_Loss(user_q, pos_item_q, neg_item_q)
        return loss

    def forward(self, support):
        user_s = support[0]
        pos_item_s = support[1]
        neg_item_s = support[2]

        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                entity_emb,
                                                self.edge_index,
                                                self.edge_type,
                                                self.interact_mat,
                                                mess_dropout=self.mess_dropout,
                                                node_dropout=self.node_dropout)
        u = user_gcn_emb[user_s]
        pos, neg = entity_gcn_emb[pos_item_s], entity_gcn_emb[neg_item_s]
        batch_size = u.shape[0]
        pos_scores = torch.sum(torch.mul(u, pos), axis=1)
        neg_scores = torch.sum(torch.mul(u, neg), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        # cul regularizer
        regularizer = (torch.norm(u) ** 2
                       + torch.norm(pos) ** 2
                       + torch.norm(neg) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        loss = mf_loss + emb_loss

        u = user_gcn_emb[user_s]
        rating = torch.matmul(u, entity_gcn_emb[:self.n_items].t())

        return loss, rating

    def forward_cold(self, batch, pos_q):
        user = [sublist[0] for sublist in batch]
        pos_item = [sublist[1] for sublist in batch]
        neg_item = [sublist[2] for sublist in batch]

        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]

        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                entity_emb,
                                                self.edge_index,
                                                self.edge_type,
                                                self.interact_mat,
                                                mess_dropout=self.mess_dropout,
                                                node_dropout=self.node_dropout)

        all_loss = 0
        for i in range(len(user)):
            u = user_gcn_emb[user[i][0]]
            pos, neg = entity_gcn_emb[pos_item[i]], entity_gcn_emb[neg_item[i]]

            '''bprloss'''
            batch_size = u.shape[0]
            pos_scores = torch.sum(torch.mul(u, pos), axis=1)
            neg_scores = torch.sum(torch.mul(u, neg), axis=1)
            bprloss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

            '''l2'''
            regularizer = (torch.norm(u) ** 2
                           + torch.norm(pos) ** 2
                           + torch.norm(neg) ** 2) / 2
            emb_loss = self.decay * regularizer / batch_size

            '''ndcgloss'''
            rating = torch.matmul(u, entity_gcn_emb[:self.n_items].t())
            q = pos_q[i]
            all_items = set(range(0, self.n_items))
            test_items = list(all_items - set(q))
            r, auc = ranklist_by_heapq(pos_item[i], test_items, rating.tolist(), [20])
            res = get_performance(pos_item[i], r, auc, [20])
            ndcgloss = 1 - res['recall'][0]

            loss = 0.1 * bprloss + 0.9 * ndcgloss + emb_loss
            all_loss += loss

        return all_loss

    '''for evaluate'''
    def generate(self, adapt_fast_weight=None):
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                entity_emb,
                                                self.edge_index,
                                                self.edge_type,
                                                self.interact_mat,
                                                fast_weights=adapt_fast_weight,
                                                mess_dropout=False, node_dropout=False)

        return entity_gcn_emb, user_gcn_emb

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())