import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class MyModel(nn.Module):
    def __init__(self, args, n_nodes, n_relations):
        super(MyModel, self).__init__()
        self.LM = BertModel.from_pretrained(args.model_name_or_path)
        self.CTR = CTRModel(args, n_nodes, n_relations)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.LM.config.hidden_size, 2)
        self.sigmoid = nn.Sigmoid()

    def resize_token_embeddings(self, length):
        self.LM.resize_token_embeddings(length)

    def forward(self, users, movies, user_neighbors, movie_neighbors, input_ids, attention_mask):
        CTR_outputs = self.CTR(users, movies, user_neighbors, movie_neighbors)
        LM_outputs = self.LM(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = LM_outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        proba = self.sigmoid(logits)
        return proba


class _MultiLayerPercep(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_MultiLayerPercep, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=True),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim, bias=True),
        )
        self._init_weight()

    def forward(self, x):
        return self.mlp(x)

    def _init_weight(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)


class CTRModel(nn.Module):
    def __init__(self, args, n_node, n_relation):
        super(CTRModel, self).__init__()
        self._parse_args(args, n_node, n_relation)
        self._init_weight()
        self.node_emb = nn.Parameter(self.node_emb)  # 将一个不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面
        self.relation_emb = nn.Parameter(self.relation_emb)
        self.attention = _MultiLayerPercep(2 * self.dim, 1)  # 注意力为一个双层感知机

    def forward(
            self,
            users: torch.Tensor,            # [batch_size]
            items: torch.Tensor,            # [batch_size]
            users_triple: torch.Tensor,     # [batch_size, [h|r|t], n_layer, neighbor_size]
            items_triple: torch.Tensor,     # [batch_size, [h|r|t], n_layer, neighbor_size]
    ):
        user_embeddings = []
        # [B, K, D]
        user_emb_origin = self.node_emb[users]  # Tensor([batch_size, n_factor=2, dim=64])
        # [B, D]
        # user_embeddings.append(torch.sum(user_emb_origin, dim=1))     # Tensor([batch_size, dim=64])
        # [B, K, D]
        user_embeddings.append(user_emb_origin)

        # transpose from [batch_size, [h|r|t], n_layer, neighbor_size] to [[h|r|t], n_layer, batch_size, neighbor_size]
        users_triple = users_triple.permute(1, 2, 0, 3)
        for i in range(self.n_layer):
            # [B, N, K, D]
            h_emb = self.node_emb[users_triple[0][i]]  # Tensor([batch_size, neighbor_size=64, n_factor=2, dim=64])
            # [B, N, K, D]
            r_emb = self.relation_emb[users_triple[1][i]]  # Tensor([batch_size, neighbor_size=64, n_factor=2, dim=64])
            # [B, N, K, D]
            t_emb = self.node_emb[users_triple[2][i]]  # Tensor([batch_size, neighbor_size=64, n_factor=2, dim=64])
            # [B, K, D]
            user_emb_layer_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            user_embeddings.append(user_emb_layer_i)

        item_embeddings = []
        # [B, K, D]
        item_emb_origin = self.node_emb[items]
        # [B, D]
        # item_embeddings.append(torch.sum(item_emb_origin, dim=1))
        # [B, K, D]
        item_embeddings.append(item_emb_origin)

        # transpose from [batch_size, [h|r|t], n_layer, neighbor_size] to [[h|r|t], n_layer, batch_size, neighbor_size]
        items_triple = items_triple.permute(1, 2, 0, 3)
        for i in range(self.n_layer):
            # [B, N, K, D]
            h_emb = self.node_emb[items_triple[0][i]]
            # [B, N, K, D]
            r_emb = self.relation_emb[items_triple[1][i]]
            # [B, N, K, D]
            t_emb = self.node_emb[items_triple[2][i]]
            # [B, D]
            item_emb_layer_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            item_embeddings.append(item_emb_layer_i)

        # scores = self.predict(user_embeddings, item_embeddings)
        # scores, crossloss = self.predict_with_crossloss(user_embeddings,
        #                                                 item_embeddings)  # Tensor([batch_size, n_factor=2, dim=64])
        # return scores, crossloss
        return user_embeddings, item_embeddings

    def caculate_rec_score(self, user_embeddings, item_embeddings):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]

        if self.agg == 'concat':
            for i in range(1, len(user_embeddings)):
                e_u = torch.cat((user_embeddings[i], e_u), dim=-1)
            for i in range(1, len(item_embeddings)):
                e_v = torch.cat((item_embeddings[i], e_v), dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(user_embeddings)):
                e_u += user_embeddings[i]
            for i in range(1, len(item_embeddings)):
                e_v += item_embeddings[i]
        elif self.agg == 'pool':
            for i in range(1, len(user_embeddings)):
                e_u = torch.max(e_u, user_embeddings[i])
            for i in range(1, len(item_embeddings)):
                e_v = torch.max(e_v, item_embeddings[i])
        elif self.agg == 'decay_sum':
            for i in range(1, len(user_embeddings)):
                gate_u = torch.sigmoid(
                    torch.sum(torch.mul(user_embeddings[0], user_embeddings[i]), dim=1, keepdim=True))
                e_u = e_u + torch.mul(gate_u, user_embeddings[i])
            for i in range(1, len(item_embeddings)):
                gate_v = torch.sigmoid(
                    torch.sum(torch.mul(item_embeddings[0], item_embeddings[i]), dim=1, keepdim=True))
                e_v = e_v + torch.mul(gate_v, item_embeddings[i])
        else:
            raise Exception("Unknown aggregator: " + self.agg)

        # scores = torch.sigmoid((e_v * e_u).sum(dim=1))
        return e_u, e_v

    def predict(self, user_embeddings, item_embeddings):
        e_u, e_v = self.caculate_rec_score(user_embeddings, item_embeddings)
        scores = torch.sigmoid((e_v * e_u).sum(dim=1))
        return scores

    def predict_with_crossloss(self, user_embeddings, item_embeddings):
        # B, K, D
        e_u, e_v = self.caculate_rec_score(user_embeddings, item_embeddings)
        scores = torch.sigmoid((e_v * e_u).sum(dim=1))

        # B, D
        e_u_final = e_u.sum(dim=1)  # Tensor(batch_size, dim)
        e_v_final = e_v.sum(dim=1)

        scores = torch.sigmoid((e_v_final * e_u_final).sum(dim=1))
        if (torch.isnan(scores).any()):
            print("scores", scores)

            # caculate crossloss
        tau = 0.6  # default = 0.8
        f = lambda x: torch.exp(x / tau)
        if self.use_cuda:
            e_u = e_u.cuda()
            e_v = e_v.cuda()

        # for user
        mlp_u = nn.Sequential(
            nn.Linear(self.dim, self.dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim, bias=True),
        )
        # B, K, D
        if self.use_cuda:
            mlp_u = mlp_u.to('cuda')
        e_u = mlp_u(e_u)
        e_u = F.normalize(e_u, dim=2)  # normalize default dim=1
        # B, D, K
        e_u_T = torch.transpose(e_u, 1, 2)

        # B, K, K
        pos_sample = f(torch.matmul(e_u, e_u_T))
        # B, K
        pos_sample = pos_sample.sum(2)  # - torch.diagonal(pos_sample, 0, dim1=-2, dim2=-1)
        # B, 1, D, K
        e_u_T = torch.unsqueeze(e_u_T, dim=1)
        # B, B, K, K
        all_sample = f(torch.matmul(e_u, e_u_T))
        # B, B, K
        all_sample = all_sample.sum(3)
        # B, K
        all_sample = all_sample.sum(1)

        loss_u = - torch.log(pos_sample / all_sample)

        # for item
        mlp_v = nn.Sequential(
            nn.Linear(self.dim, self.dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim, bias=True),
        )
        # B, K, D
        if self.use_cuda:
            mlp_v = mlp_v.to('cuda')
        e_v = mlp_v(e_v)
        e_v = F.normalize(e_v, dim=2)
        # B, D, K
        e_v_T = torch.transpose(e_v, 1, 2)

        # B, K, K
        pos_sample = f(torch.matmul(e_v, e_v_T))
        # B, K
        pos_sample = pos_sample.sum(2)  # - torch.diagonal(pos_sample, 0, dim1=-2, dim2=-1)

        # B, 1, D, K
        e_v_T = torch.unsqueeze(e_v_T, dim=1)
        # B, B, K, K
        all_sample = f(torch.matmul(e_v, e_v_T))
        # B, B, K
        all_sample = all_sample.sum(3)
        # B, K
        all_sample = all_sample.sum(1)
        # B, K
        loss_v = - torch.log(pos_sample / all_sample)

        # B, K
        crossloss = loss_u + loss_v
        # B
        crossloss = crossloss.mean()
        #         crossloss = crossloss.sum()

        return scores, crossloss

    def _init_weight(self):
        # init embedding
        initializer = nn.init.xavier_uniform_
        # 创建n_node * n_factor * dim的初始化张量
        self.node_emb = initializer(torch.randn(self.n_node, self.n_factor, self.dim))
        # 创建n_relation * n_factor * dim的初始化张量
        self.relation_emb = initializer(torch.randn(self.n_relation, self.n_factor, self.dim))

    def _parse_args(self, args, n_node, n_relation):
        self.n_node = n_node
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_factor = args.n_factor
        self.n_layer = args.n_layer
        self.agg = args.agg

    def _knowledge_attention(self, h_emb, r_emb, t_emb):
        # [B, N, K, D]
        att_input = torch.cat([h_emb, t_emb], dim=-1)
        # [B, N, K]
        att_weights = self.attention(att_input).squeeze(-1)  # Tensor([batch_size, neighbor_size=64, n_factor=2])
        att_weights_norm = F.softmax(att_weights, dim=1)  # Tensor([batch_size, neighbor_size=64, n_factor=2])
        # [B, N, K, D]
        att_emb = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)  # Tensor([batch_size, neighbor_size=64, n_factor=2])
        # [B, K, D] <- Aggregated neighbor information
        emb_i = att_emb.sum(dim=1)  # Tensor([batch_size, n_factor=2, dim=64])

        # k_weights_norm = F.softmax(emb_i, dim=1)
        # [B, D]
        # emb = torch.sum(torch.mul(emb_i, k_weights_norm),dim=1)
        # emb = emb_i.sum(dim=1)  # Tensor([batch_size, dim=64])

        return emb_i
