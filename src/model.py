import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, GPT2LMHeadModel, GPT2Tokenizer

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
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

class CTRModel(nn.Module):
    def __init__(self, args, n_node, n_relation):
        super(CTRModel, self).__init__()
        self._parse_args(args, n_node, n_relation)
        self._init_weight()
        self.node_emb = nn.Parameter(self.node_emb)
        self.relation_emb = nn.Parameter(self.relation_emb)
        self.attention = _MultiLayerPercep(2*self.dim, 1)
        self.use_cuda = args.use_cuda

    def forward(
        self,
        users: torch.LongTensor,
        items: torch.LongTensor,
        users_triple: list,
        items_triple: list,
    ):
        if(torch.isnan(self.node_emb).any()):
            print("all_emb:", self.node_emb)
        user_embeddings = []
        # [B, K, D]
        user_emb_origin = self.node_emb[users]
#         # [B, D]
#         user_embeddings.append(torch.sum(user_emb_origin, dim=1))
        # [B, K, D]
        user_embeddings.append(user_emb_origin)

        for i in range(self.n_layer):
            # [B, N, K, D]
            h_emb = self.node_emb[users_triple[0][i]]
            # [B, N, K, D]
            r_emb = self.relation_emb[users_triple[1][i]]
            # [B, N, K, D]
            t_emb = self.node_emb[users_triple[2][i]]
#             # [B, D]
#             user_emb_layer_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            # [B, K, D]
            user_emb_layer_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            user_embeddings.append(user_emb_layer_i)
        
        item_embeddings = []
        # [B, K, D]
        item_emb_origin = self.node_emb[items]
#         # [B, D]
#         item_embeddings.append(torch.sum(item_emb_origin, dim=1))
        # [B, K, D]
        item_embeddings.append(item_emb_origin)
        for i in range(self.n_layer):
            # [B, N, K, D]
            h_emb = self.node_emb[items_triple[0][i]]
            # [B, N, K, D]
            r_emb = self.relation_emb[items_triple[1][i]]
            # [B, N, K, D]
            t_emb = self.node_emb[items_triple[2][i]]
#             # [B, D]
#             item_emb_layer_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            # [B, K, D]
            item_emb_layer_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            item_embeddings.append(item_emb_layer_i)
                

        # scores, crossloss = self.predict(user_embeddings, item_embeddings)
        # return scores, crossloss
        return user_embeddings, item_embeddings
    
    def predict(self, user_embeddings, item_embeddings):
        #B, K, D
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]
    
        if self.agg == 'concat':
            for i in range(1, len(user_embeddings)):
                e_u = torch.cat((user_embeddings[i], e_u),dim=-1)
            for i in range(1, len(item_embeddings)):
                e_v = torch.cat((item_embeddings[i], e_v),dim=-1)
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
                gate_u = torch.sigmoid(torch.sum(torch.mul(user_embeddings[0], user_embeddings[i]), dim=1, keepdim=True))
                e_u = e_u + torch.mul(gate_u, user_embeddings[i])
            for i in range(1, len(item_embeddings)):
                gate_v = torch.sigmoid(torch.sum(torch.mul(item_embeddings[0], item_embeddings[i]), dim=1, keepdim=True))
                e_v = e_v + torch.mul(gate_v, item_embeddings[i])
        else:
            raise Exception("Unknown aggregator: " + self.agg)
            

        # B, D
        e_u_final = e_u.sum(dim = 1)
        e_v_final = e_v.sum(dim = 1)

        scores = torch.sigmoid((e_v_final * e_u_final).sum(dim=1))
        if(torch.isnan(scores).any()):
            print("scores",scores)
            
            
        
        #caculate crossloss
        tau = 0.6    # default = 0.8
        f = lambda x: torch.exp(x / tau)
        if self.use_cuda:
            e_u = e_u.cuda()
            e_v = e_v.cuda()
        
        #for user
        mlp_u = nn.Sequential(
                nn.Linear(self.dim, self.dim, bias=True),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim, bias=True),
                )
        # B, K, D
        if self.use_cuda:
            mlp_u = mlp_u.to('cuda')
        e_u = mlp_u(e_u)
        e_u = F.normalize(e_u, dim = 2)#normalize defalut dim=1
        #B, D, K
        e_u_T = torch.transpose(e_u, 1, 2)
        
        #B, K, K
        pos_sample = f(torch.matmul(e_u, e_u_T))
        #B, K
        pos_sample = pos_sample.sum(2)  # - torch.diagonal(pos_sample, 0, dim1=-2, dim2=-1) 
        #B, 1, D, K
        e_u_T = torch.unsqueeze(e_u_T, dim=1)
        #B, B, K, K       
        all_sample = f(torch.matmul(e_u, e_u_T))
        #B, B, K
        all_sample = all_sample.sum(3)
        #B, K
        all_sample = all_sample.sum(1)
        
        loss_u = - torch.log(pos_sample / all_sample)
        
        
        
#         #for item
        mlp_v = nn.Sequential(
                nn.Linear(self.dim, self.dim, bias=True),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim, bias=True),
                )
        # B, K, D
        if self.use_cuda:
            mlp_v = mlp_v.to('cuda')
        e_v = mlp_v(e_v)
        e_v = F.normalize(e_v, dim = 2)
        #B, D, K
        e_v_T = torch.transpose(e_v, 1, 2)
        
        #B, K, K
        pos_sample = f(torch.matmul(e_v, e_v_T))
        #B, K
        pos_sample = pos_sample.sum(2)  # - torch.diagonal(pos_sample, 0, dim1=-2, dim2=-1) 
        
        #B, 1, D, K
        e_v_T = torch.unsqueeze(e_v_T, dim=1)
        #B, B, K, K
        all_sample = f(torch.matmul(e_v, e_v_T))
        #B, B, K
        all_sample = all_sample.sum(3)
        #B, K
        all_sample = all_sample.sum(1)
        #B, K
        loss_v = - torch.log(pos_sample / all_sample)
        #B, K
        crossloss = loss_u + loss_v
        #B
        
        crossloss = crossloss.mean()
#         crossloss = crossloss.sum()

        return scores, crossloss
        
    
    def _init_weight(self):
        # init embedding
        initializer = nn.init.xavier_uniform_
        self.node_emb = initializer(torch.empty(self.n_node, self.n_factor, self.dim))
        self.relation_emb = initializer(torch.empty(self.n_relation, self.n_factor, self.dim))
    
    def _parse_args(self, args, n_node, n_relation):
        self.n_node = n_node
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_factor = args.n_factor
        self.n_layer = args.n_layer
        self.agg = args.agg
    
    def _knowledge_attention(self, h_emb, r_emb, t_emb):
        # [B, N, K]
        att_weights = self.attention(torch.cat((h_emb,r_emb),dim=-1)).squeeze(-1)
        # [B, N, K]
        att_weights_norm = F.softmax(att_weights,dim=1)
        # [B, N, K, D]
        att_emb = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)
        # [B, K, D]
        emb_i = att_emb.sum(dim=1)
        

        # [B, D]
#         emb = emb_i.sum(dim=1)
#         return emb
        return emb_i

def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

class MyModel(nn.Module):
    def __init__(self, args, n_nodes, n_relations, tokenizer):
        super(MyModel, self).__init__()
        self.mode = args.model_mode
        self.LM = BertModel.from_pretrained(args.lm_model)

        if args.extend_vocab == 'raw':
            # IMPORTANT! Resize token embeddings
            self.LM.resize_token_embeddings(len(tokenizer))

        self.CTR = CTRModel(args, n_nodes, n_relations)
        # self.promptGeneator = EmbeddingToTextModel(args)
        self.tokenizer = tokenizer

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.LM.config.hidden_size, 1)  # suit for BCELoss
        self.sigmoid = nn.Sigmoid()

        if args.model_mode != 'lm':
            freeze_model_parameters(self.LM)

    def resize_token_embeddings(self, length):
        self.LM.resize_token_embeddings(length)

    def forward(self, users, movies, user_neighbors, movie_neighbors, input_ids, attention_mask):
        if self.mode == 'ctr':
            # [layer_num, batch_size, dim]
            user_embeddings, item_embeddings = self.CTR(users, movies, user_neighbors, movie_neighbors)
            proba, _ = self.CTR.predict(user_embeddings, item_embeddings)
        elif self.mode == 'lm':
            LM_outputs = self.LM(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = LM_outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            logits = self.linear(pooled_output)
            proba = self.sigmoid(logits)
        elif self.mode == 'ctr4lm':
            # [layer_num, batch_size, dim]
            user_embeddings, item_embeddings = self.CTR(users, movies, user_neighbors, movie_neighbors)
            user_prompts = [self.promptGeneator(user_embeddings[i]) for i in range(self.CTR.n_layer)]
            item_prompts = [self.promptGeneator(item_embeddings[i]) for i in range(self.CTR.n_layer)]

            ctr_prompts = []
            for batch_idx in range(len(user_prompts[0])):
                user_ctr_prompt = "\n        - ".join(
                    ["Layer{}: {}".format(i, str(user_prompts[i][batch_idx]).strip()) for i in range(self.CTR.n_layer)])
                item_ctr_prompt = "\n        - ".join(
                    ["Layer{}: {}".format(i, str(item_prompts[i][batch_idx]).strip()) for i in range(self.CTR.n_layer)])
                ctr_prompt = """
                User Collaboration Information:
                - {}

                Movie Collaboration Information:
                - {}

                """.format(user_ctr_prompt, item_ctr_prompt)
                ctr_prompts.append(ctr_prompt)
            encoded_prompts = self.tokenizer(ctr_prompts, return_tensors='pt', padding=True, truncation=True)
            prompt_input_ids = encoded_prompts['input_ids']
            prompt_attention_mask = encoded_prompts['attention_mask']

            # print(f'Original input_ids length: {input_ids.size(1)}')
            # print(f'Original attention_mask length: {attention_mask.size(1)}')
            # print(f'Prompt input_ids length: {prompt_input_ids.size(1)}')
            # print(f'Prompt attention_mask length: {prompt_attention_mask.size(1)}')
            prompt_input_ids = prompt_input_ids.to(input_ids.device)
            prompt_attention_mask = prompt_attention_mask.to(input_ids.device)
            updated_input_ids = torch.cat([input_ids, prompt_input_ids], dim=1)
            updated_attention_mask = torch.cat([attention_mask, prompt_attention_mask], dim=1)

            max_length = 512
            if updated_input_ids.size(1) > max_length:
                print(
                    f"Warning: Input length {updated_input_ids.size(1)} exceeds the maximum length of {max_length}. Truncating.")
                updated_input_ids = updated_input_ids[:, :max_length]
                updated_attention_mask = updated_attention_mask[:, :max_length]
            LM_outputs = self.LM(input_ids=updated_input_ids, attention_mask=updated_attention_mask)
            pooled_output = LM_outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            logits = self.linear(pooled_output)
            proba = self.sigmoid(logits).squeeze()
        return proba