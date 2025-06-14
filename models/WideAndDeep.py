import torch
import torch.nn as nn
import torch.nn.functional as F

class WideAndDeep(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256):
        super(WideAndDeep, self).__init__()

        # Wide part (linear model for continuous attributes)
        self.wide_fc = nn.Linear(5, embedding_dim)

        # Deep part (neural network for categorical attributes)
        self.depature_embedding = nn.Embedding(288+1, hidden_dim)
        self.sid_embedding = nn.Embedding(256+1, hidden_dim)
        self.eid_embedding = nn.Embedding(256+1, hidden_dim)
        self.deep_fc1 = nn.Linear(hidden_dim*3, embedding_dim)
        self.deep_fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, attr, cond_drop_prob):
        # Continuous attributes
        continuous_attrs = attr[:, 1:6]

        # Categorical attributes
        depature, sid, eid = attr[:, 0].long(
        ), attr[:, 6].long(), attr[:, 7].long()

        # randomly drop some conditions
        if cond_drop_prob > 0:
            drop_ids = torch.rand(attr.shape[0], device=attr.device) < cond_drop_prob
            continuous_attrs[drop_ids,:] = 0 # 0 for drop
            depature[drop_ids] = 288 # last index for drop
            sid[drop_ids] = 256 # last index for drop
            eid[drop_ids] = 256 # last index for drop

        # Wide part
        wide_out = self.wide_fc(continuous_attrs)

        # Deep part
        depature_embed = self.depature_embedding(depature)
        sid_embed = self.sid_embedding(sid)
        eid_embed = self.eid_embedding(eid)
        categorical_embed = torch.cat(
            (depature_embed, sid_embed, eid_embed), dim=1)
        deep_out = F.relu(self.deep_fc1(categorical_embed))
        deep_out = self.deep_fc2(deep_out)
        # Combine wide and deep embeddings
        combined_embed = wide_out + deep_out

        return combined_embed