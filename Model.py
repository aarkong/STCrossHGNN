import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed(42)

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, BatchNorm
from torch.nn.functional import cross_entropy
from dhg.models import HGNN
from dhg import Hypergraph
import copy

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)  # 新增LayerNorm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(0)          # [1, N, dim]
        x2 = x2.unsqueeze(0)          # [1, N, dim]
        attn_out, attn_weights = self.attention(x1, x2, x2)
        attn_out = attn_out.squeeze(0)
        attn_out = self.dropout(attn_out)
        return self.norm(attn_out + x1.squeeze(0))

class HGNNClassifier(nn.Module):
    def __init__(self, num_features, num_classes, dropout=0.5, time_steps=8, device='cpu',  
                 lstm_hidden=64, hgnn_hid_dim=256, mask_rate=0.3, aug_ratio=0.15, seed=42):
        super(HGNNClassifier, self).__init__()
        self.device = device
        self.lstm_hidden = lstm_hidden
        self.hgnn_hid_dim = hgnn_hid_dim
        self.mask_rate = mask_rate
        self.time_steps = time_steps
        self.aug_ratio = aug_ratio 
        self.input_size = self.lstm_hidden // time_steps
        self.seed = seed

        self.feature_proj = nn.Linear(num_features, time_steps * self.input_size)

        self.lstm = nn.GRU(
            input_size=self.input_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )

        self.hgnn1 = HGNN(
            in_channels=self.lstm_hidden*2,
            hid_channels=self.hgnn_hid_dim,
            num_classes=self.hgnn_hid_dim,
            use_bn=True,
            drop_rate=dropout
        )
        self.hgnn2 = HGNN(
            in_channels=self.hgnn_hid_dim,
            hid_channels=self.hgnn_hid_dim,
            num_classes=self.hgnn_hid_dim,
            use_bn=True,
            drop_rate=dropout
        )

        self.bn_hgnn = nn.BatchNorm1d(self.hgnn_hid_dim)
        self.relu = nn.ReLU()
        self.bn_gru = nn.BatchNorm1d(self.lstm_hidden*2)

        self.node_proj = nn.Sequential(
            nn.Linear(hgnn_hid_dim, hgnn_hid_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hgnn_hid_dim, hgnn_hid_dim)
        )

        self.decoder = nn.Linear(hgnn_hid_dim, hgnn_hid_dim)
        torch.manual_seed(42)
        self.mask_token = nn.Parameter(torch.zeros(1, num_features, device=self.device))
        self.feature_proj = nn.Linear(num_features, time_steps * self.input_size).to(self.device)
        self.gru2hid = nn.Linear(self.lstm_hidden*2, self.hgnn_hid_dim)
        self.cross   = CrossAttention(self.hgnn_hid_dim)
        self.fuse    = nn.Linear(self.hgnn_hid_dim*2, self.hgnn_hid_dim)
        self.cross_norm = nn.LayerNorm(self.hgnn_hid_dim)
        self.cross_drop  = nn.Dropout(0.3)
        self.classifier = nn.Linear(hgnn_hid_dim, num_classes)
        self.register_buffer('class_weight', torch.tensor([0.3, 0.7], device=self.device))

    def cnt_node_occurrence(self, edge_index):
        return torch.bincount(edge_index[0].cpu()).tolist()

    def find_first_two_indices(self, tensor, aug_ratio, node_occurrence, node_indices):
        hyperedge_ids, counts = torch.unique(tensor, return_counts=True)
        mask = counts > 2
        hyperedge_ids = hyperedge_ids[mask]
        counts = counts[mask]
        num_to_remove = (counts.float() * aug_ratio).clamp(min=1).long()

        node_counts = torch.tensor(node_occurrence, device=tensor.device)[node_indices]
        generator = torch.Generator(device=tensor.device).manual_seed(42)
        torch.manual_seed(self.seed)
        small_noise = torch.rand_like(node_counts.float()) * 1e-8
        sorted_indices = (node_counts + small_noise).argsort(descending=True)
        safe_indices = sorted_indices[node_counts[sorted_indices] > 1]
        return safe_indices[:num_to_remove.sum()].tolist()

    def permute_edges(self, edge_index, aug_ratio=None):
        if aug_ratio is None:
            aug_ratio = self.aug_ratio

        edge_index = edge_index.clone()
        node_num = int(edge_index[0].max() + 1)
        
        node_occurrence = self.cnt_node_occurrence(edge_index)
        remove_index = self.find_first_two_indices(
            edge_index[1], aug_ratio, node_occurrence, edge_index[0]
        )

        mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
        mask[remove_index] = False
        return edge_index[:, mask]
        
        keep_index = [idx for idx in range(len(edge_index[1])) if idx not in remove_index]
        edge_after_remove = edge_index[:, keep_index]

        return edge_after_remove

    def mask_features(self, x, mask_ratio=0.1):
        num_nodes = x.size(0)
        mask = torch.ones(num_nodes, device=x.device, dtype=torch.bool)
        num_mask = int(num_nodes * mask_ratio)
        
        cpu_generator = torch.Generator('cpu').manual_seed(self.seed)
        perm = torch.randperm(num_nodes, generator=cpu_generator)
        
        mask[perm[:num_mask]] = False
        x_aug = x * mask.unsqueeze(1)
        return x_aug, mask

    def contrastive_loss(self, z1, z2, temperature=1.0):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        sim = torch.matmul(z1, z2.T) / temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)

    def data_augmentation(self, edge_index, x, aug_ratio=None):
        if aug_ratio is None:
            aug_ratio = self.aug_ratio

        aug_edge_index = self.permute_edges(edge_index, aug_ratio)

        x_aug, mask = self.mask_features(x)
        
        return aug_edge_index, x_aug, mask

    def recon_loss(self, h_time2spat, h_spat2time, h_aug_spat, h_aug_time):
        target_time2spat = h_time2spat.detach()
        target_spat2time = h_spat2time.detach()
        
        loss_time = F.mse_loss(h_aug_spat, target_time2spat)
        loss_spat = F.mse_loss(h_aug_time, target_spat2time)
        
        return (loss_time + loss_spat) / 2

    def contra_loss(self, h_time2spat, h_spat2time, h_aug_spat, h_aug_time):
        return self.contrastive_loss(h_time2spat, h_aug_spat) + self.contrastive_loss(h_spat2time, h_aug_time)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if self.training:
            aug_edge_index, x_aug, mask = self.data_augmentation(edge_index, x)
        else:
            aug_edge_index = edge_index
            x_aug = x
            mask = None

        aug_edge_index = aug_edge_index - aug_edge_index.min()
        num_v = x.size(0)
        assert aug_edge_index.max().item() < num_v
        
        edge_list = aug_edge_index.t().cpu().tolist()
        hg = Hypergraph(num_v, edge_list)
        hg = hg.to(x.device)

        x = self.feature_proj(x)
        x = x.view(-1, self.time_steps, self.input_size)
        gru_out, _ = self.lstm(x)
        gru_feat = gru_out.mean(dim=1)
        gru_feat = self.bn_gru(gru_feat)
        assert not torch.isnan(gru_feat).any(), "GRU output contains NaN"

        with torch.cuda.amp.autocast(enabled=False):
            h_spat = self.hgnn1(gru_feat.float(), hg)
            assert not torch.isnan(h_spat).any(), "HGNN1 output contains NaN"
            
            h_spat = self.bn_hgnn(h_spat)
            h_spat = self.relu(h_spat)
            assert not torch.isnan(h_spat).any(), "BN+ReLU after HGNN1 contains NaN"
            
            h_spat = self.hgnn2(h_spat, hg)
            assert not torch.isnan(h_spat).any(), "HGNN2 output contains NaN"
        h_spat = self.node_proj(h_spat)

        gru_128 = self.gru2hid(gru_feat)
        h_time2spat = self.cross(gru_128, h_spat)
        h_spat2time = self.cross(h_spat, gru_128)

        h_time2spat = self.cross_drop(self.cross_norm(h_time2spat))
        h_spat2time = self.cross_drop(self.cross_norm(h_spat2time))

        h_fused = torch.cat([h_time2spat, h_spat2time], dim=-1)
        h_final = self.fuse(h_fused)

        x_graph = global_mean_pool(h_final, data.batch)
        logits = self.classifier(x_graph)

        x_aug = self.feature_proj(x_aug)
        x_aug = x_aug.view(-1, self.time_steps, self.input_size)
        gru_aug_out, _ = self.lstm(x_aug)
        gru_aug = gru_aug_out.mean(dim=1)

        with torch.cuda.amp.autocast(enabled=False):
            h_aug_spat = self.hgnn1(gru_aug.float(), hg)
            assert not torch.isnan(h_aug_spat).any(), "Augmented HGNN1 output contains NaN"
            
            h_aug_spat = self.bn_hgnn(h_aug_spat)
            h_aug_spat = self.relu(h_aug_spat)
            assert not torch.isnan(h_aug_spat).any(), "Augmented BN+ReLU after HGNN1 contains NaN"
            
            h_aug_spat = self.hgnn2(h_aug_spat, hg)
            assert not torch.isnan(h_aug_spat).any(), "Augmented HGNN2 output contains NaN"
        h_aug_spat = self.node_proj(h_aug_spat)

        gru_aug_128 = self.gru2hid(gru_aug)
        h_aug_time = gru_aug_128
        h_aug_t2s = self.cross(h_aug_time, h_aug_spat)
        h_aug_s2t = self.cross(h_aug_spat, h_aug_time)
        h_aug_t2s = self.cross_drop(self.cross_norm(h_aug_t2s))
        h_aug_s2t = self.cross_drop(self.cross_norm(h_aug_s2t))
        h_aug_fused = torch.cat([h_aug_t2s, h_aug_s2t], dim=-1)
        h_aug_final = self.fuse(h_aug_fused)

        recon_loss = torch.tensor(0.0, device=x.device)
        contra_loss = torch.tensor(0.0, device=x.device)
        if self.training and mask is not None:
            x_recon = self.decoder(h_aug_final[mask])
            target = self.gru2hid(gru_feat[mask]).detach()
            recon_loss = F.mse_loss(x_recon, target)

        cls_loss = F.cross_entropy(logits, data.y, weight=self.class_weight)
        if self.training:
            recon_loss = self.recon_loss(h_time2spat, h_spat2time, h_aug_spat, h_aug_time)
            contra_loss = self.contra_loss(h_time2spat, h_spat2time, h_aug_spat, h_aug_time)

        loss = cls_loss +  0.02 * (recon_loss + contra_loss)

        if self.training:
            return logits, loss
        else:
            return logits