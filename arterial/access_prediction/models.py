#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool, global_max_pool, BatchNorm

class GATv2Layer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=None, dropout_rate=0.2, attn_heads=1, concat=False):
        super(GATv2Layer, self).__init__()
        self.concat = concat
        self.edge_dim = edge_dim
        self.attn_heads = attn_heads

        self.conv = GATv2Conv(in_channels=in_channels, out_channels=out_channels, edge_dim=self.edge_dim, heads=self.attn_heads, concat=self.concat)
        self.bn1 = BatchNorm(out_channels * self.attn_heads if self.concat else out_channels)
        self.ff = nn.Linear(out_channels * self.attn_heads if self.concat else out_channels, out_channels)
        self.bn2 = BatchNorm(out_channels)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr=None):
        if self.edge_dim is not None:
            x = self.conv(x, edge_index, edge_attr)
            attention_weights = None
        else:
            x, attention_weights = self.conv(x, edge_index, return_attention_weights=True)
        
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn1(x)
        
        # Apply ff layer in both cases
        x = self.ff(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn2(x)
        
        x = self.dropout(x)
        return x, attention_weights

class MLPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(MLPLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = BatchNorm(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class ArterialGNet(nn.Module):
    """
    ArterialGNet.
    
    """
    def __init__(
            self,
            global_in_dim,
            segment_node_in_dim,
            segment_edge_in_dim,
            dense_node_in_dim,
            hidden_dim=8,
            hidden_dim_dense=8,
            out_dim=2,
            num_global_layers=1,
            num_segment_layers=1,
            num_dense_layers=1,
            num_out_layers=1,
            attn_heads=1,
            aggregation="mean",
            dropout=0.2,
            concat=False,
            is_classification=True):
        super(ArterialGNet, self).__init__()
        self.global_in_dim = global_in_dim
        self.segment_node_in_dim = segment_node_in_dim
        self.segment_edge_in_dim = segment_edge_in_dim
        self.dense_node_in_dim = dense_node_in_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_dense = hidden_dim_dense
        self.out_dim = out_dim
        self.num_global_layers = num_global_layers
        self.num_segment_layers = num_segment_layers
        self.num_dense_layers = num_dense_layers
        self.num_out_layers = num_out_layers
        self.attn_heads = attn_heads
        self.aggregation_ = aggregation 
        self.dropout = dropout
        self.concat = concat
        self.is_classification = is_classification
        self.attention_weights = None

        if self.num_global_layers > 0:
            self.global_path = torch.nn.ModuleList()
            for idx in range(self.num_global_layers):
                self.global_path.append(MLPLayer(self.global_in_dim if idx == 0 else self.hidden_dim, self.hidden_dim, dropout_rate=self.dropout))
        else:
            self.global_path = None
        
        if self.num_segment_layers > 0:
            self.segment_path = torch.nn.ModuleList()
            for idx in range(self.num_segment_layers):
                in_channels = self.segment_node_in_dim if idx == 0 else self.hidden_dim
                self.segment_path.append(GATv2Layer(in_channels, self.hidden_dim, edge_dim=self.segment_edge_in_dim, dropout_rate=self.dropout, attn_heads=self.attn_heads, concat=self.concat))
        else:
            self.segment_path = None
        
        if self.num_dense_layers > 0:
            self.dense_path = torch.nn.ModuleList()
            for idx in range(self.num_dense_layers):
                in_channels = self.dense_node_in_dim if idx == 0 else self.hidden_dim_dense
                self.dense_path.append(GATv2Layer(in_channels, self.hidden_dim_dense, dropout_rate=self.dropout, attn_heads=self.attn_heads, concat=self.concat))
        else:
            self.dense_path = None
        
        self.output_path = torch.nn.ModuleList()
        if self.num_out_layers > 1:
            for idx in range(num_out_layers - 1):
                self.output_path.append(MLPLayer(self.hidden_dim * sum([self.num_global_layers > 0, self.num_segment_layers > 0]) + self.hidden_dim_dense * sum([self.num_dense_layers > 0]) if idx == 0 else hidden_dim, self.hidden_dim, dropout_rate=self.dropout))
            self.output_path.append(nn.Linear(self.hidden_dim, self.out_dim))
        else:
            self.output_path.append(nn.Linear(self.hidden_dim * sum([self.num_global_layers > 0, self.num_segment_layers > 0]) + self.hidden_dim_dense * sum([self.num_dense_layers > 0]), self.out_dim))
        
        if self.is_classification:
            self.output_path.append(nn.Softmax(dim=1))

        if self.aggregation_ == "mean":
            self.aggregation = global_mean_pool
        elif self.aggregation_ == "max":
            self.aggregation = global_max_pool
        elif self.aggregation_ == "add":
            self.aggregation = global_add_pool
        else:
            raise ValueError(f"Unknown aggregation {self.aggregation_!r}; expected 'mean', 'max' or 'add'")

    def forward(self, data):
        global_data, segment_data, dense_data = data.global_data, data.segment_data, data.dense_data
        
        # Process global features
        if self.global_path is not None:
            global_features = global_data
            for layer in self.global_path:
                global_features = layer(global_features)
        else:
            global_features = None

        # Process segment_data
        segment_x = segment_data.x
        if self.segment_path is not None:
            for layer in self.segment_path:
                segment_x, _ = layer(segment_x, segment_data.edge_index, segment_data.edge_attr)
            segment_x = self.aggregation(segment_x, segment_data.batch)
        else:
            segment_x = None

        # Process dense_graph
        dense_x = dense_data.x
        if self.dense_path is not None:
            for idx, layer in enumerate(self.dense_path):
                if idx == 0:  # Only extract attention weights from the first layer
                    dense_x, attention_weights = layer(dense_x, dense_data.edge_index)
                    self.attention_weights = attention_weights
                else:
                    dense_x, _ = layer(dense_x, dense_data.edge_index)
            dense_x = self.aggregation(dense_x, dense_data.batch)
        else:
            dense_x = None

        # Concatenate all features
        out = torch.cat([x for x in [global_features, segment_x, dense_x] if x is not None], dim=1)

        # Final output layers
        for layer in self.output_path:
            out = layer(out)

        return out, self.attention_weights