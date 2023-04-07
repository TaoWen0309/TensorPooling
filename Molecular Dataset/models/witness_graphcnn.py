import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv

import sys
sys.path.append("models/")
from mlp import MLP
from cnn import CNN, cnn_output_dim
from diagram import sublevel_persistence_diagram, persistence_images, sum_diag_from_point_cloud

from torch.autograd import Variable
import numpy as np
import networkx as nx

from tltorch import TRL, TCL, FactorizedTensor # FactorizedLinear

class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, sublevel_filtration_methods, tensor_decom_type, tensor_layer_type, PI_dim, device):
        '''
            num_layers: number of GCN layers (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            sublevel_filtration_methods: methods for sublevel filtration on PD
            decom_type: Tensor decomposition type, Tucker/CP/TT
            tensor_layer: Tensor layer type, TCL/TRL'
            PI_dim: int size of PI
            device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.num_neighbors = 5

        # point clouds pooling for nodes
        self.score_node_layer = GCNConv(input_dim, self.num_neighbors * 2)
        
        # GCN block = gcn + mlp + batch_norm + relu
        self.GCNs = torch.nn.ModuleList()
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers-1):
            if layer == 0:
                self.GCNs.append(GCNConv(input_dim,hidden_dim))
            else:
                self.GCNs.append(GCNConv(hidden_dim,hidden_dim))
            self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # PI tensor block
        self.sublevel_filtration_methods = sublevel_filtration_methods
        # first a CNN
        self.cnn = CNN(hidden_dim)
        cnn_output_shape = cnn_output_dim(PI_dim)
        # then tensor decomposition
        self.tensor_decom_type = tensor_decom_type
        # finally a tensor layer
        self.PI_dim = PI_dim
        self.tensor_input_shape = (hidden_dim,cnn_output_shape,cnn_output_shape)
        self.tensor_hidden_shape = [hidden_dim,hidden_dim,hidden_dim] # for now set all dim as hidden_dim for convenience!
        # self.tensor_output_shape = (2,2)
        if tensor_layer_type == 'TCL':
            self.tensor_layer = TCL(self.tensor_input_shape,self.tensor_hidden_shape)
        elif tensor_layer_type == 'TRL':
            self.tensor_layer = TRL(self.tensor_input_shape,self.tensor_hidden_shape)
        # self.tensor_linear = FactorizedLinear(in_tensorized_features=tuple(self.tensor_hidden_shape),out_tensorized_features=self.tensor_output_shape)

        # TODO: output layer
        # for attentional second-order pooling
        # self.dense_dim = hidden_dim * (num_layers - 1)
        # self.attend = nn.Linear(self.dense_dim, 1)
        # self.linear1 = nn.Linear(self.dense_dim, output_dim)
        # self.mlp_PI_witnesses = nn.Linear(2500, self.PI_hidden_dim)

    def compute_batch_feat_PI_tensor(self, batch_graph):
        edge_attr = None
        edge_mat_list = []
        start_idx = [0]
        pooled_x = []
        pooled_graph_sizes = []
        PI_list = []

        for i, graph in enumerate(batch_graph):
            x = graph.node_features.to(self.device)
            edge_index = graph.edge_mat.to(self.device)
            adj = nx.adjacency_matrix(graph.g).todense()
            PI_list_i = []
            
            # PI tensor
            for j in range(len(self.sublevel_filtration_methods)): # 5 methods
                pd = sublevel_persistence_diagram(adj,self.PI_dim,self.sublevel_filtration_methods[j])
                pi = torch.FloatTensor(persistence_images(pd)).to(self.device)
                PI_list_i.append(pi)
            PI_tensor_i = torch.stack(PI_list_i).to(self.device)
            PI_list.append(PI_tensor_i)
            
            # graph pooling based on node topological scores
            node_embeddings = self.score_node_layer(x, edge_index).to(self.device)
            node_point_clouds = node_embeddings.view(-1, self.num_neighbors, 2).to(self.device)
            score_lifespan = torch.FloatTensor([sum_diag_from_point_cloud(node_point_clouds[i,...]) for i in range(node_point_clouds.size(0))]).to(self.device)
            batch = torch.LongTensor([0] * x.size(0)).to(self.device)
            perm = topk(score_lifespan, 0.5, batch)
            x = x[perm]
            edge_index, _ = filter_adj(edge_index, edge_attr, perm, num_nodes=graph.node_features.size(0))
            
            start_idx.append(start_idx[i] + x.size(0))
            edge_mat_list.append(edge_index + start_idx[i])
            pooled_x.append(x)
            pooled_graph_sizes.append(x.size(0))

        PI_concat = torch.stack(PI_list).to(self.device)
        pooled_X_concat = torch.cat(pooled_x, 0).to(self.device)
        Adj_block_idx = torch.cat(edge_mat_list, 1).to(self.device)
        return Adj_block_idx, pooled_X_concat, pooled_graph_sizes, PI_concat

    def GCN_layer(self, h, edge_index, layer):
        h = self.GCNs[layer](h, edge_index)
        h = self.mlps[layer](h)
        h = self.batch_norms[layer](h)
        h = F.relu(h)
        return h

    def forward(self, batch_graph):

        Adj_block_idx, pooled_X_concat, pooled_graph_sizes, PI_concat = self.compute_batch_feat_PI_tensor(batch_graph)
        
        # PI block
        # CNN
        PI_emb = self.cnn(PI_concat) # before cnn: [batch_size,5,PI_dim,PI_dim]; after cnn: [batch_size,hidden_dim,cnn_output_shape,cnn_output_shape]
        # tensor decomposition
        PI_decom =  FactorizedTensor.from_tensor(PI_emb, rank='same', factorization=self.tensor_decom_type).to(self.device) #[batch_size,hidden_dim,,cnn_output_shape,cnn_output_shape]
        # tensor layer
        PI_hidden = self.tensor_layer(PI_decom).to(self.device) # [batch_size,hidden_dim,hidden_dim,hidden_dim]

        # GCN block
        hidden_rep = []
        h = pooled_X_concat
        edge_index = Adj_block_idx
        for layer in range(self.num_layers-1):
            h = self.GCN_layer(h, edge_index, layer) # shape: [start_idx[-1]=N,hidden_dim]
            hidden_rep.append(h)
        # batch GCN tensor
        hidden_rep = torch.stack(hidden_rep).transpose(0,1) # shape: [start_idx[-1]=N, self.num_layers-1, hidden_dim]

        # after graph pooling
        graph_sizes = pooled_graph_sizes

        # batch_graphs = torch.zeros(len(graph_sizes), self.dense_dim - self.PI_hidden_dim).to(self.device)
        # batch_graphs = Variable(batch_graphs)
        # batch_graphs_out = torch.zeros(len(graph_sizes), self.dense_dim).to(self.device)
        # batch_graphs_out = Variable(batch_graphs_out)

        node_embeddings = torch.split(hidden_rep, graph_sizes, dim=0)
        for g_i in range(len(graph_sizes)):
            # current graph GCN tensor
            cur_node_embeddings = node_embeddings[g_i] # (n,self.num_layers-1,hidden_dim)
            # TODO: fix n to input to a tensor layer
            # TODO: concat PI tensor and GCN tensor and apply attention
            # TODO: output layer
            exit(0)
            attn_coef = self.attend(cur_node_embeddings)
            attn_weights = torch.transpose(attn_coef, 0, 1)
            cur_graph_embeddings = torch.matmul(attn_weights, cur_node_embeddings)
            batch_graphs[g_i] = cur_graph_embeddings.view(self.dense_dim - self.PI_hidden_dim)

            witnesses_PI_out = (self.mlp_PI_witnesses(PI_witnesses_dgms[g_i])).view(-1)
            batch_graphs_out[g_i] = torch.cat([batch_graphs[g_i], witnesses_PI_out], dim=0)

        score = F.dropout(self.linear1(batch_graphs_out), self.final_dropout)#, training=self.training)

        return score
