import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import to_dense_adj

import sys
sys.path.append("models/")
from mlp import MLP
from cnn import CNN
from diagram import sublevel_persistence_diagram, persistence_images, sum_diag_from_point_cloud

from torch.autograd import Variable
import numpy as np
import networkx as nx

from tltorch import TRL, TCL, FactorizedLinear, FactorizedTensor


class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, neighbor_pooling_type, sublevel_filtration_methods, tensor_decom_type, tensor_layer_type, device):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            sublevel_filtration_methods: methods for sublevel filtration on PD
            decom_type: Tensor decomposition type, Tucker/CP/TT
            tensor_layer: Tensor layer type, TCL/TRL'
            device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))
        self.num_neighbors = 5

        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.sublevel_filtration_methods = sublevel_filtration_methods
        self.cnn_dim = 8
        self.cnn = CNN(self.cnn_dim)

        self.tensor_decom_type = tensor_decom_type
        self.tensor_input_shape = (8,12,12)
        self.tensor_hidden_shape = [4,4,4]
        # self.tensor_output_shape = (2,2)
        
        if tensor_layer_type == 'TCL':
            self.tensor_layer = TCL(self.tensor_input_shape,self.tensor_hidden_shape)
        elif tensor_layer_type == 'TRL':
            self.tensor_layer = TRL(self.tensor_input_shape,self.tensor_hidden_shape)
        # self.tensor_linear = FactorizedLinear(in_tensorized_features=tuple(self.tensor_hidden_shape),out_tensorized_features=self.tensor_output_shape)

        # for attentional second-order pooling
        self.PI_hidden_dim = 16
        self.total_latent_dim = input_dim + hidden_dim * (num_layers - 1) + self.PI_hidden_dim
        self.dense_dim = self.total_latent_dim
        self.attend = nn.Linear(self.total_latent_dim - self.PI_hidden_dim, 1)
        self.linear1 = nn.Linear(self.dense_dim, output_dim)
        # self.mlp_PI_witnesses = nn.Linear(2500, self.PI_hidden_dim)

        # point clouds pooling for nodes
        self.score_node_layer = GCNConv(input_dim, self.num_neighbors * 2)
        # point clouds pooling for graphs
        self.score_graph_layer = GCNConv(input_dim, 2)


    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph

        #compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                #add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                #padding, dummy data is assumed to be stored in -1
                pad.extend([-1]*(max_deg - len(pad)))

                #Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)


    def __preprocess_neighbors_sumavepool_witnesses(self, batch_graph):
        ###create block diagonal sparse matrix
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
            
            for j in range(len(self.sublevel_filtration_methods)):
                pd = sublevel_persistence_diagram(adj,50,self.sublevel_filtration_methods[j])
                pi = torch.FloatTensor(persistence_images(pd)).to(self.device)
                PI_list_i.append(pi)
            PI_tensor_i = torch.stack(PI_list_i).to(self.device)
            PI_list.append(PI_tensor_i)

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
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1]).to(self.device)

        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)]).to(self.device)
            elem = torch.ones(num_node).to(self.device)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1).to(self.device)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0).to(self.device)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]])).to(self.device)

        return Adj_block, pooled_X_concat, pooled_graph_sizes, PI_concat

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim = 0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim = 1)[0]
        return pooled_rep

    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None, learn_eps = True):
        # pooling neighboring nodes and center nodes separately by epsilon reweighting. 

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree
        
        if learn_eps:
            # reweights the center node representation when aggregating it with its neighbors
            pooled = pooled + (1 + self.eps[layer])*h
       
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h


    def forward(self, batch_graph):
        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block, pooled_X_concat, pooled_graph_sizes, PI_concat = self.__preprocess_neighbors_sumavepool_witnesses(batch_graph)
        
        PI_emb = self.cnn(PI_concat) # before cnn: [batch_size,5,50,50]; after cnn: [batch_size,8,12,12]
        PI_decom =  FactorizedTensor.from_tensor(PI_emb, rank='same', factorization=self.tensor_decom_type).to(self.device)
        PI_hidden = self.tensor_layer(PI_decom).to(self.device)
        exit()


        #list of hidden representation at each layer (including input)
        hidden_rep = [pooled_X_concat]
        h = pooled_X_concat

        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max":
                h = self.next_layer(h, layer, padded_neighbor_list = padded_neighbor_list, learn_eps = self.learn_eps)
            elif not self.neighbor_pooling_type == "max":
                h = self.next_layer(h, layer, Adj_block = Adj_block, learn_eps = self.learn_eps)
            hidden_rep.append(h)

        hidden_rep = torch.cat(hidden_rep, 1)

        # after graph pooling
        graph_sizes = pooled_graph_sizes

        batch_graphs = torch.zeros(len(graph_sizes), self.dense_dim - self.PI_hidden_dim).to(self.device)
        batch_graphs = Variable(batch_graphs)

        batch_graphs_out = torch.zeros(len(graph_sizes), self.dense_dim).to(self.device)
        batch_graphs_out = Variable(batch_graphs_out)

        node_embeddings = torch.split(hidden_rep, graph_sizes, dim=0)

        for g_i in range(len(graph_sizes)):
            cur_node_embeddings = node_embeddings[g_i]
            attn_coef = self.attend(cur_node_embeddings)
            attn_weights = torch.transpose(attn_coef, 0, 1)
            cur_graph_embeddings = torch.matmul(attn_weights, cur_node_embeddings)
            batch_graphs[g_i] = cur_graph_embeddings.view(self.dense_dim - self.PI_hidden_dim)
            witnesses_PI_out = (self.mlp_PI_witnesses(PI_witnesses_dgms[g_i])).view(-1)
            batch_graphs_out[g_i] = torch.cat([batch_graphs[g_i], witnesses_PI_out], dim=0)

        score = F.dropout(self.linear1(batch_graphs_out), self.final_dropout)#, training=self.training)

        return score
