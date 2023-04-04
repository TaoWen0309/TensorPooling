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

from torch.autograd import Variable
import gudhi as gd
import numpy as np

def persistence_images(dgm, resolution = [5,5], return_raw = False, normalization = True, bandwidth = 1., power = 1.):
    PXs, PYs = dgm[:, 0], dgm[:, 1]
    xm, xM, ym, yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
    x = np.linspace(xm, xM, resolution[0])
    y = np.linspace(ym, yM, resolution[1])
    X, Y = np.meshgrid(x, y)
    Zfinal = np.zeros(X.shape)
    X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]

    # Compute persistence image
    P0, P1 = np.reshape(dgm[:, 0], [1, 1, -1]), np.reshape(dgm[:, 1], [1, 1, -1])
    weight = np.abs(P1 - P0)
    distpts = np.sqrt((X - P0) ** 2 + (Y - P1) ** 2)

    if return_raw:
        lw = [weight[0, 0, pt] for pt in range(weight.shape[2])]
        lsum = [distpts[:, :, pt] for pt in range(distpts.shape[2])]
    else:
        weight = weight ** power
        Zfinal = (np.multiply(weight, np.exp(-distpts ** 2 / bandwidth))).sum(axis=2)

    output = [lw, lsum] if return_raw else Zfinal

    if normalization:
        norm_output = (output - np.min(output))/(np.max(output) - np.min(output))
    else:
        norm_output = output

    return norm_output

def diagram_from_simplex_tree(st, mode, dim=0):
    st.compute_persistence(min_persistence=-1.)
    dgm0 = st.persistence_intervals_in_dimension(0)[:, 1]

    if mode == "superlevel":
        # birth and death times of the connected components of the complex
        dgm0 = - dgm0[np.where(np.isfinite(dgm0))]
    elif mode == "sublevel":
        # birth and death times of the holes of the complex
        dgm0 = dgm0[np.where(np.isfinite(dgm0))]
    if dim==0:
        return dgm0
    elif dim==1:
        # birth and death times of the cycles of the complex
        dgm1 = st.persistence_intervals_in_dimension(1)[:,0]
        return dgm0, dgm1

# return sum of PD
def sum_diag_from_point_cloud(X, mode="superlevel"):
    # This constructs a simplicial complex where vertices represent the points in X, 
    # and higher-dimensional simplices are added based on the distances between them.
    rc = gd.RipsComplex(points=X)
    # The max_dimension parameter is set to 1, which means that only edges (1-simplices) are considered.
    # The simplex tree is a data structure that efficiently stores the information about the simplicial complex, and can be used to compute the persistence homology of the complex.
    st = rc.create_simplex_tree(max_dimension=1)
    # The persistence diagram is a summary of the topological features of the simplicial complex, 
    # where each point in the diagram represents a homology class that persists over a range of filtration values.
    dgm = diagram_from_simplex_tree(st, mode=mode)
    # a measure of the overall complexity of the point cloud in terms of its persistent homology features
    sum_dgm = np.sum(dgm)
    return sum_dgm


class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, device):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input node features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average); will not be used
            device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1)) # weight for the central node in pooling if not learn_eps
        self.num_neighbors = 5 # for score_node_layer

        # List of MLPs
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # all layers except the output layer
        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # for attentional second-order pooling
        self.PI_hidden_dim = 128
        self.total_latent_dim = input_dim + hidden_dim * (num_layers - 1) + self.PI_hidden_dim
        self.dense_dim = self.total_latent_dim
        self.attend = nn.Linear(self.total_latent_dim - self.PI_hidden_dim, 1)
        self.output = nn.Linear(self.dense_dim, output_dim)
        self.mlp_PI_witnesses = nn.Linear(25, self.PI_hidden_dim)
        
        # 2 dimensions for point clouds coordinates
        # point clouds pooling for nodes
        self.score_node_layer = GCNConv(input_dim, self.num_neighbors * 2) 
        # point clouds pooling for graphs
        self.score_graph_layer = GCNConv(input_dim, 2)

    # for max pooling
    def __preprocess_neighbors_maxpool(self, batch_graph):
        # create padded_neighbor_list in concatenated graph

        # compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])
        # padded neighbors of the batch
        padded_neighbor_list = []
        # start idx of each graph
        start_idx = [0]
        # each graph
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            # padded neighbors of the graph
            padded_neighbors = []
            # neighbors of each node
            for j in range(len(graph.neighbors)):
                # padded neighbors of the node
                # add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                # padding, dummy data is assumed to be stored as -1
                pad.extend([-1]*(max_deg - len(pad))) # len(pad) = len(graph.neighbors[j])
                # Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])
                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)


    def __preprocess_neighbors_sumavepool_witnesses(self, batch_graph):
        ###create block diagonal sparse matrix
        edge_attr = None # edge attributes
        edge_mat_list = []
        start_idx = [0]
        pooled_x = []
        pooled_graph_sizes = []
        PI_witnesses_dgms = []
        
        for i, graph in enumerate(batch_graph):
            x = graph.x.to(self.device) # x: node_features
            edge_index = graph.edge_index.to(self.device) # edge_index: edge_mat
            witnesses = self.score_graph_layer(x, edge_index).to(self.device) # GCN layer, (number of nodes, 2)
            
            # GNN layer, topological score(unweighted) of each node
            node_embeddings = self.score_node_layer(x, edge_index).to(self.device) # GCN layer, (number of nodes, self.num_neighbors * 2)
            node_point_clouds = node_embeddings.view(-1, self.num_neighbors, 2).to(self.device) # (number of nodes, self.num_neighbors, 2)
            score_lifespan = torch.FloatTensor([sum_diag_from_point_cloud(node_point_clouds[i,...])\
                                                for i in range(node_point_clouds.size(0))]).to(self.device) # [i,...]=[i,:,:]
            # All nodes are assigned to the same batch (batch 0) since they are from the same graph
            batch = torch.LongTensor([0] * x.size(0)).to(self.device)
            # The function selects the top-k nodes with respect to the scores, where k is set to be the top 50% of the nodes.
            perm = topk(score_lifespan, 0.5, batch)
            # features of topk nodes
            x = x[perm]
            # filters the edge connectivity and edge attributes based on the indices of the topk nodes
            edge_index, _ = filter_adj(edge_index, edge_attr, perm, num_nodes=graph.x.size(0))
            # pooled graph, node features, edge connectivity
            start_idx.append(start_idx[i] + x.size(0))
            edge_mat_list.append(edge_index + start_idx[i])
            pooled_x.append(x)
            pooled_graph_sizes.append(x.size(0))

            # Witnesses Complex layer (!independent from the above part!)
            # to_dense_adj(graph.edge_index) = (batch_size, max_num_nodes, max_num_nodes)
            # the sum of the weights of all edges connected to that node(degree)
            network_statistics = torch.sum(to_dense_adj(graph.edge_index)[0,:,:], dim = 1).to(self.device)
            # top 20% nodes as landmarks by degree centrality
            witnesses_perm = topk(network_statistics, 0.2, batch) 
            # landmarks is a subset of witnesses
            landmarks = witnesses[witnesses_perm]
            # create a witness complex based on the selected witness and landmark nodes
            witness_complex = gd.EuclideanStrongWitnessComplex(witnesses=witnesses, landmarks=landmarks)
            # max_alpha_square: the maximum filtration value for the complex
            # limit_dimension: the maximum dimension of simplices to include in the tree
            simplex_tree = witness_complex.create_simplex_tree(max_alpha_square = 1, limit_dimension = 1)
            # This computes the persistent homology of the complex represented by the simplex tree, 
            # and returns the persistence intervals of each homology group in the tree.
            simplex_tree.compute_persistence(min_persistence=-1.)
            # This line extracts the persistence intervals of the witness complex in dimension 0 (i.e., connected components). 
            # persistence_intervals_in_dimension is used to obtain the intervals, and the [:-1,:] indexing is used to exclude the interval corresponding to the infinite simplex.
            witnesses_dgm = simplex_tree.persistence_intervals_in_dimension(0)[:-1,:]
            # flattened persistence image based on witnesses diagram
            PI_witnesses_dgm = torch.FloatTensor(persistence_images(witnesses_dgm).reshape(1,-1)).to(self.device)
            PI_witnesses_dgms.append(PI_witnesses_dgm)

        pooled_X_concat = torch.cat(pooled_x, 0).to(self.device) # concat pooled features, shape: (start_idx[-1], input_dim)
        Adj_block_idx = torch.cat(edge_mat_list, 1).to(self.device) # concat pooled edges, shape: (2, number of edges)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1]).to(self.device)

        if not self.learn_eps:
            num_node = start_idx[-1] # sum(pooled_graph_sizes)
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)]).to(self.device) # shape: (2,num_node)
            elem = torch.ones(num_node).to(self.device)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1).to(self.device)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0).to(self.device)
        
        # Sparse adj mat of the batch
        # Adj_block_idx: the indices of the non-zero elements
        # Adj_block_elem: values of the non-zero elements
        # torch.Size([start_idx[-1],start_idx[-1]]: sparse matrix size
        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))

        return Adj_block.to(self.device), pooled_X_concat, pooled_graph_sizes, PI_witnesses_dgms


    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)
        
        start_idx = [0]

        #compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1./len(graph.g)]*len(graph.g))
            
            else:
            ###sum pooling
                elem.extend([1]*len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
        
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        
        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        # the minimum value of each column of h
        # handle the case where a node has no neighbors, will never affect max-pooling
        dummy = torch.min(h, dim = 0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        # the maximum value over the second dimension (i.e., over the neighbors of each node)
        # The resulting tensor has the same number of rows as padded_neighbor_list and the maximum value of each row.
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
        
        # local and global topological encoding with PD is just a prerequisite step
        Adj_block, pooled_X_concat, pooled_graph_sizes, PI_witnesses_dgms = self.__preprocess_neighbors_sumavepool_witnesses(batch_graph)

        hidden_rep = [pooled_X_concat]
        h = pooled_X_concat # shape: (start_idx[-1], input_dim)

        # continue pooling with max/sum/average
        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max":
                h = self.next_layer(h, layer, padded_neighbor_list = padded_neighbor_list, learn_eps = self.learn_eps)
            elif not self.neighbor_pooling_type == "max":
                h = self.next_layer(h, layer, Adj_block = Adj_block, learn_eps = self.learn_eps)
            hidden_rep.append(h)

        # output of input and hidden layers, shape: (start_idx[-1], self.dense_dim - self.PI_hidden_dim)
        hidden_rep = torch.cat(hidden_rep, 1)

        # after local pooling
        graph_sizes = pooled_graph_sizes

        batch_graphs = torch.zeros(len(graph_sizes), self.dense_dim - self.PI_hidden_dim).to(self.device)
        batch_graphs = Variable(batch_graphs)

        batch_graphs_out = torch.zeros(len(graph_sizes), self.dense_dim).to(self.device)
        batch_graphs_out = Variable(batch_graphs_out)

        # split batch embeddings into graphs
        # a tuple with graph embeddings of shape
        node_embeddings = torch.split(hidden_rep, graph_sizes, dim=0)

        for g_i in range(len(graph_sizes)):
            # current graph embedding, shape: (pooled_graph_size, self.dense_dim - self.PI_hidden_dim)
            cur_node_embeddings = node_embeddings[g_i]
            # attention coefficients, shape: (pooled_graph_size, 1)
            attn_coef = self.attend(cur_node_embeddings)
            # shape: (1, pooled_graph_size)
            attn_weights = torch.transpose(attn_coef, 0, 1)
            # shape: (1, self.dense_dim - self.PI_hidden_dim)
            cur_graph_embeddings = torch.matmul(attn_weights, cur_node_embeddings)
            # current graph embedding, shape: (self.dense_dim - self.PI_hidden_dim)
            batch_graphs[g_i] = cur_graph_embeddings.view(self.dense_dim - self.PI_hidden_dim)
            # witnesses persistence PI learning
            # shape: (self.PI_hidden_dim=128)
            witnesses_PI_out = (self.mlp_PI_witnesses(PI_witnesses_dgms[g_i])).view(-1)
            batch_graphs_out[g_i] = torch.cat([batch_graphs[g_i], witnesses_PI_out], dim=0)

        score = F.dropout(self.output(batch_graphs_out), self.final_dropout)#, training=self.training)

        return score
