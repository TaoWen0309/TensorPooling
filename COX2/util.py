import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T

# Molecular
class S2VGraph(object):
    def __init__(self, g, label, node_tags=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag, used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.g = g
        self.label = label
        self.node_tags = node_tags

        self.neighbors = [] # neighboring nodes of eahc node
        self.node_features = 0 # ?
        self.edge_mat = 0 # (2, 2 * number_of_edges)
        self.max_neighbor = 0 # max degree of all nodes

# Molecular
def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {} # labels of all graphs
    feat_dict = {} # node tags for all graphs

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip()) # the first line contains the number of graphs
        for i in range(n_g):
            row = f.readline().strip().split()
            # for each graph:  number of nodes, label
            n, l = [int(w) for w in row] 
            # label indexed globally from 0
            if not l in label_dict:
                mapped = len(label_dict) 
                label_dict[l] = mapped
            
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            
            for j in range(n):
                # g.add_node(j)
                # next n rows
                row = f.readline().strip().split()
                g.add_node(row[0])
                # each row: node + number of edges + adjacent nodes
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                
                # node tag indexed globally from 0
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])
                
                # if node features provided
                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    # g.add_edge(j, row[k])
                    g.add_edge(row[0], row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags)) # node features and n_edges are not used at all!

    # add labels and edge_mat       
    for g in g_list:
        # add neighbors for each node
        g.neighbors = [[] for _ in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        # maximum degree of the nodes in the graph
        degree_list = []
        for i in range(len(g.g)):
            # g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)
        # l -> label_dict[l]
        g.label = label_dict[g.label]
        # edges are undirectional!
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])
        # shape: (2, 2 * number of edges)
        g.edge_mat = torch.LongTensor(edges).transpose(1,0)
        # deg_list = list(dict(g.g.degree(range(len(g.g)))).values())

        # use degrees not idx as node tags (could be duplicates!)
        if degree_as_tag:
            g.node_tags = list(dict(g.g.degree).values())

    # extracting unique node tags   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))
    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    # use ? as node_features
    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset)) # 0: number of local nodes; 1: number of global nodes 
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1 # same node, different index

    print('# classes: %d' % len(label_dict))
    print('# total number of nodes: %d' % len(tagset))
    print("# number of graphs: %d" % len(g_list))

    return g_list, len(label_dict)

# Molecular
def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]

    train_idx, test_idx = list(skf.split(np.zeros(len(labels)), labels))[fold_idx]
    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

# Chemical Compound
def separate_TUDataset(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [int(graph.y) for graph in graph_list] # have a built-in label    
    
    train_idx, test_idx = list(skf.split(np.zeros(len(labels)), labels))[fold_idx]
    train_graph_list = [graph_list[int(i)] for i in train_idx]
    test_graph_list = [graph_list[int(i)] for i in test_idx]

    return train_graph_list, test_graph_list

# Social
def get_dataset(name, sparse=True, cleaned=False):
    dataset = TUDataset(root='/tmp/' + name, name=name)
    dataset.data.edge_attr = None

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset.copy(torch.tensor(indices))

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])

    return dataset