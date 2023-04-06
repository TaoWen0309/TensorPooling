import networkx as nx
from scipy.sparse import csgraph, csr_matrix
from scipy.io import loadmat
from scipy.linalg import eigh
import numpy as np
from sklearn.metrics import pairwise_distances
import gudhi as gd

'''
For filtration functions, I put 5 options by using 5 different network statistics. You can also run a toy example to check the output (i.e., a PD). 
Note that, there are 2 inputs for each function - adj_matrix and max_scale, where the max_scale is a pre-defined hyperparameters and is used to replace inf value; 
in general, you can set it to be 50 or 100 (or other large number). 
In this case, by using these 5 filtration functions, you can generate 5 PDs. 
Then you can feed a PD to the 'persistence_images' function and get the corresponding PI (you can set the size of PI by changing resolution; 
in general, we consider 20x20, 50x50, or 100x100.
'''

def sublevel_persistence_diagram(A, max_scale, method):
    
    assert method in ['degree','betweenness','communicability','eigenvector','closeness']
    
    G = nx.from_numpy_array(A)
    if method == 'degree':
        node_features = np.sum(A, axis=1)
    elif method == 'betweenness':
        node_features_dict = nx.betweenness_centrality(G)
        node_features = [i for i in node_features_dict.values()]
    elif method == 'communicability':
        node_features_dict = nx.communicability_betweenness_centrality(G)
        node_features = [i for i in node_features_dict.values()]
    elif method == 'eigenvector':
        node_features_dict = nx.eigenvector_centrality(G)
        node_features = [i for i in node_features_dict.values()]
    elif method == 'closeness':
        node_features_dict = nx.closeness_centrality(G)
        node_features = [i for i in node_features_dict.values()]
    
    stb = gd.SimplexTree()
    (xs, ys) = np.where(np.triu(A))
    for j in range(A.shape[0]):
        stb.insert([j], filtration=-1e10)

    for idx, x in enumerate(xs):
        stb.insert([x, ys[idx]], filtration=-1e10)

    for j in range(A.shape[0]):
        stb.assign_filtration([j], node_features[j])

    stb.make_filtration_non_decreasing()
    dgm = stb.persistence()
    pd = [dgm[i][1] if dgm[i][1][1] != np.inf else (dgm[i][1][0], max_scale) for i in np.arange(0, len(dgm), 1)]

    return np.array(pd)

# persistence image
def persistence_images(dgm, resolution = [50,50], return_raw = False, normalization = True, bandwidth = 1., power = 1.):
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

    return np.array(norm_output)

# a toy example
g = nx.gnp_random_graph(n = 100, p=0.3)
adj = nx.adjacency_matrix(g).toarray()
methods = ['degree','betweenness','communicability','eigenvector','closeness']

PI_set = []
for mtd in methods:
    pd = sublevel_persistence_diagram(adj,50,mtd)
    pi = persistence_images(pd)
    PI_set.append(pi)
 
PI_tensor = np.concatenate((np.stack(PI_set),),axis=0)
print('PI tensor shape {}'.format(PI_tensor.shape))
print(PI_tensor)