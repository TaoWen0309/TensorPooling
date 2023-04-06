import networkx as nx
from scipy.sparse import csgraph, csr_matrix
from scipy.io import loadmat
from scipy.linalg import eigh
import numpy as np
from sklearn.metrics import pairwise_distances
import gudhi as gd

# sublevel filtration on degree
def sublevel_degree_persistence_diagram(A, max_scale):
    nodes_degree = np.sum(A, axis=1)

    stb = gd.SimplexTree()
    (xs, ys) = np.where(np.triu(A))
    for j in range(A.shape[0]):
        stb.insert([j], filtration=-1e10)

    for idx, x in enumerate(xs):
        stb.insert([x, ys[idx]], filtration=-1e10)

    for j in range(A.shape[0]):
        stb.assign_filtration([j], nodes_degree[j])

    stb.make_filtration_non_decreasing()
    dgm = stb.persistence()
    pd = [dgm[i][1] if dgm[i][1][1] != np.inf else (dgm[i][1][0], max_scale) for i in np.arange(0, len(dgm), 1)]

    return np.array(pd)


# sublevel filtration on betweenness
def sublevel_betweenness_persistence_diagram(A, max_scale):
    G = nx.from_numpy_matrix(A)
    nodes_betweenness_dict = nx.betweenness_centrality(G)
    nodes_betweenness = [i for i in nodes_betweenness_dict.values()]

    stb = gd.SimplexTree()
    (xs, ys) = np.where(np.triu(A))
    for j in range(A.shape[0]):
        stb.insert([j], filtration=-1e10)

    for idx, x in enumerate(xs):
        stb.insert([x, ys[idx]], filtration=-1e10)

    for j in range(A.shape[0]):
        stb.assign_filtration([j], nodes_betweenness[j])

    stb.make_filtration_non_decreasing()
    dgm = stb.persistence()
    pd = [dgm[i][1] if dgm[i][1][1] != np.inf else (dgm[i][1][0], max_scale) for i in np.arange(0, len(dgm), 1)]

    return np.array(pd)

# sublevel filtration on communicability
def sublevel_communicability_betweenness_persistence_diagram(A, max_scale):
    G = nx.from_numpy_matrix(A)
    nodes_communicability_betweenness_dict = nx.communicability_betweenness_centrality(G)
    nodes_communicability_betweenness = [i for i in nodes_communicability_betweenness_dict.values()]

    stb = gd.SimplexTree()
    (xs, ys) = np.where(np.triu(A))
    for j in range(A.shape[0]):
        stb.insert([j], filtration=-1e10)

    for idx, x in enumerate(xs):
        stb.insert([x, ys[idx]], filtration=-1e10)

    for j in range(A.shape[0]):
        stb.assign_filtration([j], nodes_communicability_betweenness[j])

    stb.make_filtration_non_decreasing()
    dgm = stb.persistence()
    pd = [dgm[i][1] if dgm[i][1][1] != np.inf else (dgm[i][1][0], max_scale) for i in np.arange(0, len(dgm), 1)]

    return np.array(pd)


# sublevel filtration on eigenvector
def sublevel_eigenvector_persistence_diagram(A, max_scale):
    G = nx.from_numpy_matrix(A)
    nodes_eigenvector_dict = nx.eigenvector_centrality(G)
    nodes_eigenvector = [i for i in nodes_eigenvector_dict.values()]

    stb = gd.SimplexTree()
    (xs, ys) = np.where(np.triu(A))
    for j in range(A.shape[0]):
        stb.insert([j], filtration=-1e10)

    for idx, x in enumerate(xs):
        stb.insert([x, ys[idx]], filtration=-1e10)

    for j in range(A.shape[0]):
        stb.assign_filtration([j], nodes_eigenvector[j])

    stb.make_filtration_non_decreasing()
    dgm = stb.persistence()
    pd = [dgm[i][1] if dgm[i][1][1] != np.inf else (dgm[i][1][0], max_scale) for i in np.arange(0, len(dgm), 1)]

    return np.array(pd)

# sublevel filtration on closeness
def sublevel_closeness_persistence_diagram(A, max_scale):
    G = nx.from_numpy_matrix(A)
    nodes_closeness_dict = nx.closeness_centrality(G)
    nodes_closeness = [i for i in nodes_closeness_dict.values()]

    stb = gd.SimplexTree()
    (xs, ys) = np.where(np.triu(A))
    for j in range(A.shape[0]):
        stb.insert([j], filtration=-1e10)

    for idx, x in enumerate(xs):
        stb.insert([x, ys[idx]], filtration=-1e10)

    for j in range(A.shape[0]):
        stb.assign_filtration([j], nodes_closeness[j])

    stb.make_filtration_non_decreasing()
    dgm = stb.persistence()
    pd = [dgm[i][1] if dgm[i][1][1] != np.inf else (dgm[i][1][0], max_scale) for i in np.arange(0, len(dgm), 1)]

    return np.array(pd)

# an toy example
g = nx.gnp_random_graph(n = 100, p=0.3)
adj = nx.adjacency_matrix(g).toarray()
pd = sublevel_communicability_betweenness_persistence_diagram(A= adj, max_scale=2.) # define max_scale by your own

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

    return norm_output
