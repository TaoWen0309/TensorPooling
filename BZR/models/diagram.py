import gudhi as gd
import numpy as np
import networkx as nx

def sublevel_persistence_diagram(A, method, max_scale=50):
    
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
        node_features_dict = nx.eigenvector_centrality(G,max_iter=100000)
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

def persistence_images(dgm, resolution = [50,50], return_raw = False, normalization = True, bandwidth = 1., power = 1.):
    PXs, PYs = dgm[:, 0], dgm[:, 1]
    xm, xM, ym, yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
    x = np.linspace(xm, xM, resolution[0])
    y = np.linspace(ym, yM, resolution[1])
    X, Y = np.meshgrid(x, y)
    Zfinal = np.zeros(X.shape)
    X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]

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

    max_output = np.max(output)
    min_output = np.min(output)
    if normalization and (max_output != min_output):
        norm_output = (output - min_output)/(max_output - min_output)
    else:
        norm_output = output

    return norm_output


def diagram_from_simplex_tree(st, mode, dim=0):
    st.compute_persistence(min_persistence=-1.)
    dgm0 = st.persistence_intervals_in_dimension(0)[:, 1]

    if mode == "superlevel":
        dgm0 = - dgm0[np.where(np.isfinite(dgm0))]
    elif mode == "sublevel":
        dgm0 = dgm0[np.where(np.isfinite(dgm0))]
    if dim==0:
        return dgm0
    elif dim==1:
        dgm1 = st.persistence_intervals_in_dimension(1)[:,0]
        return dgm0, dgm1

def sum_diag_from_point_cloud(X, mode="superlevel"):
    rc = gd.RipsComplex(points=X)
    st = rc.create_simplex_tree(max_dimension=1)
    dgm = diagram_from_simplex_tree(st, mode=mode)
    sum_dgm = np.sum(dgm)
    return sum_dgm