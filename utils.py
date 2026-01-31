import os.path as osp
import numpy as np
import scipy.sparse as sp
import networkx as nx
import json
import pandas as pd
from ogb.nodeproppred import NodePropPredDataset
import torch
import os
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import NormalizeFeatures
from networkx.readwrite import json_graph
from torch_geometric.datasets import Planetoid,WikipediaNetwork, WebKB, TUDataset, Coauthor, Reddit, Amazon, CoraFull, IMDB
from torch_geometric.utils import to_dense_adj, to_undirected
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from torch_geometric.utils.convert import to_scipy_sparse_matrix    
import sys
from ogb.nodeproppred import PygNodePropPredDataset
import scipy.sparse as sp
from scipy import io
from torch_geometric.utils.convert import from_scipy_sparse_matrix

def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def fetch_normalization(type):
   switcher = {
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func


def encode_onehot(labels):
    classes = set(labels)
    # print(f'labels: {labels}')
    # print(f'classes: {classes}')
    classes_dict = {c.item(): np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    # print(f'classes_dict: {classes_dict}')
    # print(list(map(classes_dict.get, labels)))
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)


    return labels_onehot

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']


def load_data_gs(path="./data/", dataset="MUTAG", device=None):
    print('Loading {} dataset...'.format(dataset))
    pre_transform = NormalizeFeatures()
    data = TUDataset(path, dataset, pre_transform=pre_transform)[0].to(device)
    print(data)
    features, labels, edges, batch = data.x, data.y, data.edge_index, data.batch
    # adj contains all information from all graphs    
    adj = to_dense_adj(to_undirected(edges)).squeeze()

    return adj, features, labels, batch


def split(dataset, split_type="random", num_train_per_class=20, num_val=500, num_test=1000):
    data = dataset.get(0)
    if split_type=="public" and hasattr(data, "train_mask"):
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    else:
        train_mask = torch.zeros_like(data.y, dtype=torch.bool)
        val_mask = torch.zeros_like(data.y, dtype=torch.bool)
        test_mask = torch.zeros_like(data.y, dtype=torch.bool)

        for c in range(dataset.num_classes):
            idx = (data.y == c).nonzero(as_tuple=False).view(-1)
            idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
            train_mask[idx] = True

        remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
        remaining = remaining[torch.randperm(remaining.size(0))]

        val_mask[remaining[:num_val]] = True
        test_mask[remaining[num_val:num_val + num_test]] = True
    return (train_mask, val_mask, test_mask)

def split_random( n, n_train, n_val):
    rnd = np.random.permutation(n)

    train_idx = np.sort(rnd[:n_train])
    val_idx = np.sort(rnd[n_train:n_train + n_val])

    train_val_idx = np.concatenate((train_idx, val_idx))
    test_idx = np.sort(np.setdiff1d(np.arange(n), train_val_idx))

    return train_idx, val_idx, test_idx


def load_data(path="./data/", dataset_name="Cora",training_id = 0 , nb_nodes=20, nb_graphs=20, p =None, q=None, device=None):
    print('Loading {} dataset...'.format(dataset_name))

    pre_transform = NormalizeFeatures()

    if dataset_name == "SBM":
        dataset = SBM_pyg('./data/SBM', nb_nodes=nb_nodes, nb_graphs= nb_graphs, p = p, q= q,  pre_transform=None)
        features, labels, adj = [], [], []
        idx_train, idx_val, idx_test = [], [], []
        for exp in range(len(dataset)):
            data = dataset[exp]
            features.append(data.x.view(-1,1))
            labels.append(data.y)
            adj.append(to_dense_adj(to_undirected(data.edge_index)).squeeze())
            idx_train.append(data.train_mask)
            idx_val.append(data.val_mask)
            idx_test.append(data.test_mask)

    elif dataset_name in {"Cora", "CiteSeer", "PubMed"}:
        data = Planetoid(path, dataset_name, pre_transform=pre_transform)[0].to(device)
        features, labels, edges = data.x, data.y, data.edge_index
        adj = to_scipy_sparse_matrix(to_undirected(edges))
        #adj = to_dense_adj(to_undirected(edges)).squeeze()
        idx_train = data.train_mask
        idx_val = data.val_mask
        idx_test = data.test_mask


        #adj = to_dense_adj(to_undirected(edges)).squeeze()
        idx_train = data.train_mask
        idx_val = data.val_mask
        idx_test = data.test_mask
    elif dataset_name == 'CS' or  dataset_name == 'Physics':
        dataset = Coauthor(root=path, name=dataset_name, transform=pre_transform)
        data = dataset[0].to(device)
        features, labels, edges = data.x, data.y, data.edge_index
        adj = to_scipy_sparse_matrix(to_undirected(edges))
        #adj = to_dense_adj(to_undirected(edges)).squeeze()
        idx_train, idx_val, idx_test = split(dataset, split_type="random", num_train_per_class=20, num_val=500, num_test=1000)

    elif dataset_name == 'Computers' or  dataset_name == 'Photo':
        dataset = Amazon(root=path, name=dataset_name, transform=pre_transform)
        data = dataset[0].to(device)
        features, labels, edges = data.x, data.y, data.edge_index
        adj = to_scipy_sparse_matrix(to_undirected(edges))
        #adj = to_dense_adj(to_undirected(edges)).squeeze()
        idx_train, idx_val, idx_test = split(dataset, split_type="random", num_train_per_class=20, num_val=500, num_test=1000)

    elif dataset_name == 'ogbn-arxiv' :
        # The graph is very large, we cannot work with Dense adjacency matrix
        dataset = PygNodePropPredDataset(name = dataset_name)
        data =  dataset[0].to(device)
        features, labels, edges = data.x, data.y.squeeze(1), data.edge_index
        split_idx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
        a_train = torch.zeros(data.num_nodes, dtype=torch.bool)
        a_train[idx_train] = True
        a_val = torch.zeros(data.num_nodes, dtype=torch.bool)
        a_val[idx_val] = True
        a_test = torch.zeros(data.num_nodes, dtype=torch.bool)
        a_test[idx_test] = True
        idx_train, idx_val, idx_test = a_train , a_val, a_test
        adj = to_scipy_sparse_matrix(to_undirected(edges))
        #adj = to_dense_adj(to_undirected(edges)).squeeze()

    elif dataset_name == 'Reddit' :
        
        #G_data = json.load(open("/home/yassine/Projects/Shift_Operators/reddit-G.json"))
        #G = json_graph.node_link_graph(G_data)
        #if isinstance(G.nodes()[0], int):
        #    conversion = lambda n : int(n)
        #else:
        #    conversion = lambda n : n

        #if os.path.exists("/home/yassine/Projects/Shift_Operators/reddit-feats.npy"):
        #    feats = np.load( "/home/yassine/Projects/Shift_Operators/reddit-feats.npy")
        #else:
        #    print("No features present.. Only identity features will be used.")
        #    feats = None
        #id_map = json.load(open("/home/yassine/Projects/Shift_Operators/reddit-id_map.json"))
        #id_map = {conversion(k):int(v) for k,v in id_map.items()}
        #walks = []
        #class_map = json.load(open( "/home/yassine/Projects/Shift_Operators/reddit-class_map.json"))
        #if isinstance(list(class_map.values())[0], list):
        #    lab_conversion = lambda n : n
        #else:
        #    lab_conversion = lambda n : int(n)

        #class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}
        #labels = torch.tensor(list(class_map.values())).to(device)
        ## Remove all nodes that do not have val/test annotations
        ## (necessary because of networkx weirdness with the Reddit data)
        #broken_count = 0
        #nodes = list(G.nodes()).copy()
        #for node in nodes :
        #    if not 'val' in G.nodes[node] or not 'test' in G.nodes[node]:
        #        G.remove_node(node)
        #        broken_count += 1
        #print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

        ## Make sure the graph has edge train_removed annotations
        ## (some datasets might already have this..)
        #print("Loaded data.. now preprocessing..")
        #for edge in G.edges():
        #    if (G.nodes[edge[0]]['val'] or G.nodes[edge[1]]['val'] or
        #        G.nodes[edge[0]]['test'] or G.nodes[edge[1]]['test']):
        #        G[edge[0]][edge[1]]['train_removed'] = True
        #    else:
        #        G[edge[0]][edge[1]]['train_removed'] = False
                
        #normalize = True 
        #if normalize and not feats is None:
        #    from sklearn.preprocessing import StandardScaler
        #    train_ids = np.array([id_map[n] for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']])
        #    train_feats = feats[train_ids]
        #    scaler = StandardScaler()
        #    scaler.fit(train_feats)
        #    feats = scaler.transform(feats)
        #nodes_ids = np.array([id_map[n] for n in G.nodes() ])
        #adj = nx.adjacency_matrix(G , nodelist = list(G.nodes()) )
        #features = torch.tensor(feats).to(device)
        #idx_train = np.array([id_map[n] for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']])
        #idx_train = torch.tensor(idx_train).to(device)
        #idx_val = np.array([id_map[n] for n in G.nodes() if G.nodes[n]['val']])
        #idx_val = torch.tensor(idx_val).to(device)
        #idx_test = np.array([id_map[n] for n in G.nodes() if G.nodes[n]['test']])
        #idx_test = torch.tensor(idx_test).to(device)
        
        # Importation 2 
        loaded_data = np.load('/home/yassine/Projects/Shift_Operators/23742119.npz')
        adj = sp.csr_matrix((loaded_data['adj_data'], loaded_data['adj_indices'], loaded_data['adj_indptr']),
                                    shape=loaded_data['adj_shape']).tocoo()
        labels = torch.tensor(loaded_data['labels']).to(device)
        features = torch.tensor(loaded_data['attr_matrix']).to(device)
        edge_index, edge_weight = from_scipy_sparse_matrix(adj)

        ntrain_div_classes = 20 # Yaml file in the github
        n, d = loaded_data["attr_matrix"].shape
        num_classes = loaded_data["labels"].max() + 1
        n_train = num_classes * ntrain_div_classes
        n_val = n_train * 10
        idx_train, idx_val, idx_test = split_random(n, n_train, n_val)
        idx_train = torch.tensor(idx_train).to(device)
        idx_val = torch.tensor(idx_val).to(device)
        idx_test = torch.tensor(idx_test).to(device)


        ###################
        ### Importation 3 
        # data = Reddit(path, pre_transform=pre_transform)[0].to(device)
        # features, labels, edges = data.x, data.y, data.edge_index
        # adj = to_scipy_sparse_matrix(to_undirected(edges))
        # idx_train = data.train_mask
        # idx_val = data.val_mask
        # idx_test = data.test_mask
    elif dataset_name == 'genius' :
        filename = 'genius'
        fulldata = scipy.io.loadmat(f'/home/yassine/Projects/GMNN/comparaison/Benchmark/data/genius.mat')

        edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long).to(device)
        features = torch.tensor(fulldata['node_feat'], dtype=torch.float).to(device)
        labels = torch.tensor(fulldata['label'], dtype=torch.long).squeeze().to(device)
        num_nodes = labels.shape[0]
        adj = to_scipy_sparse_matrix(edge_index)
        graph = {'edge_index': edge_index,
                        'edge_feat': None,
                        'node_feat': features,
                        'num_nodes': num_nodes}
        idx_train, idx_val, idx_test = rand_train_test_idx(
                labels, train_prop=0.5, valid_prop=.25, ignore_negative=True)

    elif dataset_name == 'Penn94' :
        mat = scipy.io.loadmat('/home/yassine/Projects/Adaptive_GNN/Benchmark/data/facebook100/Penn94.mat')
        A = mat['A']
        metadata = mat['local_info']
        edge_index = torch.tensor(A.nonzero(), dtype=torch.long).to(device)
        metadata = metadata.astype(np.int)
        labels = metadata[:, 1] - 1  # gender labels, -1 means unlabeled

        # make features into one-hot encodings
        feature_vals = np.hstack(
            (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
        features = np.empty((A.shape[0], 0))
        for col in range(feature_vals.shape[1]):
            feat_col = feature_vals[:, col]
            feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
            features = np.hstack((features, feat_onehot))
        adj = to_scipy_sparse_matrix(edge_index)
        features = torch.tensor(features, dtype=torch.float).to(device)
        num_nodes = metadata.shape[0]
        labels = torch.tensor(labels).to(device)
        
        idx_train, idx_val, idx_test = rand_train_test_idx(
                labels, train_prop=0.5, valid_prop=.25, ignore_negative=True)

    elif dataset_name == 'twitch-gamers' :
        edges = pd.read_csv('/home/yassine/Projects/Adaptive_GNN/Benchmark/data/twitch-gamer_edges.csv')
        nodes = pd.read_csv('/home/yassine/Projects/Adaptive_GNN/Benchmark/data/twitch-gamer_feat.csv')

        edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor).to(device)
        num_nodes = len(nodes)
        labels, features = load_twitch_gamer(nodes, "mature")

        features = torch.tensor(features, dtype=torch.float)
        features = features.to(device)
        normalize = True
        if normalize:
            features = features - features.mean(dim=0, keepdim=True)
            features = features / features.std(dim=0, keepdim=True)
  
        dataset = NCDataset("twitch-gamer")
        dataset.graph = {'edge_index': edge_index,
                        'features': features,
                        'edge_feat': None,
                        'num_nodes': num_nodes}
        dataset.label = torch.tensor(labels)
        split_idx = dataset.get_idx_split(train_prop=0.5, valid_prop=0.25)
        idx_train = split_idx['train']
        idx_val = split_idx['valid']
        idx_test = split_idx['test']
        labels = dataset.label.to(device)
        adj = to_scipy_sparse_matrix(edge_index)
    elif dataset_name == "arxiv-year" :
        nclass=5
        filename = 'arxiv-year'
        dataset = NCDataset(filename)
        ogb_dataset = NodePropPredDataset(name='ogbn-arxiv')
        dataset.graph = ogb_dataset.graph
        dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
        dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

        label = even_quantile_labels(
            dataset.graph['node_year'].flatten(), nclass, verbose=False)
        dataset.label = torch.as_tensor(label).reshape(-1, 1)
        labels = dataset.label.to(device)
        labels = labels.squeeze(1)
        features = dataset.graph['node_feat'].to(device)
        edge_index = dataset.graph['edge_index']
        adj = to_scipy_sparse_matrix(edge_index)
        split_idx = dataset.get_idx_split(train_prop=0.5, valid_prop=0.25)
        idx_train = split_idx['train'].to(device)
        idx_val = split_idx['valid'].to(device)
        idx_test = split_idx['test'].to(device)
    elif dataset_name == 'deezer-europe':
        filename = 'deezer-europe'
        dataset = NCDataset(filename)
        deezer = io.loadmat('/home/yassine/Projects/GS_GSO/deezer-europe.mat')

        A, label, features = deezer['A'], deezer['label'], deezer['features']
        edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
        node_feat = torch.tensor(features.todense(), dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long).squeeze()
        num_nodes = label.shape[0]

        dataset.graph = {'edge_index': edge_index,
                        'edge_feat': None,
                        'node_feat': node_feat,
                        'num_nodes': num_nodes}
        dataset.label = label
        labels = dataset.label.to(device)
        features = node_feat.to(device)
        edge_index = edge_index.to(device)
        adj = to_scipy_sparse_matrix(edge_index)
        split_idx = dataset.get_idx_split(train_prop=0.5, valid_prop=0.25)
        idx_train = split_idx['train'].to(device)
        idx_val = split_idx['valid'].to(device)
        idx_test = split_idx['test'].to(device)
    elif dataset_name == "imdb" :
        data = IMDB(path, pre_transform=pre_transform)[0].to(device)
        features, labels, edges = data.x, data.y, data.edge_index
        adj = to_scipy_sparse_matrix(to_undirected(edges))
        #adj = to_dense_adj(to_undirected(edges)).squeeze()
        idx_train = data.train_mask
        idx_val = data.val_mask
        idx_test = data.test_mask
    elif dataset_name == "chameleon" or dataset_name == "squirrel" or dataset_name   == "crocodile":
        data = WikipediaNetwork(path,dataset_name, pre_transform=pre_transform)[0].to(device)
        features, labels, edges = data.x, data.y, data.edge_index
        adj = to_scipy_sparse_matrix(to_undirected(edges))
        #adj = to_dense_adj(to_undirected(edges)).squeeze()
        idx_train = data.train_mask
        idx_val = data.val_mask
        idx_test = data.test_mask

        idx_train = idx_train[:,training_id]
        idx_val = idx_val[:,training_id]
        idx_test = idx_test[:,training_id]
    elif dataset_name == "Cornell" or dataset_name ==  "Texas" or dataset_name ==  "Wisconsin" :
        data = WebKB(path,dataset_name, pre_transform=pre_transform)[0].to(device)
        features, labels, edges = data.x, data.y, data.edge_index
        adj = to_scipy_sparse_matrix(to_undirected(edges))
        #adj = to_dense_adj(to_undirected(edges)).squeeze()
        idx_train = data.train_mask
        idx_val = data.val_mask
        idx_test = data.test_mask
        idx_train = idx_train[:,training_id]
        idx_val = idx_val[:,training_id]
        idx_test = idx_test[:,training_id]
    else:
        print("Not a correct dataset name!")
        exit()

    return adj, features, labels, idx_train, idx_val, idx_test







def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label



def load_twitch_gamer(nodes, task="dead_account"):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding
    
    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()
    
    return label, features

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx



def load_data_old(path="./data/", dataset="cora", device=None):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    # path = osp.join(path,dataset)
    idx_features_labels = np.genfromtxt("{}/{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}/{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)

    # adj = normalize(adj + sp.eye(adj.shape[0]))
    # adj = adj + sp.eye(adj.shape[0])
    # A = adj + sp.eye(adj.shape[0])
    # A = adj
    # (n, n) = A.shape
    # diags = A.sum(axis=1).flatten()
    # p, q = 0.0,-1
    # p_diags, q_diags = diags.astype(float), diags.astype(float) 
    # p_diags[0, np.diag_indices(n)] = np.float_power(p_diags, p)   
    # q_diags[0, np.diag_indices(n)] = np.float_power(q_diags, q) 
    # pD = sp.spdiags(p_diags, [0], n, n, format='csr')
    # qD = sp.spdiags(q_diags, [0], n, n, format='csr')
    # generalized_laplacian = pD + qD.dot(A)
    # generalized_laplacian = pD - qD.dot(A)
    # adj = generalized_laplacian
    # adj = normalize(generalized_laplacian)
    # adj = sparse_mx_to_torch_sparse_tensor(generalized_laplacian)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense())).to(device)
    labels = torch.LongTensor(np.where(labels)[1]).to(device)

    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)

    # # make generalized laplacian 
    # temp_adj = adj + torch.eye(adj.shape[0]).to(device)
    # diags = temp_adj.sum(1)
    # n = adj.shape[0]
    # identity = torch.eye(n).to(device)


    # p_diags = torch.zeros([n,n]).to(device)
    # q_diags = torch.zeros([n,n]).to(device)
    # r_diags = torch.zeros([n,n]).to(device)

    # ind = np.diag_indices(n)
    # p_diags[ind[0], ind[1]] = diags.clone() ** (-1.2456) 
    # q_diags[ind[0], ind[1]] = diags.clone() ** (0.018)
    # r_diags[ind[0], ind[1]] = diags.clone() ** (0.24)        
    # # gen_adj = p_diags - self.s*(q_diags.mm(temp_adj)).mm(r_diags)
    # gen_adj = p_diags - (q_diags.mm(temp_adj)).mm(r_diags) + (-3.95) * identity

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    # return gen_adj, features, labels, idx_train, idx_val, idx_test

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to_dense()


def condition_number(mx):
    _, spectral_norm, _ = sp.linalg.svds(mx.detach().numpy(),k=1)
    _, spectral_norm_inv, _ = sp.linalg.svds(torch.inverse(mx).detach().numpy(),k=1)
    # print(f'condition number: {spectral_norm * spectral_norm_inv}')
    # return torch.norm(mx, 2) * torch.norm(torch.inverse(mx), 2)
    return spectral_norm * spectral_norm_inv

class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

