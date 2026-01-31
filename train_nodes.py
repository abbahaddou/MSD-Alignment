import time
import argparse
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.utils import degree, contains_self_loops
from torch_geometric.data import Data
from torch_geometric.utils import loop
import torch.optim as optim
from utils import load_data, load_data_old, accuracy, condition_number
from models.gcn_conv import GCN_node_classification
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import networkx as nx
from numpy import dot
import wandb
import sys


########################################################################################
# Train and test functions 
########################################################################################

def train(epoch):
    t = time.time()
    model.train()
    
    optimizer.zero_grad()
    output = model(features.float(), edge_index.detach() ,edge_index_id.detach(), diags.float()) 
    loss_train = criterion(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    acc_train = accuracy(output[idx_train], labels[idx_train])
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features.float(), edge_index , edge_index_id.detach(),diags.float())

    loss_val = criterion(output[idx_val], labels[idx_val]).detach()
    acc_val = accuracy(output[idx_val], labels[idx_val])
    # cond_values.append(condition_number(model.gen_adj).item())

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t)
        #   'cond: {:.1f}'.format(condition_number(model.gen_adj))
          )

    whole_state = {
        'epoch': epoch,
        'model_state_dict': {key:val.clone() for key,val in model.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_train': loss_train.detach(),
        'loss_val': loss_val.detach(),
        'acc_train': acc_train.detach(),
        'acc_val': acc_val.detach(),
        }
    return whole_state


def test():
    model.eval()
    output = model(features.float(), edge_index ,edge_index_id.detach(), diags.float())
    loss_test = criterion(output[idx_test], labels[idx_test])
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()


########################################################################################
# Parse arguments 
########################################################################################

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default='./data/',  help='Directory of datasets; default is ./data/')
parser.add_argument('--gso', type=str, default='NORMADJ',  choices=['ADJ', 'UNORMLAPLACIAN','SINGLESSLAPLACIAN','RWLAPLACIAN','SYMLAPLACIAN','NORMADJ', 'MEANAGG', 'Test'])
parser.add_argument('--outdir', type=str, default='./exps/', help='Directory of experiments output; default is ./exps/')
parser.add_argument('--dataset', type=str, default='Wisconsin', help='Dataset name; default is Cora' ,choices=["Cora" ,"CiteSeer", "PubMed", "CS", "ogbn-arxiv", 'Reddit', 'genius', 'Penn94','Computers','Photo', "Physics", 'deezer-europe', 'arxiv-year' , "imdb" , "chameleon", "crocodile", "squirrel","Cornell", "Texas", "Wisconsin"])
parser.add_argument('--device', type=int, default=1,help='Set CUDA device number; if set to -1, disables cuda.')    
parser.add_argument('--fastmode', action='store_true', default=False,help='Validate during training pass.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--exp_lr', type=float, default=0.005,help='Initial learning rate for exponential parameters.')
parser.add_argument('--lr_patience', type=float, default=50, help='Number of epochs waiting for the next lr decay.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=512,help='Number of hidden units.')
parser.add_argument('--num_layers', type=int, default=2,  help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate (1 - keep probability).')
parser.add_argument('--genlap', action='store_true', default=False,help='Utilization of GenLap')
parser.add_argument('--use_wandb', type= bool,default = False , choices=[True, False])
args = parser.parse_args()

if args.dataset == 'ogbn-arxiv' :
    args.hidden = 512
if args.dataset == 'Cora' :
    args.lr = 0.01
    args.hidden = 64
    args.dropout = 0.8

elif args.dataset == 'CiteSeer' :
    args.lr = 0.01
    args.hidden = 64
    args.dropout = 0.4
elif args.dataset == 'PubMed' :
    args.lr = 0.01
    args.hidden = 64
    args.dropout = 0.2
elif args.dataset == 'CS' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.4
elif args.dataset == 'genius' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.8
elif args.dataset == 'Penn94' :
    args.lr = 0.01
    args.hidden = 64
    args.dropout = 0.2
elif args.dataset == 'Computers' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.2
elif args.dataset == 'Photo' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.6

elif args.dataset == 'Physics' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.4
elif args.dataset == 'twitch-gamers' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.2   
elif args.dataset == 'deezer-europe' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.2     
elif args.dataset == 'imdb' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.2     
elif args.dataset == 'chameleon' :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.2  
elif args.dataset == "Cornell" or args.dataset ==  "Texas" or args.dataset ==  "Wisconsin" :
    args.lr = 0.01
    args.hidden = 512
    args.dropout = 0.2  

device = torch.device('cuda:'+str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
#np.random.seed(args.seed)
#torch.manual_seed(args.seed)
#if args.device > 0:
#    torch.cuda.manual_seed(args.seed)
expfolder = osp.join(args.outdir, args.dataset)
expfolder = osp.join(expfolder,datetime.now().strftime("%Y%m%d_%H%M%S"))
Path(expfolder).mkdir(parents=True, exist_ok=True)

########################################################################################
# Data loading and model setup 
########################################################################################
all_test_acc = []
for t_ in range(10) :
    adj, features, labels, idx_train, idx_val, idx_test = load_data(path = args.datadir, dataset_name = args.dataset,device =  device)

    # !!!!!!!!
    # !!!!!!!!
    # ADDING self loops is learnable in the PGSO
    n = features.size(0)
    #  adj = adj + sp.identity(n)

    edge_index_without_loops, edge_weight_without_loops = from_scipy_sparse_matrix(adj)
    edge_index_without_loops = edge_index_without_loops.to(features.device)

    row, col = edge_index_without_loops
    deg = degree(col, n, dtype=features.dtype)
    diags = deg


    

    # Now that we have computed the centralities without self loops, we add self loops in the edge index

    edge_index, edge_weight = from_scipy_sparse_matrix(adj)

    edge_index = edge_index.to(features.device)
    row, col = edge_index
    diags = diags.clone()
    edge_index_id, edge_weight_id = from_scipy_sparse_matrix(sp.identity(n))
    edge_index_id = edge_index_id .to(features.device)
    norm_diag  = torch.tensor([diags[edge_index[0,i].item()].item() if edge_index[0,i].item() == edge_index[1,i].item() else 0 for i in range(edge_index.size(-1))  ]).to(diags.device)
    # norm_identity  = torch.tensor([1 if edge_index[0,i].item() == edge_index[1,i].item() else 0 for i in range(edge_index.size(-1))  ]).to(diags.device)


    # Model and optimizer
    model = GCN_node_classification(input_dim=features.shape[1],
                hidden_dim=args.hidden,
                output_dim=labels.max().item() + 1,
                num_layers=args.num_layers,
                dropout=args.dropout, gso = args.gso).to(device)
    
    # Freezing the weights of the PGSO
    model.m1.requires_grad = False
    model.m2.requires_grad = False
    model.m3.requires_grad = False
    model.e1.requires_grad = False
    model.e2.requires_grad = False
    model.e3.requires_grad = False
    model.a.requires_grad = False

    # Exponential parameters have a different learning rate than other multiplicative parameters
    exp_param_list = ['e1', 'e2' , 'e3']
    exp_params = list(filter(lambda kv: kv[0] in exp_param_list, model.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in exp_param_list, model.named_parameters()))
    exp_params = [param[1] for param in exp_params]
    base_params = [param[1] for param in base_params]
    
    optimizer = optim.Adam([
                            {'params': base_params, 'lr':args.lr},
                            {'params': exp_params, 'lr': args.exp_lr}
                            ], lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    lr_scheduler = None
    if args.lr_patience > 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_patience, gamma=0.6)

    cond_values = []

    # Train model
    t_total = time.time() 
    m1_values, m2_values, e1_values, e2_values = [], [], [], []

    states = []
    for epoch in range(1,args.epochs+1):
        state = train(epoch)
        states.append(state)

        if args.lr_patience > 0:
            lr_scheduler.step()

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    acc_test =  test()
    all_test_acc.append(acc_test)

print('test_accuracy : {}'.format(all_test_acc) )

