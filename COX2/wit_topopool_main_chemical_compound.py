import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.datasets import TUDataset

from tqdm import tqdm

from util import separate_TUDataset
from models.witness_graphcnn import TenGCN

criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.y for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()      

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(model, device, train_graphs, test_graphs):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.y for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.y for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))
    return acc_train, acc_test

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="COX2",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of GCN layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    # below are new model specific arguments
    parser.add_argument('--sublevel_filtration_methods', nargs='+', type=str, default=['degree','betweenness','communicability','eigenvector','closeness'],
    					help='Methods for sublevel filtration on PDs')
    parser.add_argument('--tensor_layer_type', type = str, default = "TCL", choices=["TCL","TRL"],
                                        help='Tensor layer type: TCL/TRL')
    parser.add_argument('--PI_dim', type=int, default=50,
                        help='PI size: PI_dim * PI_dim')
    parser.add_argument('--node_pooling', action="store_false",
    					help='node pooling based on node scores')
    args = parser.parse_args()

    #set up seeds and gpu device
    random_seed = 0
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    graphs = TUDataset(root='/tmp/' + args.dataset, name=args.dataset)
    num_classes = graphs.num_classes

    train_graphs, test_graphs = separate_TUDataset(graphs, args.seed, args.fold_idx)

    model = TenGCN(args.num_layers, args.num_mlp_layers, train_graphs[0].x.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.sublevel_filtration_methods, args.tensor_layer_type, args.PI_dim, args.node_pooling, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    max_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print("Current epoch is:", epoch)

        avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
        scheduler.step() # only takes effect at the specified step_size
        acc_train, acc_test = test(model, device, train_graphs, test_graphs)

        max_acc = max(max_acc, acc_test)

        if not args.filename == "":
            with open(args.filename, 'a+') as f:
                f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                f.write("\n")

    with open('acc_results.txt', 'a+') as f:
        f.write(str(args.dataset) + ' ' + str(max_acc) + '\n')
    

if __name__ == '__main__':
    main()