import dgl
import torch
import torch.nn.functional as F
import numpy as np
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import dgl.nn.pytorch as dglnn
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from torch.utils.data import Subset
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt




# dataset = dgl.data.CSVDataset('./dgl-csv-dev-map-all-parallel-patterns')
dataset_pg = dgl.data.CSVDataset('./dgl-csv-rose')

test_idx = []
train_idx = []
ind_count = 0
# for data_ll in dataset_ll:
#     ll_file_name = data_ll[3]['file_name'].split('/')[3]
#     if 'cg-' in ll_file_name and (('cg-10' not in ll_file_name) and ('cg-12' not in ll_file_name)\
#             and ('cg-15' not in ll_file_name) and ('cg-17' not in ll_file_name)):
#         test_idx.append(ind_count)
#     else :
#         print(ll_file_name)
#         train_idx.append(ind_count)
#     ind_count += 1

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv4 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv5 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv6 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs is features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv4(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv5(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv6(graph, h)
        return h


class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['feat']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            # pass
            # Calculate graph representation by average readout.
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
                # pass
            return self.classify(hg)

whole_exp = 0

prev_max_num_correct = -1000

for whole_exp in tqdm(range(1)):

    num_examples = len(dataset_pg)

    output_final = []
    label_final = []


    # new added
    dataset_indices = list(range(num_examples))
    np.random.shuffle(dataset_indices)
    test_split_index = 90


    train_idx, test_idx = dataset_indices[test_split_index:], dataset_indices[:test_split_index]
    train_sampler = SubsetRandomSampler(train_idx)
    dataset_test = Subset(dataset_pg, test_idx)

    etypes = [('control', 'control', 'control'), ('control', 'call', 'control'), ('control', 'data', 'variable'), ('variable', 'data', 'control')]
    class_names = ['Private', 'Reduction', 'Non-Parallel']

    train_dataloader_pg = GraphDataLoader(dataset_pg, shuffle=False, batch_size=100, sampler=train_sampler)
    test_dataloader_pg = GraphDataLoader(dataset_test, batch_size=100)


    model_pg = HeteroClassifier(120, 64, 3, etypes)
    # model_pg = torch.load('./Combined-Training-Models/model-300-2.pt')
    opt = torch.optim.Adam(model_pg.parameters(), lr=0.01)
    total_loss = 0
    loss_list = []
    epoch_list = []
    for epoch in tqdm(range(120)):
        total_loss = 0
        for batched_graph, labels in train_dataloader_pg:
            logits = model_pg(batched_graph)
            flattened_labels = labels.flatten()
            loss = F.cross_entropy(logits, flattened_labels)
            total_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        loss_list.append(total_loss)
        epoch_list.append(epoch)


    num_correct = 0
    num_tests = 0
    total_pred = []
    total_label = []

    for batched_graph, labels in test_dataloader_pg:
        pred = model_pg(batched_graph)

        pred_numpy = pred.detach().numpy()

        for ind_pred, ind_label in zip(pred_numpy, labels):
            if np.argmax(ind_pred) == ind_label:
                num_correct += 1
            total_pred.append(np.argmax(ind_pred))

        num_tests += len(labels)

        label_tmp = labels.data.cpu().numpy()
        total_label.extend(label_tmp)

        label_final = labels
        output_final = total_pred


    print('Report ', whole_exp)
    print('PG')
    print(classification_report(total_label, total_pred, target_names=class_names))

    cf_matrix = confusion_matrix(total_label, total_pred)

    print(cf_matrix)

