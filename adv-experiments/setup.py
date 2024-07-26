# %%
# install necessary packages
from torch_geometric.data import Data, DataLoader
from torch_geometric.explain import GNNExplainer,Explainer,GraphMaskExplainer,PGExplainer, DummyExplainer
import torch_geometric.transforms as T
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch.nn import Linear,Softmax
import os
from tqdm import tqdm, trange
import pickle
from torch_geometric.explain.metric import fidelity, characterization_score
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from stringdb_alias import HGNCMapper
import sys
import pandas as pd

# %%
class SimpleGAT_1(torch.nn.Module):
    """
    A graph attention network with 4 graph layers and 2 linear layers.
    Uses v2 of graph attention that provides dynamic instead of static attention.
    The graph layer dimension and number of attention heads can be specified.
    """
    #torch.device('mps')
    def __init__(self, dataset, dim=8, num_heads=4):
        super(SimpleGAT_1, self).__init__()
        torch.manual_seed(seed=123)
        self.conv1 = GATv2Conv(in_channels=dataset.num_features, out_channels=dim, heads=num_heads,edge_dim=dataset.edge_attr.shape[1])
        self.conv2 = GATv2Conv(in_channels=dim * num_heads, out_channels=dim, heads=num_heads,edge_dim=dataset.edge_attr.shape[1])
        self.lin1 = Linear(dim * num_heads,dim)
        self.lin2 = Linear(dim,1)

    def forward(self, x, edge_index,edge_attr):
        h = self.conv1(x, edge_index,edge_attr).relu()
        h = self.conv2(h, edge_index,edge_attr).relu()
        h = self.lin1(h).relu()
        #print(h)
        h = F.dropout(h, p=0.1, training=self.training)
        out = self.lin2(h)[:,0]
        out = torch.sigmoid(out)
        return out

# %%
data = torch.load('../new_SREBP2_2.pt')
model_path = '../SimpleGAT_1_model_lr5e-05_dp_0.0SREBP2.pth'
model = SimpleGAT_1(data,dim = 16)
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
model.eval()

# %%
with open('../feature_index.json') as json_file:
    feat_labels = json.load(json_file)

feat_labels = list(feat_labels.values())

# %%
# extract node labels and testing nodes from node order 

""" 
# 9606.ENSP00000289989 lowest False Negative / predicted value: 0.0281679704785347 / node index: 17397
# 9606.ENSP00000358777 highest True Positive / predicted value: 0.9995648264884949 / node index: 9005
# 9606.ENSP00000216099 lowest True Negative / predicted value: 7.5380116e-30 / node index: 8665
# 9606.ENSP00000357637 highest False Positive / predicted value: 0.9998927 / node index: 12633
# 9606.ENSP00000357226 furthest to target with label 0 / predicted value: 1.4533093e-07 / node index: 18233
# 9606.ENSP00000344741 closest to target with label 1 / predicted value: 0.8364509 / node index: 12257
# 9606.ENSP00000470087 closest to target with label 0 / predicted value: 0.3407591 / node index: 1674
# 9606.ENSP00000289989 furthest to target with label 1 / predicted value: 0.02816797 / node index: 17397
# 9606.ENSP00000348069: SREBF1  / predicted value: 0.55870503 / node index: 6334
# 9606.ENSP00000370695: EXCO1 / predicted value: 0.50875914/ node index: 1883
"""

with open('../node_order.pickle', 'rb') as f:
    node_order = pickle.load(f)

node_id_labels = list(node_order.keys())

test_nodes = [node_order['9606.ENSP00000289989'], # lowest False Negative / predicted value: 0.0281679704785347 / node index: 17397
                node_order['9606.ENSP00000358777'], # highest True Positive / predicted value: 0.9995648264884949 / node index: 9005
                node_order['9606.ENSP00000216099'], # lowest True Negative / predicted value: 7.5380116e-30 / node index: 8665
                node_order['9606.ENSP00000357637'],  # highest False Positive / predicted value: 0.9998927 / node index: 12633
                node_order['9606.ENSP00000357226'], # furthest to target with label 0 / predicted value: 1.4533093e-07 / node index: 18233
                node_order['9606.ENSP00000344741'], # closest to target with label 1 / predicted value: 0.8364509 / node index: 12257
                node_order['9606.ENSP00000470087'], # closest to target with label 0 / predicted value: 0.3407591 / node index: 1674
                node_order['9606.ENSP00000289989'], # furthest to target with label 1 / predicted value: 0.02816797 / node index: 17397
                node_order['9606.ENSP00000348069'], # SREBF1  / predicted value: 0.55870503 / node index: 6334
                node_order['9606.ENSP00000370695']] # : EXCO1 / predicted value: 0.50875914/ node index: 1883 ]

# %%
# get node labels for gene ids

mapper = HGNCMapper('../9606.protein.info.v11.5.txt.gz', '../9606.protein.aliases.v11.5.txt.gz')

node_series = mapper.get_hgnc_ids(node_id_labels)

# %%
node_labels = mapper.get_hgnc_ids(['9606.ENSP00000289989', # lowest False Negative / predicted value: 0.0281679704785347 / node index: 17397
                '9606.ENSP00000358777', # highest True Positive / predicted value: 0.9995648264884949 / node index: 9005
                '9606.ENSP00000216099', # lowest True Negative / predicted value: 7.5380116e-30 / node index: 8665
                '9606.ENSP00000357637',  # highest False Positive / predicted value: 0.9998927 / node index: 12633
                '9606.ENSP00000357226', # furthest to target with label 0 / predicted value: 1.4533093e-07 / node index: 18233
                '9606.ENSP00000344741', # closest to target with label 1 / predicted value: 0.8364509 / node index: 12257
                '9606.ENSP00000470087', # closest to target with label 0 / predicted value: 0.3407591 / node index: 1674
                '9606.ENSP00000289989', # furthest to target with label 1 / predicted value: 0.02816797 / node index: 17397
                '9606.ENSP00000348069', # SREBF1  / predicted value: 0.55870503 / node index: 6334
                '9606.ENSP00000370695'])

# %%
k_vals = [10, 20, 40, 80]

# %%
def pos_fidelity_score(fids):
    """
    Takes in list of fidelity scores across multiple instances, and returns overall fidelity score at the model level
    """
    return 1 - sum(fids)/len(fids)

def neg_fidelity_score(fids):
    return sum(fids) / len(fids)

def GNN_attr_feature_experiment(k_vals, test_nodes):
    # measure fidelity across k without edge mask
    wall_times = []
    process_times = []
    explanations = {k:[] for k in k_vals}
    fid_pos_overall = []
    fid_neg_overall = []

    for k in k_vals:
        sys.stdout.write(f"GNN ATTR: k={k}")
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type=None,
            threshold_config=dict(
                threshold_type="topk",
                value=k
            ),
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )
        fid_pos = []
        fid_neg = []
        for node in test_nodes:
            sys.stdout.write(f"GNN ATTR k={k} AND node={node}")
            start = time.time()
            process_start = time.process_time()
            explanation = explainer(data.x, data.edge_index, index = node, edge_attr=data.edge_attr)
            process_end = time.process_time()
            end = time.time()
            fid = fidelity(explainer, explanation)
            fid_pos.append(fid[0])
            fid_neg.append(fid[1])
            wall_times.append(end - start)
            process_times.append(process_end - process_start)
            explanations[k].append(explanation)
            # explanation.visualize_subgraph(path=f"trial_figs/{explainer}_{node}.pdf", backend='graphviz')
        fid_pos_overall.append(pos_fidelity_score(fid_pos))
        fid_neg_overall.append(neg_fidelity_score(fid_neg))
        

    return {'positive_fid' : fid_pos_overall, 'negative_fid': fid_neg_overall, 'wall_times': wall_times, 'process_times': process_times, 'explanations': explanations}

def GNN_comm_feature_experiment(k_vals, test_nodes):
    # measure fidelity across k without edge mask
    wall_times = []
    process_times = []
    explanations = {k:[] for k in k_vals}
    fid_pos_overall = []
    fid_neg_overall = []

    for k in k_vals:
        sys.stdout.write(f"GNN COMM AT k={k}")
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='common_attributes',
            edge_mask_type=None,
            threshold_config=dict(
                threshold_type="topk",
                value=k
            ),
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )
        fid_pos = []
        fid_neg = []
        for node in test_nodes:
            sys.stdout.write(f"GNN COMM AT k={k} AND node={node}")
            start = time.time()
            process_start = time.process_time()
            explanation = explainer(data.x, data.edge_index, index = node, edge_attr=data.edge_attr)
            process_end = time.process_time()
            end = time.time()
            fid = fidelity(explainer, explanation)
            fid_pos.append(fid[0])
            fid_neg.append(fid[1])
            wall_times.append(end - start)
            process_times.append(process_end - process_start)
            explanations[k].append(explanation)
            # explanation.visualize_subgraph(path=f"trial_figs/{explainer}_{node}.pdf", backend='graphviz')
        fid_pos_overall.append(pos_fidelity_score(fid_pos))
        fid_neg_overall.append(neg_fidelity_score(fid_neg))
        

    return {'positive_fid' : fid_pos_overall, 'negative_fid': fid_neg_overall, 'wall_times': wall_times, 'process_times': process_times, 'explanations': explanations}

# %%
def GM_attr_feature_experiment(k_vals, test_nodes):
    # measure fidelity across k without edge mask
    wall_times = []
    process_times = []
    explanations = {k:[] for k in k_vals}
    fid_pos_overall = []
    fid_neg_overall = []

    for k in k_vals:
        sys.stdout.write(f"GM ATTR AT k={k}")
        explainer = Explainer(
            model=model,
            algorithm=GraphMaskExplainer(2, epochs=5),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type=None,
            threshold_config=dict(
                threshold_type="topk",
                value=k
            ),
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )
        fid_pos = []
        fid_neg = []
        for node in test_nodes:
            sys.stdout.write(f"GM ATTR AT k={k} AND node={node}")
            start = time.time()
            process_start = time.process_time()
            explanation = explainer(data.x, data.edge_index, index = node, edge_attr=data.edge_attr)
            process_end = time.process_time()
            end = time.time()
            fid = fidelity(explainer, explanation)
            fid_pos.append(fid[0])
            fid_neg.append(fid[1])
            wall_times.append(end - start)
            process_times.append(process_end - process_start)
            explanations[k].append(explanation)
            # explanation.visualize_subgraph(path=f"trial_figs/{explainer}_{node}.pdf", backend='graphviz')
        fid_pos_overall.append(pos_fidelity_score(fid_pos))
        fid_neg_overall.append(neg_fidelity_score(fid_neg))
        

    return {'positive_fid' : fid_pos_overall, 'negative_fid': fid_neg_overall, 'wall_times': wall_times, 'process_times': process_times, 'explanations': explanations}

# %%
def GM_comm_feature_experiment(k_vals, test_nodes):
    # measure fidelity across k without edge mask
    wall_times = []
    process_times = []
    explanations = {k:[] for k in k_vals}
    fid_pos_overall = []
    fid_neg_overall = []

    for k in k_vals:
        sys.stdout.write(f"GM COMM AT k={k}")
        explainer = Explainer(
            model=model,
            algorithm=GraphMaskExplainer(2, epochs=5),
            explanation_type='model',
            node_mask_type='common_attributes',
            edge_mask_type=None,
            threshold_config=dict(
                threshold_type="topk",
                value=k
            ),
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )
        fid_pos = []
        fid_neg = []
        for node in test_nodes:
            sys.stdout.write(f"GM COMM AT k={k} AND node={node}")
            start = time.time()
            process_start = time.process_time()
            explanation = explainer(data.x, data.edge_index, index = node, edge_attr=data.edge_attr)
            process_end = time.process_time()
            end = time.time()
            fid = fidelity(explainer, explanation)
            fid_pos.append(fid[0])
            fid_neg.append(fid[1])
            wall_times.append(end - start)
            process_times.append(process_end - process_start)
            explanations[k].append(explanation)
            # explanation.visualize_subgraph(path=f"trial_figs/{explainer}_{node}.pdf", backend='graphviz')
        fid_pos_overall.append(pos_fidelity_score(fid_pos))
        fid_neg_overall.append(neg_fidelity_score(fid_neg))
        

    return {'positive_fid' : fid_pos_overall, 'negative_fid': fid_neg_overall, 'wall_times': wall_times, 'process_times': process_times, 'explanations': explanations}

# %% [markdown]
# ## Plot fidelity curves for subgraphs with various values of k (10, 20, 40, 80)
# 

# %%
def gnn_attr_acrossk(k_vals, test_nodes):
    wall_times = []
    process_times = []
    explanations = {k:[] for k in k_vals}
    fid_pos_overall = []
    fid_neg_overall = []

    for k in k_vals:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            threshold_config=dict(
                threshold_type="topk",
                value=k
            ),
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )
        fid_pos = []
        fid_neg = []
        for node in test_nodes:
            start = time.time()
            process_start = time.process_time()
            explanation = explainer(data.x, data.edge_index, index = node, edge_attr=data.edge_attr)
            process_end = time.process_time()
            end = time.time()
            fid = fidelity(explainer, explanation)
            fid_pos.append(fid[0])
            fid_neg.append(fid[1])
            wall_times.append(end - start)
            process_times.append(process_end - process_start)
            explanations[k].append(explanation)
            # explanation.visualize_subgraph(path=f"trial_figs/{explainer}_{node}.pdf", backend='graphviz')
        fid_pos_overall.append(pos_fidelity_score(fid_pos))
        fid_neg_overall.append(neg_fidelity_score(fid_neg))
        

    return {'positive_fid' : fid_pos_overall, 'negative_fid': fid_neg_overall, 'wall_times': wall_times, 'process_times': process_times, 'explanations': explanations}

# %%
def gnn_comm_acrossk(k_vals, test_nodes):
    wall_times = []
    process_times = []
    explanations = {k:[] for k in k_vals}
    fid_pos_overall = []
    fid_neg_overall = []

    for k in k_vals:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='common_attributes',
            edge_mask_type='object',
            threshold_config=dict(
                threshold_type="topk",
                value=k
            ),
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )
        fid_pos = []
        fid_neg = []
        for node in test_nodes:
            start = time.time()
            process_start = time.process_time()
            explanation = explainer(data.x, data.edge_index, index = node, edge_attr=data.edge_attr)
            process_end = time.process_time()
            end = time.time()
            fid = fidelity(explainer, explanation)
            fid_pos.append(fid[0])
            fid_neg.append(fid[1])
            wall_times.append(end - start)
            process_times.append(process_end - process_start)
            explanations[k].append(explanation)
            # explanation.visualize_subgraph(path=f"trial_figs/{explainer}_{node}.pdf", backend='graphviz')
        fid_pos_overall.append(pos_fidelity_score(fid_pos))
        fid_neg_overall.append(neg_fidelity_score(fid_neg))
        

    return {'positive_fid' : fid_pos_overall, 'negative_fid': fid_neg_overall, 'wall_times': wall_times, 'process_times': process_times, 'explanations': explanations}

# %%
def pg_acrossk(k_vals, test_nodes):
    wall_times = []
    process_times = []
    explanations = {k:[] for k in k_vals}
    fid_pos_overall = []
    fid_neg_overall = []

    for k in k_vals:
        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=30, lr=0.003),
            explanation_type='phenomenon',
            edge_mask_type='object',
            threshold_config=dict(
                threshold_type="topk",
                value=k 
            ),
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )
        fid_pos = []
        fid_neg = []
        for node in test_nodes:
            start = time.time()
            process_start = time.process_time()
            for epoch in range(30):
                loss = explainer.algorithm.train(epoch, model, data.x, data.edge_index,
                                         target=data.y, index=node, edge_attr = data.edge_attr)

            explanation = explainer(data.x, data.edge_index, target=data.y, index=node, edge_attr = data.edge_attr)            
            process_end = time.process_time()
            end = time.time()
            fid = fidelity(explainer, explanation)
            fid_pos.append(fid[0])
            fid_neg.append(fid[1])
            wall_times.append(end - start)
            process_times.append(process_end - process_start)
            explanations[k].append(explanation)
            # explanation.visualize_subgraph(path=f"trial_figs/{explainer}_{node}.pdf", backend='graphviz')
        fid_pos_overall.append(pos_fidelity_score(fid_pos))
        fid_neg_overall.append(neg_fidelity_score(fid_neg))
        

    return {'positive_fid' : fid_pos_overall, 'negative_fid': fid_neg_overall, 'wall_times': wall_times, 'process_times': process_times, 'explanations': explanations}

# %%
def GM_attr_acrossk(k_vals, test_nodes):
    # measure fidelity across k without edge mask
    wall_times = []
    process_times = []
    explanations = {k:[] for k in k_vals}
    fid_pos_overall = []
    fid_neg_overall = []

    for k in k_vals:
        explainer = Explainer(
            model=model,
            algorithm=GraphMaskExplainer(2, epochs=5),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            threshold_config=dict(
                threshold_type="topk",
                value=k
            ),
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )
        fid_pos = []
        fid_neg = []
        for node in test_nodes:
            start = time.time()
            process_start = time.process_time()
            explanation = explainer(data.x, data.edge_index, index = node, edge_attr=data.edge_attr)
            process_end = time.process_time()
            end = time.time()
            fid = fidelity(explainer, explanation)
            fid_pos.append(fid[0])
            fid_neg.append(fid[1])
            wall_times.append(end - start)
            process_times.append(process_end - process_start)
            explanations[k].append(explanation)
            # explanation.visualize_subgraph(path=f"trial_figs/{explainer}_{node}.pdf", backend='graphviz')
        fid_pos_overall.append(pos_fidelity_score(fid_pos))
        fid_neg_overall.append(neg_fidelity_score(fid_neg))
        

    return {'positive_fid' : fid_pos_overall, 'negative_fid': fid_neg_overall, 'wall_times': wall_times, 'process_times': process_times, 'explanations': explanations}

# %%
def GM_comm_acrossk(k_vals, test_nodes):
    # measure fidelity across k without edge mask
    wall_times = []
    process_times = []
    explanations = {k:[] for k in k_vals}
    fid_pos_overall = []
    fid_neg_overall = []

    for k in k_vals:
        explainer = Explainer(
            model=model,
            algorithm=GraphMaskExplainer(2, epochs=5),
            explanation_type='model',
            node_mask_type='common_attributes',
            edge_mask_type='object',
            threshold_config=dict(
                threshold_type="topk",
                value=k
            ),
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )
        fid_pos = []
        fid_neg = []
        for node in test_nodes:
            start = time.time()
            process_start = time.process_time()
            explanation = explainer(data.x, data.edge_index, index = node, edge_attr=data.edge_attr)
            process_end = time.process_time()
            end = time.time()
            fid = fidelity(explainer, explanation)
            fid_pos.append(fid[0])
            fid_neg.append(fid[1])
            wall_times.append(end - start)
            process_times.append(process_end - process_start)
            explanations[k].append(explanation)
            # explanation.visualize_subgraph(path=f"trial_figs/{explainer}_{node}.pdf", backend='graphviz')
        fid_pos_overall.append(pos_fidelity_score(fid_pos))
        fid_neg_overall.append(neg_fidelity_score(fid_neg))
        

    return {'positive_fid' : fid_pos_overall, 'negative_fid': fid_neg_overall, 'wall_times': wall_times, 'process_times': process_times, 'explanations': explanations}

# %% [markdown]
# ## Revised stability calculations
# - Can we do 5  runs for each of 10  instances?
#     - Calculate average Jaccard similarity for each instance
#     - Make a box plot or something similar to show distribution of similarity measures across instances
#     - Per Yuriy’s suggestion, we could also calculate Ruzicka similarity (https://en.wikipedia.org/wiki/Jaccard_index) on soft masks

# %%
# define explainers
gnn_attr = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    threshold_config=dict(
        threshold_type="topk",
        value=10 
    ),
    model_config=dict(
        mode='binary_classification',
        task_level='node',
        return_type='probs',
    ),
)

gnn_common = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='common_attributes',
    edge_mask_type='object',
    threshold_config=dict(
        threshold_type="topk",
        value=10 
    ),
    model_config=dict(
        mode='binary_classification',
        task_level='node',
        return_type='probs',
    ),
)

pg = Explainer(
    model=model,
    algorithm=PGExplainer(epochs=30, lr=0.003),
    explanation_type='phenomenon',
    edge_mask_type='object',
    threshold_config=dict(
        threshold_type="topk",
        value=10 
    ),
    model_config=dict(
        mode='binary_classification',
        task_level='node',
        return_type='probs',
    ),
)

gm_attr = Explainer(
    model=model,
    algorithm=GraphMaskExplainer(2, epochs=5),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    threshold_config=dict(
        threshold_type="topk",
        value=10 
    ),
    model_config=dict(
        mode='binary_classification',
        task_level='node',
        return_type='probs',
    ), 
)

gm_comm = Explainer(
    model=model,
    algorithm=GraphMaskExplainer(2, epochs=5),
    explanation_type='model',
    node_mask_type='common_attributes',
    edge_mask_type='object',
    threshold_config=dict(
        threshold_type="topk",
        value=10 
    ),
    model_config=dict(
        mode='binary_classification',
        task_level='node',
        return_type='probs',
    ),
)

explainers = [gnn_attr, gnn_common, pg, gm_attr, gm_comm]

# %%
def stability_trial(explainer, test_nodes):
    """
    One trial of a stability experiment - Using one explainer, finds an explanation on the input target node. 
    For each explanation, the positive fidelity score, negative fidelity score, wall runtime, process runtime, and explanation object are recorded.
    Returns a dictionary with keys {'positive_fid', 'negative_fid',  'wall_times', 'process_times', 'explanations'}
    This dict includes multiple explanations of the same explainer on the same node to explore stability 
    """
    results = {}

    for node in test_nodes:
        fid_pos = []
        fid_neg = []
        wall_times = []
        process_times = []
        explanations = []
        for i in range(5):
            start = time.time()
            process_start = time.process_time()
            explanation = explainer(data.x, data.edge_index, edge_attr=data.edge_attr, index = node)
            process_end = time.process_time()
            end = time.time()
            fid = fidelity(explainer, explanation)
            fid_pos.append(fid[0])
            fid_neg.append(fid[1])
            wall_times.append(end - start)
            process_times.append(process_end - process_start)
            explanations.append(explanation)
            # explanation.visualize_subgraph(path=f"trial_figs/{explainer}_{node}.pdf", backend='graphviz')
        results[node] = {'positive_fid' : fid_pos, 'negative_fid': fid_neg, 'wall_times': wall_times, 'process_times': process_times, 'explanations': explanations}

    return results

# %%
def pg_stability_trial(explainer, test_nodes):
    """
    One trial of a stability experiment - Using one explainer, finds an explanation on the input target node. 
    For each explanation, the positive fidelity score, negative fidelity score, wall runtime, process runtime, and explanation object are recorded.
    Returns a dictionary with keys {'positive_fid', 'negative_fid',  'wall_times', 'process_times', 'explanations'}
    This dict includes multiple explanations of the same explainer on the same node to explore stability 
    """
    results = {}

    for node in test_nodes:
        fid_pos = []
        fid_neg = []
        wall_times = []
        process_times = []
        explanations = []
        for i in range(5):
            start = time.time()
            process_start = time.process_time()
            for epoch in range(30):
                loss = explainer.algorithm.train(epoch, model, data.x, data.edge_index,
                                            target=data.y, index=node, edge_attr = data.edge_attr)

            explanation = explainer(data.x, data.edge_index, target=data.y, index=node, edge_attr = data.edge_attr)     
            process_end = time.process_time()
            end = time.time()
            fid = fidelity(explainer, explanation)
            fid_pos.append(fid[0])
            fid_neg.append(fid[1])
            wall_times.append(end - start)
            process_times.append(process_end - process_start)
            explanations.append(explanation)
            # explanation.visualize_subgraph(path=f"trial_figs/{explainer}_{node}.pdf", backend='graphviz')
        results[node] = {'positive_fid' : fid_pos, 'negative_fid': fid_neg, 'wall_times': wall_times, 'process_times': process_times, 'explanations': explanations}

    return results

# %%
def edge_jaccard(edge_mask1, edge_mask2):
    edge_mask1 = np.array(edge_mask1)
    edge_mask2 = np.array(edge_mask2)
    edge_mask1[edge_mask1 > 0] = 1
    edge_mask2[edge_mask2 > 0] = 1
    im1 = np.asarray(edge_mask1).astype(bool)
    im2 = np.asarray(edge_mask2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)

    union = np.logical_or(im1, im2)
    if float(union.sum()) == 0:
        return 1
    else:
        return intersection.sum() / float(union.sum())

def get_jaccards(explanations):
    jaccards = []
    for i in range(len(explanations)):
        for j in range(i, len(explanations)):
            if i == j:
                continue
            exp1 = i
            exp2 = j
            edge_mask1 = explanations[i].edge_mask
            edge_mask2 = explanations[j].edge_mask
            edge_sim = edge_jaccard(edge_mask1, edge_mask2)
            jaccards.append(edge_sim)
    return jaccards

# %% [markdown]
# # Dummy Explainer

# %%
dummy = Explainer(
    model=model,
    algorithm=DummyExplainer(),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type=None,
    threshold_config=dict(
        threshold_type="topk",
        value=10 
    ),
    model_config=dict(
        mode='binary_classification',
        task_level='node',
        return_type='probs',
    ),
)

# %%
def dummy_acrossk(k_vals, test_nodes):
    wall_times = []
    process_times = []
    explanations = {k:[] for k in k_vals}
    fid_pos_overall = []
    fid_neg_overall = []

    for k in k_vals:
        explainer = Explainer(
            model=model,
            algorithm=DummyExplainer(),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            threshold_config=dict(
                threshold_type="topk",
                value=k 
            ),
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )
        fid_pos = []
        fid_neg = []
        for node in test_nodes:
            start = time.time()
            process_start = time.process_time()
            explanation = explainer(data.x, data.edge_index, index = node, edge_attr=data.edge_attr)
            process_end = time.process_time()
            end = time.time()
            fid = fidelity(explainer, explanation)
            fid_pos.append(fid[0])
            fid_neg.append(fid[1])
            wall_times.append(end - start)
            process_times.append(process_end - process_start)
            explanations[k].append(explanation)
            # explanation.visualize_subgraph(path=f"trial_figs/{explainer}_{node}.pdf", backend='graphviz')
        fid_pos_overall.append(pos_fidelity_score(fid_pos))
        fid_neg_overall.append(neg_fidelity_score(fid_neg))
        

    return {'positive_fid' : fid_pos_overall, 'negative_fid': fid_neg_overall, 'wall_times': wall_times, 'process_times': process_times, 'explanations': explanations}


def dummy_node_importance(k_vals, test_nodes):
    wall_times = []
    process_times = []
    explanations = {k:[] for k in k_vals}
    fid_pos_overall = []
    fid_neg_overall = []

    for k in k_vals:
        explainer = Explainer(
            model=model,
            algorithm=DummyExplainer(),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type=None,
            threshold_config=dict(
                threshold_type="topk",
                value=k 
            ),
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )
        fid_pos = []
        fid_neg = []
        for node in test_nodes:
            start = time.time()
            process_start = time.process_time()
            explanation = explainer(data.x, data.edge_index, index = node, edge_attr = data.edge_attr)
            process_end = time.process_time()
            end = time.time()
            fid = fidelity(explainer, explanation)
            fid_pos.append(fid[0])
            fid_neg.append(fid[1])
            wall_times.append(end - start)
            process_times.append(process_end - process_start)
            explanations[k].append(explanation)
            # explanation.visualize_subgraph(path=f"trial_figs/{explainer}_{node}.pdf", backend='graphviz')
        fid_pos_overall.append(pos_fidelity_score(fid_pos))
        fid_neg_overall.append(neg_fidelity_score(fid_neg))
        

    return {'positive_fid' : fid_pos_overall, 'negative_fid': fid_neg_overall, 'wall_times': wall_times, 'process_times': process_times, 'explanations': explanations}


def get_k_features(explanation, k, feat_labels):
    node_mask = explanation.get('node_mask')
    if node_mask is None:
        raise ValueError("The attribute 'node_mask' is not available ")
    if node_mask.dim() != 2 or node_mask.size(1) <= 1:
        raise ValueError(f"Cannot compute feature importance for "
                            f"object-level 'node_mask' "
                            f"(got shape {node_mask.size()})")
    score = node_mask.sum(dim=0)
    score = score.cpu().numpy()
    df = pd.DataFrame({'score': score}, index=feat_labels)
    df['feature_name'] = df.index
    df = df.sort_values('score', ascending=False)
    df = df.round(decimals=3)
    return df.head(k)['feature_name'].tolist()

def feature_jaccard(features1, features2):
    intersection = len(list(set(features1).intersection(features2)))
    union = (len(set(features2)) + len(set(features2))) - intersection
    return float(intersection) / union

def get_feat_jaccards(explanations):
    jaccards = []
    for i in range(len(explanations)):
        for j in range(i, len(explanations)):
            if i == j:
                continue
            exp1 = i
            exp2 = j
            features1 = get_k_features(explanations[i], 10, feat_labels)
            features2 = get_k_features(explanations[j], 10, feat_labels)
            feat_sim = feature_jaccard(features1, features2)
            jaccards.append(feat_sim)
    return jaccards


# set up saliency explainer