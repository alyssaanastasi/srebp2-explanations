# This file is to get the pickle files and visualize both the sub graph structure and feature importance. 

import setup
import pickle
import graphviz
import matplotlib.pyplot as plt

import copy
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor


with open('pickle-files/dummy_node_importance.pkl', 'rb') as f:
    dummy_node = pickle.load(f)

with open('pickle-files/dummy_acrossk.pkl', 'rb') as f:
    dummy_acrossk = pickle.load(f)

with open('pickle-files/gm_attr_node_importance.pkl', 'rb') as f:
    gm_attr_node = pickle.load(f)

with open('pickle-files/gm_attr_acrossk.pkl', 'rb') as f:
    gm_attr_acrossk = pickle.load(f)

with open('pickle-files/gm_comm_node_importance.pkl', 'rb') as f:
    gm_comm_node = pickle.load(f)

with open('pickle-files/gm_comm_acrossk.pkl', 'rb') as f:
    gm_comm_acrossk = pickle.load(f)

with open('pickle-files/gnn_attr_acrossk.pkl', 'rb') as f:
    gnn_attr_acrossk = pickle.load(f)

with open('pickle-files/gnn_attr_node_importance.pkl', 'rb') as f:
    gnn_attr_node = pickle.load(f)

with open('pickle-files/gnn_comm_acrossk.pkl', 'rb') as f:
    gnn_comm_acrossk = pickle.load(f)

with open('pickle-files/pg_acrossk.pkl', 'rb') as f:
    pg_acrossk = pickle.load(f)

def visualize_feature_importance(
    explainer,
    path: Optional[str] = None,
    feat_labels: Optional[List[str]] = None,
    top_k: Optional[int] = None,
):
    r"""Creates a bar plot of the node feature importances by summing up
    the node mask across all nodes.

    Args:
        path (str, optional): The path to where the plot is saved.
            If set to :obj:`None`, will visualize the plot on-the-fly.
            (default: :obj:`None`)
        feat_labels (List[str], optional): The labels of features.
            (default :obj:`None`)
        top_k (int, optional): Top k features to plot. If :obj:`None`
            plots all features. (default: :obj:`None`)
    """
    node_mask = explainer.node_mask

    if feat_labels is None:
        feat_labels = range(node_mask.size(1))

    score = node_mask.sum(dim=0)

    return _visualize_score(score, feat_labels, path, top_k)

def _visualize_score(
    score: torch.Tensor,
    labels: List[str],
    path: Optional[str] = None,
    top_k: Optional[int] = None,
):
    import matplotlib.pyplot as plt
    import pandas as pd

    if len(labels) != score.numel():
        raise ValueError(f"The number of labels (got {len(labels)}) must "
                         f"match the number of scores (got {score.numel()})")

    score = score.cpu().numpy()

    df = pd.DataFrame({'score': score}, index=labels)
    df = df.sort_values('score', ascending=False)
    df = df.round(decimals=3)

    if top_k is not None:
        df = df.head(top_k)
        title = f"Feature importance for top {len(df)} features"
    else:
        title = f"Feature importance for {len(df)} features"

    plt.rc('font', size=16) 
    ax = df.plot(
        kind='barh',
        figsize=(10, 7),
        title=title,
        ylabel='Feature label',
        xlim=[0, float(df['score'].max()) + 0.3],
        legend=False,
    )
    plt.gca().invert_yaxis()
    ax.bar_label(container=ax.containers[0], label_type='edge')

    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def explanation_graphs(explanations, title, k):
    ind = 0
    plt.clf()
    plt.rc('font', size=16)   
    plt.plot(setup.k_vals, explanations['positive_fid'])
    plt.ylim(bottom=0, top=1.1)
    plt.xlabel('Max Threshold on Edges and Features')
    plt.ylabel('Necessity')
    plt.xticks([10, 20, 40, 80])
    plt.savefig(f"figures/{title}_positive.png", bbox_inches='tight')

    plt.clf()
    plt.rc('font', size=16)  
    plt.plot(setup.k_vals, explanations['negative_fid'])
    plt.ylim(bottom=0, top=1.1)
    plt.xlabel('Max Threshold on Edges and Features')
    plt.ylabel('Sufficiency')
    plt.xticks([10, 20, 40, 80])
    plt.savefig(f"figures/{title}_negative.png", bbox_inches='tight')

    exp_10 = explanations['explanations'][k]
    for explanation in exp_10:
        try:
            explanation.visualize_graph(path=f"figures/{title}_{setup.node_labels[ind]}_{k}.png", backend='graphviz', node_labels=setup.node_series)
        except:
            pass
        try:
            plt.rc('font', size=16) 
            visualize_feature_importance(explainer=explanation, path=f"figures/{title}_{setup.node_labels[ind]}_{k}_features.png", top_k=k, feat_labels=setup.feat_labels)
        except:
            pass
        ind += 1



exps = [dummy_node, dummy_acrossk, gm_attr_node, gm_attr_acrossk, gm_comm_node, gm_comm_acrossk, gnn_attr_acrossk, gnn_attr_node, gnn_comm_acrossk, pg_acrossk]
names = ['dummy_node', 'dummy_acrossk', 'gm_attr_node', 'gm_attr_acrossk', 'gm_comm_node', 'gm_comm_acrossk', 'gnn_attr_acrossk', 'gnn_attr_node', 'gnn_comm_acrossk', 'pg_acrossk']


explanation_graphs(gnn_attr_acrossk, "gnn_attr_acrossk", 10)
explanation_graphs(gm_attr_acrossk, "gm_attr_acrossk", 10)
