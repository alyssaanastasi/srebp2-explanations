# ## Revised stability calculations
# - Can we do 5  runs for each of 10  instances?
#     - Calculate average Jaccard similarity for each instance
#     - Make a box plot or something similar to show distribution of similarity measures across instances
#     - Per Yuriy’s suggestion, we could also calculate Ruzicka similarity (https://en.wikipedia.org/wiki/Jaccard_index) on soft masks


import setup
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pandas as pd

with open('pickle-files/gm_attr_stability.pkl', 'rb') as f:
    gm_attr_stability = pickle.load(f)

with open('pickle-files/gm_comm_stability.pkl', 'rb') as f:
    gm_comm_stability = pickle.load(f)

with open('pickle-files/pg_stability.pkl', 'rb') as f:
    pg_stability = pickle.load(f)

with open('pickle-files/gnn_attr_stability.pkl', 'rb') as f:
    gnn_attr_stability = pickle.load(f)

with open('pickle-files/gnn_comm_stability.pkl', 'rb') as f:
    gnn_comm_stability = pickle.load(f)


stability_studies = [gm_attr_stability, pg_stability, gnn_attr_stability]
names = ["Graph Mask", "PG Explainer", "GNN Explainer"]
# make dataframe of points for each stability trial
data = []
for i in range(len(stability_studies)):
    stab = stability_studies[i]
    explainer = names[i]
    for node in stab: 
        explanations = stab[node]['explanations']
        edge_jaccards = setup.get_jaccards(explanations)
        edge_avg_jacc = np.mean(edge_jaccards)
        if explainer != 'PG Explainer':
            feat_jaccards = setup.get_feat_jaccards(explanations)
            feat_avg_jacc = np.mean(feat_jaccards)
        else: 
            feat_avg_jacc = None
        data.append({'Explainer': explainer, "Average Edge Jaccard Similarity": edge_avg_jacc, "Average Feat Jaccard Similarity": feat_avg_jacc})

data = pd.DataFrame(data)

sns.set_theme(font_scale=1.2, style="white")
stripplot = sns.stripplot(data=data, x='Explainer', y='Average Edge Jaccard Similarity', order=['GNN Explainer', 'PG Explainer', 'Graph Mask'])
stripplot.set(xlabel=None, ylabel='Average Jaccard Similarity', title='Average Edge Similarity among Instances')
fig = stripplot.get_figure()
fig.savefig("edge_stab_stripplot.png", bbox_inches="tight")

plt.clf()
data2 = data[data['Explainer'] != 'PGExplainer']
stripplot = sns.stripplot(data=data2, x='Explainer', y='Average Feat Jaccard Similarity', order=['GNN Explainer', 'Graph Mask'])
stripplot.set(xlabel=None, ylabel='Average Jaccard Similarity', title='Average Feature Similarity among Instances')
stripplot.set(ylim=(0, 1.0))
fig = stripplot.get_figure()
fig.savefig("feat_stab_stripplot.png", bbox_inches="tight")


""" for node in gnn_attr_stability:
    explanations = gnn_attr_stability[node]['explanations']
    k = 0
    for explanation in explanations:
        try:
            explanation.visualize_graph(path=f"figures/gnn_attr_stab_{node}_{k}.png", backend='graphviz', node_labels=setup.node_series)
        except:
            pass
        k+=1 """

