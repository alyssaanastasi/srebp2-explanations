import setup
import matplotlib.pyplot
import pickle
import pandas as pd
import seaborn as sns

# compile runtimes for all of each explainer


trials = []
with open('pickle-files/gm_attr_acrossk.pkl', 'rb') as f:
    gm_attr_acrossk = pickle.load(f)
    trials.append(gm_attr_acrossk)

with open('pickle-files/gm_comm_acrossk.pkl', 'rb') as f:
    gm_comm_acrossk = pickle.load(f)
    trials.append(gm_comm_acrossk)

with open('pickle-files/gnn_attr_acrossk.pkl', 'rb') as f:
    gnn_attr_acrossk = pickle.load(f)
    trials.append(gnn_attr_acrossk)

with open('pickle-files/gnn_comm_acrossk.pkl', 'rb') as f:
    gnn_comm_acrossk = pickle.load(f)
    trials.append(gnn_comm_acrossk)

with open('pickle-files/pg_acrossk.pkl', 'rb') as f:
    pg_acrossk = pickle.load(f)
    trials.append(pg_acrossk)

explainers = ["Graph Mask", "Graph Mask", "GNN Explainer", "GNN Explainer", "PG Explainer"]

# make df of runtimes
data = []
for i in range(len(trials)):
    runtimes = trials[i]['process_times']
    explainer = explainers[i]
    for time in runtimes:
        data.append({'Explainer': explainer, "Runtime": time / 60})

data = pd.DataFrame(data)

sns.set_theme(font_scale=1.2, style="white")
barplot = sns.barplot(data=data, x='Explainer', y='Runtime', errorbar="sd", order=['GNN Explainer', 'PG Explainer', 'Graph Mask'])
barplot.set(xlabel=None, ylabel='Runtime (Minutes)', title='Average Runtime of Explainers')
fig = barplot.get_figure()
fig.savefig("runtime_barplot.png", bbox_inches="tight")


