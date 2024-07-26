import setup
import pickle
import matplotlib.pyplot as plt

def fidelity_plots(explanations, title, filename):
    plt.clf()
    try:
        plt.plot(setup.k_vals, explanations['positive_fid'])
    except:
    plt.ylim(bottom=0, top=1.1)
    plt.xlabel('K Values')
    plt.ylabel('Positive Fidelity')
    plt.xticks([10, 20, 40, 80])
    plt.title(f"Positive Fidelity of {title}")
    plt.savefig(f"figures/{filename}_positive.png")

    plt.clf()
    plt.plot(setup.k_vals, explanations['negative_fid'])
    plt.ylim(bottom=0, top=1.1)
    plt.xlabel('K Values')
    plt.ylabel('Negative Fidelity')
    plt.xticks([10, 20, 40, 80])
    plt.title(f"Negative Fidelity of {title}")
    plt.savefig(f"figures/{filename}_negative.png")


with open('pickle-files/dummy_node_importance.pkl', 'rb') as f:
    dummy_node = pickle.load(f)

with open('pickle-files/dummy_acrossk.pkl', 'rb') as f:
    dummy_acrossk = pickle.load(f)

with open('pickle-files/gm_attr_node_importance.pkl', 'rb') as f:
    gm_attr_node = pickle.load(f)


with open('pickle-files/gm_comm_node_importance.pkl', 'rb') as f:
    gm_comm_node = pickle.load(f)

with open('pickle-files/gnn_attr_acrossk.pkl', 'rb') as f:
    gnn_attr_acrossk = pickle.load(f)

with open('pickle-files/gnn_attr_node_importance.pkl', 'rb') as f:
    gnn_attr_node = pickle.load(f)


fidelity_plots(dummy_node, "Dummy Explainer with Attributes and EdgeMask=None", "dummy_node")
fidelity_plots(dummy_acrossk, "Dummy Explainer with Attributes Across K", "dummy_acrossk")
fidelity_plots(gm_attr_node, "GraphMask with Attributes and EdgeMask=None", "gm_attr_node")
fidelity_plots(gm_comm_node, "GraphMask with Common Attributes and EdgeMask=None", "gm_comm_node")
fidelity_plots(gnn_attr_acrossk, "GNNExplainer with Attributes Across K", "gnn_attr_acrossk")
fidelity_plots(gnn_attr_node, "GNNExplainer with Attributes with EdgeMask=None", "gnn_attr_node")