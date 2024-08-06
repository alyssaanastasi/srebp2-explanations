# This file runs the experiments where the edge mask = None

import setup
import pickle
import sys


gnn_comm_node_importance = setup.GNN_comm_feature_experiment(setup.k_vals, setup.test_nodes)

with open('gnn_comm_node_importance.pkl', 'wb') as file:
    pickle.dump(gnn_comm_node_importance, file) 

gm_attr_node_importance = setup.GM_attr_feature_experiment(setup.k_vals, setup.test_nodes)

with open('gm_attr_node_importance.pkl', 'wb') as file:
    pickle.dump(gm_attr_node_importance, file)

gm_comm_node_importance = setup.GM_comm_feature_experiment(setup.k_vals, setup.test_nodes)

with open('gm_comm_node_importance.pkl', 'wb') as file:
    pickle.dump(gm_comm_node_importance, file)

    
