# This file conducts the stability experiments for each explainer.
import setup 
import pickle


gnn_attr_stability = setup.stability_trial(setup.gnn_attr, setup.test_nodes)

with open('gnn_attr_stability.pkl', 'wb') as file:
    pickle.dump(gnn_attr_stability, file)

gnn_comm_stability = setup.stability_trial(setup.gnn_comm, setup.test_nodes)

with open('gnn_comm_stability.pkl', 'wb') as file:
    pickle.dump(gnn_comm_stability, file)

pg_stability = setup.pg_stability_trial(setup.pg, setup.test_nodes)

with open('pg_stability.pkl', 'wb') as file:
    pickle.dump(pg_stability, file)

gm_attr_stability = setup.stability_trial(setup.gm_attr, setup.test_nodes)

with open('gm_attr_stability.pkl', 'wb') as file:
    pickle.dump(gm_attr_stability, file)

gm_comm_stability = setup.stability_trial(setup.gm_comm, setup.test_nodes)

with open('gm_comm_stability.pkl', 'wb') as file:
    pickle.dump(gm_comm_stability, file)