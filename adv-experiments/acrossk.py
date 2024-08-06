# This file runs the experiment looking at explanations across values of k for all of the explanation methods. 

import setup 
import pickle

gnn_attr_acrossk = setup.gnn_attr_acrossk(setup.k_vals, setup.test_nodes)

with open('gnn_attr_acrossk.pkl', 'wb') as file:
    pickle.dump(gnn_attr_acrossk, file) 

gnn_comm_acrossk = setup.gnn_comm_acrossk(setup.k_vals, setup.test_nodes)

with open('gnn_comm_acrossk.pkl', 'wb') as file:
    pickle.dump(gnn_comm_acrossk, file) 

pg_acrossk = setup.pg_acrossk(setup.k_vals, setup.test_nodes)

with open('pg_acrossk.pkl', 'wb') as file:
    pickle.dump(pg_acrossk, file) 

gm_attr_acrossk = setup.GM_attr_acrossk(setup.k_vals, setup.test_nodes)

with open('gm_attr_acrossk.pkl', 'wb') as file:
    pickle.dump(gm_attr_acrossk, file)

gm_comm_acrossk = setup.GM_comm_acrossk(setup.k_vals, setup.test_nodes)

with open('gm_comm_acrossk.pkl', 'wb') as file:
    pickle.dump(gm_comm_acrossk, file)

