import setup 
import pickle

gm_comm_stability = setup.stability_trial(setup.gm_comm, setup.test_nodes)

with open('gm_comm_stability.pkl', 'wb') as file:
    pickle.dump(gm_comm_stability, file)
