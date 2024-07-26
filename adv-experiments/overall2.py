import setup 
import pickle

pg_stability = setup.pg_stability_trial(setup.pg, setup.test_nodes)

with open('pg_stability.pkl', 'wb') as file:
    pickle.dump(pg_stability, file)