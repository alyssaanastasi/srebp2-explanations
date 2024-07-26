import setup
import pickle

dummy_acrossk = setup.dummy_acrossk(setup.k_vals, setup.test_nodes)

with open('pickle-files/dummy_acrossk.pkl', 'wb') as file:
    pickle.dump(dummy_acrossk, file)