""" Configuration settings
"""
import pickle
import os.path as path

global_path = path.dirname(path.abspath(__file__))

with open(path.join(global_path, "data/particle_data.ppo"), "rb") as ppofile:
    spec_data = pickle.load(ppofile, encoding='bytes')

new_spec_data = {}    
for datakey, dataval in spec_data.items():
    if type(datakey) is int:
        new_spec_data[datakey] = {bkey.decode('ascii'): dataval.get(bkey) for bkey in dataval.keys() }
    else:
        new_spec_data[datakey.decode('ascii')] = spec_data.get(datakey)
spec_data = new_spec_data

debug_level = 3 # defines the information output of the code

print_module = False

tau_dec_threshold = 0  # lifetime threshold to consider a species

max_A = 56  # maximum mass of the model. Cannot be greater than 208