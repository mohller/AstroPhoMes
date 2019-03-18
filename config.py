""" Configuration settings
"""
import pickle
import os.path as path


global_path = path.dirname(path.abspath(__file__))

spec_data = pickle.load(
    open(path.join(global_path, "data/particle_data.ppo"), "rb"))

debug_level = 3 # defines the information output of the code

print_module = False