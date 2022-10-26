import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import uniform
import numpy as np
from PIL import Image

G = nx.Graph()
G.add_node(1,pos=(0, 0))
G.add_node(1,pos=(1, 0))
G.add_node(1,pos=(2, 0))

# x = range(0, 650)
# y = range(0,340)
# def load_occupancy_map():
# 	# Read image from disk using PIL
# 	occupancy_map_img = Image.open('/Users/anushsriramramesh/Library/CloudStorage/OneDrive-NortheasternUniversity/MR/MR-HW3/Utilities/occupancy_map.png')
# 	occupancy_grid = (np.asarray(occupancy_map_img) > 0).astype(int)
# 	return occupancy_grid


# x = int(np.random.uniform(0, load_occupancy_map().shape[0]+1))
# y = int(np.random.uniform(0, load_occupancy_map().shape[1]+1))

# print(x)
# print(y)

for i in G.nodes:
    print(G.nodes[i]['pos'])
    print(len(G.nodes))
