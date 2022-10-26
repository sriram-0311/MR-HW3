from turtle import width
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
import math
from pyparsing import alphas
import scipy

from torch import is_tensor

neighbor_map = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]

def load_occupancy_map():
	# Read image from disk using PIL
	occupancy_map_img = Image.open('/Users/anushsriramramesh/Library/CloudStorage/OneDrive-NortheasternUniversity/MR/MR-HW3/Utilities/occupancy_map.png')
	occupancy_grid = (np.asarray(occupancy_map_img) > 0).astype(int)
	return occupancy_grid

def rejectionSampler(occupancyGrid):
    uniformProposalFunction = scipy.stats.uniform()
    # for x in range(occupancyGrid.shape[0]):
    #     for y in range(occupancyGrid.shape[1]):
    #         sampleSetVertices.append((x,y))

    #sampleSetVertices = [(i,j) for i in range(occupancyGrid[0]) for j in range(occupancyGrid[1])]

    while True:
        x = int(np.random.uniform(0, occupancyGrid.shape[0]))
        y = int(np.random.uniform(0, occupancyGrid.shape[1]))
        if occupancyGrid[x][y] == 1:
            return (x,y)
        else:
            continue

def Neighbours(graph, v):
    neighbors = []
    for x,y in neighbor_map:
        search_point = (v[0]+x,v[1]+y)
        neighbors.append(search_point)
    return neighbors

def reachabilityCheck(occupancyGrid, v1, v2):
    # print("v1: ", v1)
    # print("v2: ", v2)
    current = v1
    while current != v2:
        neighbors = Neighbours(occupancyGrid, current)
        current = min(neighbors, key=lambda x: math.dist(x, v2))
        if occupancyGrid[current] == 0:
            print("no path found..")
            return False
        else:
            continue
    print("Path found..")
    return True

class PRM():
    def __init__(self, N, dmax) -> None:
        super().__init__()
        self.graph = nx.Graph()
        self.vertices = []
        self.edges = []
        self.occupancy_grid = load_occupancy_map()
        self.NumberOfSamples = N
        self.maxDistance = dmax
        self.counter = 0

    def build_graph(self):
        for i in range(self.NumberOfSamples):
            sampleVertex = rejectionSampler(self.occupancy_grid)
            self.addVertex(sampleVertex, i)
        #print("sampleV", sampleSetVertices)
        print("graph", self.graph.nodes[0])

    def addVertex(self, sampleVertex, itr):
        #print("sampleVertex", sampleVertex)
        self.graph.add_node(self.counter, pos=sampleVertex)
        self.counter = self.counter + 1
        # print("graph", self.graph.nodes[itr]['pos'])
        # print("graph", self.graph.nodes[itr]['pos'])
        if self.counter > 1:
            for v in self.graph.nodes:
                #print("v", v)
                #if self.graph.
                if (self.graph.nodes[v]['pos'] != sampleVertex) and (math.dist(self.graph.nodes[v]['pos'], sampleVertex) <= self.maxDistance):
                    if (reachabilityCheck(self.occupancy_grid, self.graph.nodes[v]['pos'], sampleVertex)):
                        self.graph.add_edge(v,itr, weight=math.dist(sampleVertex, self.graph.nodes[v]['pos']))
                        return
                    else:
                        continue
                else:
                    continue
            return None
        else:
            return

if __name__ == "__main__":
    prm = PRM(2500, 75)
    prm.build_graph()
    grid = load_occupancy_map()
    grid[np.where(grid>0)] = 255
    #plt.imshow(grid, cmap='Dark2',  origin='upper')
    # for v in prm.graph.nodes:
    #     plt.plot(prm.graph.nodes[v]['pos'][0], prm.graph.nodes[v]['pos'][1], 'r.',alpha=0.1)
    nx.draw_networkx(prm.graph, pos=nx.get_node_attributes(prm.graph, 'pos'), with_labels=False, width=0.2, node_size=0.2)
    plt.show()
