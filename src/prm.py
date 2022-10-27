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

#sampleSetVertices = [(i,j) for i in range(load_occupancy_map()[0]) for j in range(load_occupancy_map()[1])]

def rejectionSampler(occupancyGrid, freeVertices):
    uniformProposalFunction = scipy.stats.uniform()
    # for x in range(occupancyGrid.shape[0]):
    #     for y in range(occupancyGrid.shape[1]):
    #         sampleSetVertices.append((x,y))

    #sampleSetVertices = [(i,j) for i in range(occupancyGrid[0]) for j in range(occupancyGrid[1])]

    while True:
        x = int(np.random.uniform(0, occupancyGrid.shape[0]))
        y = int(np.random.uniform(0, occupancyGrid.shape[1]))
        if occupancyGrid[x][y] > 0:
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
            #print("no path found..")
            return False
        else:
            continue
    #print("Path found..")
    return True

class PRM():
    def __init__(self, N, dmax) -> None:
        super().__init__()
        self.graph = nx.Graph()
        self.vertices = []
        self.edges = []
        self.occupancy_grid = load_occupancy_map()
        print("occupancy_grid", self.occupancy_grid.shape[0])
        print("occupancy_grid", self.occupancy_grid.shape[1])
        self.NumberOfSamples = N
        self.maxDistance = dmax
        self.counter = 0

        self.freeVertices = []
        # for x in range(self.occupancy_grid[0]):
        #     for y in range(self.occupancy_grid[1]):
        #         if self.occupancy_grid[x][y] == 1:
        #             self.freeVertices.append((x,y))

    def build_graph(self, n):
        for i in range(n):
            sampleVertex = rejectionSampler(self.occupancy_grid, self.freeVertices)
            self.addVertex(sampleVertex, i)
        #print("sampleV", sampleSetVertices)
        print("graph", self.graph.nodes[0])

    def addVertex(self, sampleVertex, itr):
        #print("sampleVertex", sampleVertex)
        self.graph.add_node(itr, pos=sampleVertex)
        # print("graph", self.graph.nodes[itr]['pos'])
        # print("graph", self.graph.nodes[itr]['pos'])
        if itr > 1:
            for v in self.graph.nodes:
                #print("v", v)
                #if self.graph.
                if (self.graph.nodes[v]['pos'] != sampleVertex) and (math.dist(self.graph.nodes[v]['pos'], sampleVertex) <= self.maxDistance):
                    if (reachabilityCheck(self.occupancy_grid, self.graph.nodes[v]['pos'], sampleVertex)):
                        self.graph.add_edge(v,itr, weight=math.dist(sampleVertex, self.graph.nodes[v]['pos']))
                        self.vertices.append(sampleVertex)
                    else:
                        continue
                else:
                    continue
            return
        else:
            return

    def aStartSearch(self, start, goal):
        path = []
        if (start != goal):
            if start not in self.graph.nodes:
                print("Start or Goal not in graph")
                self.addVertex(start, self.graph.number_of_nodes() +1)
                print("added start")
            if goal not in self.graph.nodes:
                self.addVertex(goal, self.graph.number_of_nodes() +1)
                print("added goal")
            #path = nx.astar_path(self.graph, start, goal, heuristic=math.dist, weight='weight')
            return path
        else:
            return None

if __name__ == "__main__":
    prm = PRM(10000, 75)
    prm.build_graph(2000)
    path = prm.aStartSearch((600,200), (350, 400))
    if (635,140) in prm.graph.nodes:
        print("True")
    else:
        print("False")
    grid = load_occupancy_map()
    grid[np.where(grid>0)] = 255
    plt.imshow(np.transpose(grid), cmap="inferno", origin='lower')
    # print("graph", prm.graph.number_of_nodes())
    # print("graph", prm.graph.number_of_edges())
    # for v in prm.graph.nodes:
    #     plt.plot(prm.graph.nodes[v]['pos'][1], prm.graph.nodes[v]['pos'][0], 'r',alpha=0.2)
    nx.draw_networkx(prm.graph, pos=nx.get_node_attributes(prm.graph, 'pos'), with_labels=False, width=0.2, node_size=0.2)
    plt.savefig('prm.png')
    plt.show()
