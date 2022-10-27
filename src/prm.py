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
        print("start", start)
        print("goal", goal)
        path = []
        poses = []
        for i in prm.graph.nodes.data('pos'):
            poses.append(i[1])
        if (start != goal):
            if start not in poses:
                print("Start or Goal not in graph")
                startind = self.graph.number_of_nodes() +1
                self.addVertex(start, self.graph.number_of_nodes() +1)
                print("added start")
            if goal not in poses:
                stopindex = self.graph.number_of_nodes() +1
                self.addVertex(goal, self.graph.number_of_nodes() +1)
                print("added goal")
            path = nx.astar_path(self.graph, startind, stopindex)
            return path
        else:
            return None

if __name__ == "__main__":
    prm = PRM(5000, 75)
    prm.build_graph(5000)
    start = (635,140)
    goal = (350,400)
    path = prm.aStartSearch(start, goal)
    # print("graph", prm.graph.nodes)
    # print("start in graph", prm.graph.nodes.data('pos'))
    # print("start in graph", prm.graph.nodes[2002]['pos'])
    print("PATHHHHH",path)
    poses = []
    for i in prm.graph.nodes.data('pos'):
        poses.append(i[1])
    if start in poses:
        print("True")
    else:
        print("False")
    grid = load_occupancy_map()
    grid[np.where(grid>0)] = 255
    for i in path:
        grid[prm.graph.nodes[i]['pos'][0]][prm.graph.nodes[i]['pos'][1]] = 127
    plt.imshow(np.transpose(grid), cmap="inferno", origin='lower')
    # print("graph", prm.graph.number_of_nodes())
    # print("graph", prm.graph.number_of_edges())
    # for v in prm.graph.nodes:
    #     plt.plot(prm.graph.nodes[v]['pos'][1], prm.graph.nodes[v]['pos'][0], 'r',alpha=0.2)
    for i in range(len(path)):
        #plt.plot(prm.graph.nodes[path[i]]['pos'][0], prm.graph.nodes[path[i]]['pos'][1], 'ro',alpha=1)
        if i < len(path)-1:
            #plt.plot([prm.graph.nodes[path[i]]['pos'][0],prm.graph.nodes[path[i+1]]['pos'][0]], [prm.graph.nodes[path[i]]['pos'][1],prm.graph.nodes[path[i+1]]['pos'][1]], 'r',alpha=1)
            nx.draw_networkx_edges(prm.graph, pos=nx.get_node_attributes(prm.graph, 'pos'), edgelist=[(path[i], path[i+1])], width=2, alpha=0.5, edge_color='g')
    nx.draw_networkx_nodes(prm.graph, pos=nx.get_node_attributes(prm.graph, 'pos'),node_size=0.2)
    plt.savefig('prm.png')
    plt.show()
