from cProfile import label
from turtle import width
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
import math
from pyparsing import alphas
import scipy

neighbor_map = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]

def SaveFigs(path, graph, totalCost, start, goal):
    occupancygrid = load_occupancy_map()
    occupancygrid[np.where(occupancygrid>0)] = 255
    plt.imshow(np.transpose(occupancygrid), cmap="inferno", origin='lower')
    nx.draw_networkx_nodes(graph, pos=nx.get_node_attributes(graph, 'pos'),node_size=0.2)
    plt.legend(title='PRM Graph - 2500 nodes alone', loc= 'upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
    plt.savefig('PRMGraph_Nodes.png', dpi=500)
    plt.close()
    plt.imshow(np.transpose(occupancygrid), cmap="inferno", origin='lower')
    nx.draw_networkx_nodes(graph, pos=nx.get_node_attributes(graph, 'pos'),node_size=0.2)
    nx.draw_networkx_edges(graph, pos=nx.get_node_attributes(graph, 'pos'),width=0.01)
    plt.legend(title='PRM Graph - 2500 nodes and edges', loc= 'upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
    plt.savefig('PRMGraph_NodesEdges.png', dpi=500)
    plt.close()
    for i in range(len(path)):
        #plt.plot(prm.graph.nodes[path[i]]['pos'][0], prm.graph.nodes[path[i]]['pos'][1], 'ro',alpha=1)
        if i < len(path)-1:
            #plt.plot([prm.graph.nodes[path[i]]['pos'][0],prm.graph.nodes[path[i+1]]['pos'][0]], [prm.graph.nodes[path[i]]['pos'][1],prm.graph.nodes[path[i+1]]['pos'][1]], 'r',alpha=1)
            nx.draw_networkx_edges(graph, pos=nx.get_node_attributes(graph, 'pos'), edgelist=[(path[i], path[i+1])], width=2, alpha=0.5, edge_color='g')
    plt.plot(start[0], start[1], 'go',alpha=1, label='Start')
    plt.plot(goal[0], goal[1], 'ro',alpha=1, label='Goal')
    nx.draw_networkx_nodes(graph, pos=nx.get_node_attributes(graph, 'pos'),node_size=0.1)
    plt.imshow(np.transpose(occupancygrid), cmap="inferno", origin='lower')
    plt.legend(title='PRM Graph with Astar path with total length - {}'.format(totalCost), loc= 'upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
    plt.savefig('PRMGraph_AstarPath.png', dpi=500)

def load_occupancy_map():
	# Read image from disk using PIL
	occupancy_map_img = Image.open('/Users/anushsriramramesh/Library/CloudStorage/OneDrive-NortheasternUniversity/MR/MR-HW3/Utilities/occupancy_map.png')
	occupancy_grid = (np.asarray(occupancy_map_img) > 0).astype(int)
	return occupancy_grid

#rejectionSamples(occupancyGrid) : returns a uniformly randomly sampled point from the free space in the occupancy grid
#occupancyGrid : 2D numpy array of the occupancy grid
#returns : a tuple of the form (x,y) where x and y are the coordinates of the sampled point
def rejectionSampler(occupancyGrid):
    while True:
        x = int(np.random.uniform(0, occupancyGrid.shape[0]))
        y = int(np.random.uniform(0, occupancyGrid.shape[1]))
        if occupancyGrid[x][y] > 0:
            return (x,y)
        else:
            continue

def get_line(x1, y1, x2, y2):
        points = []
        issteep = abs(y2-y1) > abs(x2-x1)
        if issteep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        rev = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            rev = True
        deltax = x2 - x1
        deltay = abs(y2-y1)
        error = int(deltax / 2)
        y = y1
        ystep = None
        if y1 < y2:
            ystep = 1
        else:
            ystep = -1
        for x in range(x1, x2 + 1):
            if issteep:
                points.append((y, x))
            else:
                points.append((x, y))
            error -= deltay
            if error < 0:
                y += ystep
                error += deltax
        if rev:
            points.reverse()
        return points

def Neighbours(graph, v):
    neighbors = []
    for x,y in neighbor_map:
        search_point = (v[0]+x,v[1]+y)
        neighbors.append(search_point)
    return neighbors

#reachabilityCheck(occupancyGrid, point1, point2) : checks if there is a line of sight between point1 and point2
#occupancyGrid : 2D numpy array of the occupancy grid
#point1 : tuple of the form (x,y) where x and y are the coordinates of the first point
#point2 : tuple of the form (x,y) where x and y are the coordinates of the second point
#returns : True if there is a line of sight between point1 and point2, False otherwise
def reachabilityCheck(occupancyGrid, v1, v2):
    # print("v1: ", v1)
    # print("v2: ", v2)
    current = v1
    while current != v2:
        neighbors = Neighbours(occupancyGrid, current)
        current = min(neighbors, key=lambda x: math.dist(x, v2))
        if occupancyGrid[current] == 0:
            return False
        else:
            continue
    return True

def bressenhamReachabilityCheck(v1, v2, occupancyGrid):
    points = []
    m_new = 2 * (v2[1] - v1[1])
    slope_error_new = m_new - (v2[1] - v1[1])
    y = v1[1]
    for x in range(v1[0], v2[0]+1):
        points.append((x,y))
        #print("(", x, ",", y, ")\n")
        # Add slope to increment angle formed
        slope_error_new = slope_error_new + m_new
        # Slope error reached limit, time to
        # increment y and update slope error.
        if (slope_error_new >= 0):
            y = y+1
            slope_error_new = slope_error_new - 2 * (v2[0] - v1[0])

    for indx in points :
        for x,y in neighbor_map :
            search_point = (indx[0]+x,indx[1]+y)
            if occupancyGrid[search_point] == 0:
                return False
    return True

def reachablity_check(occupancygrid,v1, v2):

    points = get_line(v1[0],v1[1],v2[0],v2[1])

    for indx in points :
        for x,y in neighbor_map :
            search_point = (indx[0]+x,indx[1]+y)
            if occupancygrid[search_point] == 0:
                return False
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

    #build_graph() : builds the PRM graph using the rejection sampling algorithm for the given number of samples and maximum distance
    #number_of_samples : number of samples to be generated
    #max_distance : maximum distance between two nodes in the graph
    #returns : None
    #updates the graph, vertices and edges attributes of the class PRM
    def build_graph(self, n):
        for i in range(n):
            sampleVertex = rejectionSampler(self.occupancy_grid)
            self.addVertex(sampleVertex, i)

    #addVertex(vertex, index) : adds a vertex to the graph
    #vertex : tuple of the form (x,y) where x and y are the coordinates of the vertex to be added sampled from the free space
    #itr : index of the vertex to be added
    #returns : None
    def addVertex(self, sampleVertex, itr):
        self.graph.add_node(itr, pos=sampleVertex)
        if itr > 1:
            for v in self.graph.nodes:
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
            totalCost = nx.astar_path_length(self.graph, startind, stopindex)
            return path, totalCost
        else:
            return None

if __name__ == "__main__":
    prm = PRM(2500, 75)
    prm.build_graph(2500)
    start = (635,140)
    goal = (350,400)
    path, totalCost = prm.aStartSearch(start, goal)
    print("PATHHHHH",path)
    poses = []
    for i in prm.graph.nodes.data('pos'):
        poses.append(i[1])
    if start in poses:
        print("True")
    else:
        print("False")
    SaveFigs(path,prm.graph, totalCost, start, goal)
    point1 = (120,240)
    point2 = (240,500)
    plt.show()