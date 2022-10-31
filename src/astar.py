from curses import noecho
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import heapq
import time
import sys
import os
import math
import random
import graphlib
from PIL import Image
import os, sys
#from ..Utilities.utils import load_occupancy_map
import heapq
from collections import defaultdict
import scipy
import cv2

neighbor_map = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]

def load_occupancy_map():
	# Read image from disk using PIL
	occupancy_map_img = Image.open('/Users/anushsriramramesh/Library/CloudStorage/OneDrive-NortheasternUniversity/MR/MR-HW3/Utilities/occupancy_map.png')
	occupancy_grid = (np.asarray(occupancy_map_img) > 0).astype(int)
	return occupancy_grid

#Neighbour function : takes in occupancy grid and current vertex as input and return the list of neighbours of #the current vertex as a list of tuples. [(N1),(n2),(n3),...]
#graph : occupancy grid
#v : current vertex
def Neighbours(graph, v):
    neighbors = []
    for x,y in neighbor_map:
        search_point = (v[0]+x,v[1]+y)
        if graph[search_point] == 1:
            neighbors.append(search_point)
    return neighbors

#distance(v1,v2) : returns the distance between two vertices v1 and v2
#v1 : vertex 1
def distance(v1, v2):
    return math.dist(v1, v2)
class AStar:
    def __init__(self, graph, start, goal):
        self.graph = graph
        self.vertices = [(i,j) for i in range(self.graph.shape[0]) for j in range(self.graph.shape[1])]
        #print("vertices", self.vertices)
        self.start = start
        self.goal = goal
        print("start = ", self.start)
        print("goal = ", self.goal)
        self.path = []
        self.visited = []
        self.cost = 0
        self.Q = [(distance(start, goal),start)]
        heapq.heapify(self.Q)
        self.Qhelper = {start: distance(start, goal)}
        self.pred = {}
        self.costTo = defaultdict(lambda: float("inf"))
        self.estTotalCost = defaultdict(lambda: float("inf"))
        for v in self.vertices:
            self.costTo[v] = np.inf
            self.estTotalCost[v] = np.inf
        # self.costTo = {v: 9999999999999999 for v in self.vertices}
        # self.estTotalCost = {v: 9999999999999999 for v in self.vertices}
        self.costTo[start] = 0
        self.estTotalCost[start] = distance(start, goal)

    #Recover path from start to goal using predecessor dictionary
    #start : start point of the path
    #goal : goal point of the path
    #pred : predecessor dictionary returned from astar search function
    def RecoverPath(self, start, goal, pred):
        current = goal
        retracedPath = []
        retracedPath.append(goal)
        totalCost = 0
        #print("pred = ", pred)
        while current in pred:
            #print("current = ", current)
            prevPath = pred[current]
            retracedPath.append(prevPath)
            current = prevPath
            if current == start:
                break
        if current != start:
            print("no path found")

        for i in range(len(retracedPath) - 1):
            totalCost = totalCost + distance(retracedPath[i], retracedPath[i+1])
        return retracedPath, totalCost

#Astar search algorithm
#graph : occupancy grid
#start : start point of the path
#goal : goal point of the path
#Function is implemented as a member function of the class AStar
def a_star(self):
while self.Q != []:
    v = heapq.heappop(self.Q)[1]
    del self.Qhelper[v]
    if v == self.goal:
        print("goal found")
        return self.RecoverPath(self.start, self.goal, self.pred)
    #print("neighbors of ", v, " = ", Neighbours(self.graph, v))
    for n in Neighbours(self.graph,v):
        #print("n = ", n)
        pvi = self.costTo[v] + distance(v, n)
        if pvi < self.costTo[n]:
            #The path to i through v is better than the previously-known best path to i,
            # so record it as the new best path to i.
            self.pred[n] = v
            self.costTo[n] = pvi
            self.estTotalCost[n] = pvi + distance(n, self.goal)
            if n in self.Qhelper:
                tmpVal = self.Qhelper[n]
                self.Q.remove((tmpVal, n ))
                heapq.heappush(self.Q, (self.estTotalCost[n], n))
                self.Qhelper[n] = self.estTotalCost[n]
            else:
                heapq.heappush(self.Q, (self.estTotalCost[n], n))
                self.Qhelper[n] = self.estTotalCost[n]
return []

if __name__ == "__main__":
    #occupancyGrid = utils.convertToBinary()
    #print(load_occupancy_map())
    astar = AStar(load_occupancy_map(), (635,140),(350,400))
    #neighbors = Neighbours(occupancyGrid, (np.random.randint(2, len(occupancyGrid[0])-2), np.random.randint(2, len(occupancyGrid)-2)))
    #dis = distance((np.random.randint(2, len(occupancyGrid[0])-2), np.random.randint(2, len(occupancyGrid)-2)), (np.random.randint(2, len(occupancyGrid[0])-2), np.random.randint(2, len(occupancyGrid)-2)))
    finalPath, cost = astar.a_star()
    print(finalPath)
    print(cost)
    grid = load_occupancy_map()
    grid[np.where(grid>0)] = 255
    for x in finalPath:
        plt.plot(x[0], x[1], 'go', alpha=0.1)

    plt.plot(astar.start[0], astar.start[1], 'ro', alpha=0.8, label= 'start')
    plt.plot(astar.goal[0], astar.goal[1], 'bo', alpha=0.8, label= 'goal')

    plt.imshow(np.transpose(grid), cmap="inferno", origin='lower')
    print("img show")
    # img = cv2.imread('/Users/anushsriramramesh/Library/CloudStorage/OneDrive-NortheasternUniversity/MR/MR-HW3/Utilities/occupancy_map.png')
    # for x in finalPath:
    #     image = cv2.circle(img, x, radius=1, color=(0, 0, 255), thickness=1)
    # while(1):
    #     cv2.imshow('img',image)
    #     k = cv2.waitKey(0)
    #     if k==27:    # Esc key to stop
    #         break
    #     elif k==-1:  # normally -1 returned,so don't print it
    #         continue
    #     else:
    #         print(k)
        # else print its value
    # cv2.imshow('image', grid)
    # im = Image.fromarray(grid)
    # im.save("your_file.jpeg")
    # matplotlib.image.imsave('Astar_Path.png', grid)
    plt.legend(title = "Astar Path with length - {}".format(cost))
    plt.savefig('Astar_Path.png', dpi=500)
    #plt.show()
    # scipy.misc.imsave('outfile.jpg', grid)
    # print(neighbors)
    # print(dis)