import math
from PIL import Image
import numpy as np
import heapq
from matplotlib import pyplot as plt
import matplotlib
# matplotlib.use('xcb') # no UI backend
INF = 1000000000

neighbor_map = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]

def distance(p1,p2):
	return math.dist(p2,p1)

def h(p1,p2):
	return distance(p1,p2)

def w(p1,p2):
	return distance(p1,p2)

def N(graph,v):
	neighbors = []
	for x,y in neighbor_map:
		search_point = (v[0]+x,v[1]+y)
		if graph[search_point] == 1:
			neighbors.append(search_point)
	return neighbors

def load_occupancy_map():
	# Read image from disk using PIL
	occupancy_map_img = Image.open('/Users/anushsriramramesh/Library/CloudStorage/OneDrive-NortheasternUniversity/MR/MR-HW3/Utilities/occupancy_map.png')
	occupancy_grid = (np.asarray(occupancy_map_img) > 0).astype(int)
	return occupancy_grid

class A_star_search_driver():
	def __init__(self,vertices,start,goal,N,w,h):
		self.occupancy_grid = load_occupancy_map()
		self.vertices = [(i,j) for i in range(self.occupancy_grid.shape[0]) for j in range(self.occupancy_grid.shape[1])]
		self.start = start
		self.goal = goal
		self.N = N
		self.w = w
		self.h = h
		self.pred = {}
		self.CostTo = {v:INF for v in self.vertices}
		self.EstCostTo = {v:INF for v in self.vertices}
		self.CostTo[start] = 0
		self.EstCostTo[start] = self.h(start,goal)
		self.Q = [(self.h(start,goal),start)]
		heapq.heapify(self.Q)
		self.Q_tracker = {start:self.h(start,goal)}

	def RecoverPath(self,s,g,pred):
		curr_vertex = g
		optimal_path = []
		while curr_vertex in pred:
			predecessor = pred[curr_vertex]
			optimal_path.append(predecessor)
			curr_vertex = predecessor
			if curr_vertex == s:
				break
		if curr_vertex != s:
			print('path not found')
		return optimal_path

	def AStarSearch(self):
		while self.Q!=[]:
			v = heapq.heappop(self.Q)[1]
			del self.Q_tracker[v]
			if v == self.goal:
				return self.RecoverPath(self.start,self.goal,self.pred)
			for nbs in self.N(self.occupancy_grid,v):
				pvi = self.CostTo[v] + self.w(v,nbs)
				if pvi < self.CostTo[nbs]:
					self.pred[nbs]=v
					self.CostTo[nbs]=pvi
					self.EstCostTo[nbs]=pvi+self.h(nbs,self.goal)
					if nbs in self.Q_tracker:
						val = self.Q_tracker[nbs]
						self.Q.remove((val,nbs))
						heapq.heappush(self.Q,(self.EstCostTo[nbs],nbs))
						self.Q_tracker[nbs]=self.EstCostTo[nbs]
					else:
						heapq.heappush(self.Q,(self.EstCostTo[nbs],nbs))
						self.Q_tracker[nbs]=self.EstCostTo[nbs]
		return []

searcher = A_star_search_driver([],(635,140),(350,400),N,w,h)
path = searcher.AStarSearch()

occ_map = load_occupancy_map()
occ_map[np.where(occ_map>0)]=255
for v in path:
	occ_map[v[0]][v[1]]=127

print(occ_map)
matplotlib.image.imsave('name.png', occ_map)
