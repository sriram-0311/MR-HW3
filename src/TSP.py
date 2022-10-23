#Dynamic programming algorithm for TSP  (Traveling Salesman Problem)
#Input:  a distance matrix
#Output:  the optimal tour and its cost
#      the optimal subtours

import numpy as np
import graphviz
import matplotlib.pyplot as plt
import networkx as nx
import random
import math
def permutations(vertex):
    if len(vertex) == 0:
        return []
    if len(vertex) == 1:
        return [vertex]
    l = []
    for i in range(len(vertex)):
        m = vertex[i]
        rem = vertex[:i] + vertex[i + 1:]
        for p in permutations(rem):
            l.append([m] + p)
    return l

def TravellingSalesmanProblem(graph, s):
    # store all vertex apart from source vertex
    vertex = []
    for i in range(len(graph)):
        if i != s:
            vertex.append(i)
            print(vertex)
    # store minimum weight Hamiltonian Cycle
    min_path = math.inf
    next_permutation=permutations(vertex)
    for i in next_permutation:
        # store current Path weight(cost)
        current_pathweight = 0
        # compute current path weight
        k = s
        for j in i:
            current_pathweight += graph[k][j]
            k = j
        current_pathweight += graph[k][s]
        # update minimum
        min_path = min(min_path, current_pathweight)
    return min_path

def dpTSPHelper(graph, s, visited, memo):
    if visited == (1 << (n+1)) - 1:
        return graph[s][0]
    if memo[s][visited] != -1:
        return memo[s][visited]
    ans = math.inf
    for city in range(n):
        if visited & (1 << city) == 0:
            newAns = graph[s][city] + dpTSPHelper(graph, city, visited | (1 << city), memo)
            ans = min(ans, newAns)
    memo[s][visited] = ans
    return ans

def TSP(graph, s):
    n = len(graph)
    visited = []
    queue = []

    visited.append(s)
    queue.append([1,2,3])
    for i in range(n):
        for j in queue:
            

if __name__ == "__main__":
    graph = [[0, 2, 1, 1], [2, 0, 1, 1], [1, 2, 0, 1], [1, 1, 2, 0]]
    s = 0
    #print(TravellingSalesmanProblem(graph, s))
    n = len(graph)
    memo = [[-1 for i in range(1 << n)] for j in range(n)]
    print(dpTSPHelper(graph, s, 1, memo))

