import numpy as np
import graphviz
import matplotlib.pyplot as plt
import networkx as nx
import random
import math

def createNodesAndEdges(N):
    NodesList = []
    EdgeDict = dict()
    for i in range(int(N)):
        NodesList.append(chr(i + 65))

    for i in NodesList:
        #print("debug:",tempList)
        for j in NodesList[NodesList.index(i):]:
            if (i+j == "AB"):
                EdgeDict[i+j] = 2
            elif (i+j == "AC"):
                EdgeDict[i+j] = 1
            elif (i+j == "AD"):
                EdgeDict[i+j] = 1
            elif (i+j == "BC"):
                EdgeDict[i+j] = "x"
            elif (i+j == "BD"):
                EdgeDict[i+j] = 2
            elif (i+j == "CD"):
                EdgeDict[i+j] = 1
    return NodesList, EdgeDict

def createGraph(NodesList, EdgeDict):
    G = nx.Graph()
    G.add_nodes_from(NodesList)
    for i in EdgeDict:
        G.add_edge(i[0],i[1],weight=EdgeDict[i])
    return G

def drawGraph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

if __name__ == "__main__":
    nodes, edges = createNodesAndEdges(4)
    print("Nodes list = ",nodes)
    print("edges and cost of travel dict = ",edges)
    G = createGraph(nodes, edges)
    drawGraph(G)

