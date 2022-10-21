from ast import Num
from lib2to3.pytree import Node
from tkinter import N
import numpy as np

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

if __name__ == "__main__":
    nodes, edges = createNodesAndEdges(4)
    print("Nodes list = ",nodes)
    print("edges and cost of travel dict = ",edges)
