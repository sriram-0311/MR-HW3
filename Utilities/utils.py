from ast import Num
from lib2to3.pytree import Node
from tkinter import N
import numpy as np
from PIL import Image
import numpy as np
import os, sys

def convertToBinary():
    #current directory printing
    print("Current Directory: ", os.getcwd())
    #read image
    occupancyGrid = np.array(Image.open('/Users/anushsriramramesh/Library/CloudStorage/OneDrive-NortheasternUniversity/MR/MR-HW3/Utilities/occupancy_map.png'))
    #convert to binary
    occupancyGrid = np.where(occupancyGrid > 0, 1, 0)

    return occupancyGrid



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
    # nodes, edges = createNodesAndEdges(4)
    # print("Nodes list = ",nodes)
    # print("edges and cost of travel dict = ",edges)
    occupancyGrid = convertToBinary()
    # for i in range(len(occupancyGrid)):
    #     for j in range(len(occupancyGrid[i])):
    #         if occupancyGrid[i][j] > 0:
    #             print("1", end = " ")
    print(occupancyGrid)

