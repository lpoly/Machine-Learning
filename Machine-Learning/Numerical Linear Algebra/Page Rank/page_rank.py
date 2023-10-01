# -*- coding: utf-8 -*-
"""
@author: lpoly
"""

# FILES graph0.txt,graph1.txt,graph2.txt MUST BE IN THE SAME DIRECTORY  

import numpy as np
from tools import Graph, pmethod, rescale

# set numpy print options
np.set_printoptions(precision = 4, linewidth = 75, suppress = True)

# %% 
# collect data 

data0 =np.loadtxt("./graph0.txt")
data1 =np.loadtxt("./graph1.txt")
data2 =np.loadtxt("./graph2.txt")

Data = [data0, data1, data2]


# %%
# iterate thru data

for i in range(3):
    G = Graph()
    # Fill Graph with data
    for j in range(len(Data[i])):
        
        G.AddVertex(int(Data[i][j][0]))
        G.AddVertex(int(Data[i][j][1]))
        G.AddEdge(int(Data[i][j][0]), int(Data[i][j][1]))
    
    A = G.AdjacencyΜatrix()                
    # Normalize columns
    for col in range(len(A[0])):
        A[:,col] = A[:,col]/sum(A[:,col]) 
    
    # Compute M            
    N = len(G.Vertices())
    d = 0.85 # damping factor
    M = d * A + ((1-d)/N) * np.ones( (N,N) )
    
    # Compute dominant eigenvalue and corresponding eigenvector
    p = pmethod(M, 1/N * np.ones((N,1)), 1e-6, 1000, ratio = False)
    dom_eigenval = p[0]; cor_eigenvec = p[1];
    
    # rescale eigenvector
    rescaled = rescale(cor_eigenvec)
    
    # Output
    print(75*"-" + "\n")
    print(34*" " + "GRAPH {0:d}".format(i) + 34*" ")
    print("Dominant eigenvalue using Power Method: {0:0.6f}\n"
          .format(dom_eigenval))
    print("Corresponding eigenvector :\n {0}\n".format(cor_eigenvec.flatten()))
    print("Eigenvector sums to : {0:0.3f}\n\n"
          .format(float(sum(cor_eigenvec))))
    print("Rescaled Eigenvector :\n {0}\n".format(rescaled))
    print("Rescaled Eigenvecto sums to : {0:0.4f}\n\n\n".format(sum(rescaled)))
    
    # Sort vertices in relation to their score
    verticesImportance = dict([(x,rescaled[x-1]) for x in G.Vertices()])
    DecendingScore = sorted(verticesImportance.items(),
                            key = lambda item: item[1], reverse = True)
    
    for e in DecendingScore:
        print(21*" " + "Vertex: {0:3d}    |    Score: {1:0.4f}"
              .format(e[0],e[1]) + 21*" ")
    
    print("\n" + 75*"-" + "\n")