# -*- coding: utf-8 -*-
"""
@author: LEFTERIS POLYCHRONAKIS
ID:      MATH6090
"""

import numpy as np
import warnings

warnings.filterwarnings("ignore")

np.random.seed(6090)

def pmethod(matrix, initial_guess, error, max_it, ratio = False):
    """
    
    Parameters
    ----------
    matrix : TYPE numpy.ndarray
        DESCRIPTION. A matrix in which we use the Power Method to compute the dominant eigenvalue
                     and the corresponding eigenvector
    initial_guess : TYPE numpy.ndarray
        DESCRIPTION. The initial guess for the eigenvector corresponding to the dominant eigenvalue
    error : TYPE int, float
        DESCRIPTION. When the value of the difference ||x_k - x_(k-1)|| < error (2-norm) the iteration stops.
                     x_k, x_(k-1) are two successive computations of the eigenvector
    max_it : TYPE int
        DESCRIPTION. If number of iterations exceed this limit and the method is not convergant for given error
                     (i.e ||x_k - x_(k-1)|| >= error) iteration stops and a None value is returned
    ratio : TYPE, optional
        DESCRIPTION. The default is False. When set to True prints iteration number and the ratio:
                     |lambda_2 / lambda_1| / ( d_k/d_(k-1) ) where lambda_1, lambda_2 denotes the first and second largest 
                     eigenvalues and d_k denotes the 2-norm of the differnce x_k - x_(k-1)

    Returns
    -------
    RayleighQ : TYPE
        DESCRIPTION. The Rayliegh quotient of the last value of the eigenvector.
                     This quotient is the dominant eigenvalue.
    xnew : TYPE numpy.ndarray
        DESCRIPTION. The eigenvector corresponding to the dominant eigenvalue
    it_ctr : TYPE int
        DESCRIPTION. Number of iterations executed until given tolerance is reached.

    """
    
    xold = np.ones((len(matrix[0]),1))
    xnew = np.ones((len(matrix[0]),1))
    xold[:] = initial_guess
    condition = True
    it_ctr = 0
    dold = 0
    
    
    if ratio == True:
        eigs = sorted(np.linalg.eigvals(matrix), reverse = True)
        rl = abs(eigs[1] / eigs[0])
       
    while condition:
        
        xnew[:] = matrix @ xold
        xnew[:] = xnew/ np.linalg.norm(xnew) # 2-norm
        dnew = np.linalg.norm(xnew - xold)
        it_ctr += 1
        
        if ratio == True:
            print('Iteration : {0:3d}, Ratio = {1:0.4f}'.format(it_ctr, (dnew/dold) / rl))
                        
        if it_ctr > max_it:
            print('Power Method not convergent for {0:d} iterations.'.format(max_it))
            return None
            
        elif error > dnew :
            condition = False
    
        np.copyto(xold,xnew); dold = dnew
        
    RayleighQ = ( (xnew.T @ matrix @ xnew) / np.dot(xnew.T, xnew) )[0][0]
    
    return RayleighQ , xnew, it_ctr


#%%

def rescale(A):
    """
    
    Parameters
    ----------
    A : TYPE numpy.ndarray
        DESCRIPTION. A numpy vector

    Returns
    -------
    out : TYPE numpy.ndarray
        DESCRIPTION. Rescaled vector in the interval [0, 1]

    """
    
    out = np.zeros(len(A))
    Sum = sum(A)
    
    for i in range(len(A)):
        out[i] = A[i] / Sum 
        
    return out


# %%

class Graph():

    def __init__(self, V=[], directed = True):
        self.vertices = {}
        for x in V:
            self.vertices[x] = []
        self.directed = directed

    def __str__(self):
        s = 'Vertices: {}\nEdges:\n'.format(list(self.vertices.keys()))
        for v in self.vertices:
            for x in self.vertices[v]:
                s += '({}->{}, weight:{})\n'.format(v, x[0], x[1])
            if (len(self.vertices[v]) > 0):
                s += '\n'
        return s

    def Vertices(self):
        """
        Returns
        -------
        TYPE dict_keys
            DESCRIPTION. Vertice names of the Graph

        """
        return self.vertices.keys()

    def Edges(self):
        """
        
        Returns
        -------
        TYPE list of tupples
            DESCRIPTION.  List of edges of the graph in the form (origin, destination)

        """
        edges = []
        for v in self.vertices:
            for x in self.vertices[v]:
                edges.append((v, x[0], x[1]))
        return edges

    def IncidentEdges(self, v):
        """
        
        Parameters
        ----------
        v : TYPE
            DESCRIPTION. Name of the vertex

        Returns
        -------
        TYPE list
            DESCRIPTION. Name of the vertices incident to the input vertex

        """
        return self.vertices[v]

    def AddVertex(self, v):
        """

        Parameters
        ----------
        v : TYPE
            DESCRIPTION. Adds vertex V in the Graph if not already there
            
        """
        if v not in self.vertices:
            self.vertices[v] = []
        return self

    def AddEdge(self, x, y, weight = 1):
        """
        
        Parameters
        ----------
        x : TYPE
            DESCRIPTION. Origin vertex of an edge
        y : TYPE
            DESCRIPTION. Destination vertex of an edge
        weight : TYPE float, optional
            DESCRIPTION. The default is 1. Adds the weight of the edge

        """
        self.vertices[x].append((y, weight))
        if not self.directed:
            self.vertices[y].append((x, weight))
        return self
    
    def AdjacencyΜatrix(self):
        """
        
        Returns
        -------
        M : TYPE numpy.ndarray
            DESCRIPTION. The Adjacency Matrix of the Graph.

        """
        import numpy as np
        
        M = np.zeros((len(self.Vertices()),len(self.Vertices())))
        
        #make Adjacency Μatrix
        for v in self.vertices:
            for t in self.IncidentEdges(v):
                M[t[0] - 1, v - 1] = 1

        return M