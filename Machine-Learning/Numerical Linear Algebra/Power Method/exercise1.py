# -*- coding: utf-8 -*-
"""
@author:  lpoly
"""

# Do not change or move files:
# 'R.csv','F.csv','W.csv' data will be read from them

import numpy as np
import power_inverse_power__methods as tools

#Array input 
R = np.loadtxt("./matrixR.csv", delimiter=",", comments="#")
F = np.loadtxt("./matrixF.csv", delimiter=",", comments="#")
W = np.loadtxt("./matrixW.csv", delimiter=",", comments="#")

   
# Question A

   
def main(R,F,W):
    
    s = "RFW"
    Matrices = [R,F,W]
    dom_eigenvalues = []
    iterations = []
    matrixNP_eigs = []
    
    #collect data
    for matrix in Matrices:
        
        #QUESTION A AND B
        ans = tools.Power_Method(matrix,np.ones(len(matrix[0])),1e-3,1000)
        dom_eigenvalues.append(ans[0])
        iterations.append(ans[1])
        matrixNP_eigs.append(max(np.linalg.eigvals(matrix)))
        
    #QUESTION C
    i=0
    for matrix in Matrices:
        ans = tools.Inverse_Power_Method(matrix,np.ones(len(matrix[0])),
                                         1e-6,1000, 0.9*matrixNP_eigs[i])
        dom_eigenvalues.append(ans[0])
        iterations.append(ans[1])
        i += 1
        
        
        
        
    print(" "+26*"="+"QUESTION A"+27*"=")
    print(" |%-15s |%-20s |%-10s |%-10s|"
          % ("MATRIX","EIGENVALUE","ERROR","ITERATIONS"))
    
    for n in range(3):
        print(" |%-15s |%-20.5f |%-10s |%-10s|" % (s[n],dom_eigenvalues[n],
                                                   1e-3,iterations[n]))
        
    print(" "+63*"="+3*"\n"+" "+26*"="+"QUESTION B"+27*"=")
    print(" |%-20s |%-20s |%-10s|" % ("POWER METHOD ESTIMATION",
                                      "NUMPY ESTIMATION","ABSOLUTE DIFF."))
    
    for n in range(3):
        print(" |%-23.6f |%-20.6f |%-10.6f    |" % (dom_eigenvalues[n],
                                                    matrixNP_eigs[n],
                                                    abs(dom_eigenvalues[n]
                                                        -matrixNP_eigs[n])))
        
    print(" "+63*"="+3*"\n"+" "+26*"="+"QUESTION C"+27*"=")
    print(" |%-15s |%-20s |%-10s |%-10s|" % ("MATRIX","EIGENVALUE",
                                             "ERROR","ITERATIONS"))
    for n in range(3):
        print(" |%-15s |%-20.8f |%-10s |%-10s|" % (s[n],dom_eigenvalues[n+3],
                                                   1e-6,iterations[n+3]))

    print(" "+63*"=")
    return None 

main(R,F,W)