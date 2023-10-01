# -*- coding: utf-8 -*-
"""
@author: lpoly
"""
import numpy as np
import matplotlib.pyplot as plt

# FILES uoc_logo.png, python_logo.png MUST BE IN THE SAME DIRECTORY 

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [16, 8]

files = ["UoC","Python"]

for image in files:
    
    matrix = plt.imread("./" + image + "_logo.png")
    
    # plot original
    plt.imshow(matrix)
    plt.grid(False)
    plt.axis('off')
    plt.title('Original')
    plt.show()
    
    print(75*"-")
    print("File: {0}\n".format(image + "_logo.png"))
    
    xlist = []; ylist = []
    
    # Split image into color channels
    rC = matrix[:,:,0]
    gC = matrix[:,:,1]
    bC = matrix[:,:,2]
    aC = matrix[:,:,3]
    
    rU, rS, rVh = np.linalg.svd(rC)
    bU, bS, bVh = np.linalg.svd(bC)
    gU, gS, gVh = np.linalg.svd(gC)
    aU, aS, aVh = np.linalg.svd(aC)
    
    min_dim = min(np.shape(matrix)[0],np.shape(matrix)[1])
    ciel = min_dim - int(str(min_dim)[-1])

    # reconstruct each color channel for each value of singular values 
    # and combine the channels to form the reconstructed image
    for n in range(1,ciel,9):

        rC_recon = np.clip(rU[:, :n] @ np.diag(rS[:n]) @ rVh[:n, :], 0, 1)
        gC_recon = np.clip(gU[:, :n] @ np.diag(gS[:n]) @ gVh[:n, :], 0, 1)
        bC_recon = np.clip(bU[:, :n] @ np.diag(bS[:n]) @ bVh[:n, :], 0, 1)
        aC_recon = np.clip(aU[:, :n] @ np.diag(aS[:n]) @ aVh[:n, :], 0, 1)
        
        matrix_recon = np.stack([rC_recon, gC_recon, bC_recon, aC_recon], 
                                axis=2) 
        
        # save data to plot later
        error = np.linalg.norm(matrix - matrix_recon)
        xlist.append(n);ylist.append(error)
        
        if n in [10, 55, 100]:
                        
            plt.imshow(matrix_recon)
            plt.title("k = {0:d}".format(n))
            plt.grid(False)
            plt.axis('off')
            plt.show()
            
        if (n-2) % 10 == 0 or n % 4 == 0: #print some of the errors during iter
            print("For k = {0:3d}   Error = {1:0.4f}".format(n,error))
    
    # Error plot format
    plt.title("{0} Logo error".format(image))
    plt.plot(xlist, ylist,color = "maroon", label = "sequence of error")  
    plt.xlabel("k")
    plt.ylabel("error")
    plt.legend()
    plt.show()
    
    print(75*"-" + "\n")