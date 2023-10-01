
# -*- coding: utf-8 -*-
"""
@author: LEFTERIS POLYCHRONAKIS
ID:      MATH6090
"""
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

files = ["./uoc_logo.png", "./python_logo.png"]

no = 0

for image in files:

    matrix = plt.imread(image)

    # Split image into color channels
    rC = matrix[:,:,0]
    gC = matrix[:,:,1]
    bC = matrix[:,:,2]
    aC = matrix[:,:,3]
    
    rU, rS, rVh = np.linalg.svd(rC)
    bU, bS, bVh = np.linalg.svd(bC)
    gU, gS, gVh = np.linalg.svd(gC)
    aU, aS, aVh = np.linalg.svd(aC)
    
    # RED
    plt.figure(1)
    plt.axvline(x=10, color = "gold", label = "k = 10", linestyle = "--", zorder =2)
    plt.axvline(x=55, color = "gray",label = "k = 55",linestyle = "--", zorder = 3)
    plt.axvline(x=100, color = "black", label = "k = 100",linestyle = "--", zorder = 4)
    plt.plot(range(len(rS)), rS, color = "maroon", linewidth = 3, zorder = 1)
    plt.title('Red Channel Singular Values ({} Logo)'
              .format('UoC' if no == 0 else 'Python'))
    plt.yscale("log")
    plt.xlabel("k")
    plt.ylabel("singular value")
    plt.legend()
    plt.show()
    
    plt.figure(2)
    plt.plot(np.cumsum(rS) / np.sum(rS),linewidth = 3, color = "maroon")
    plt.axvline(x= 10, color = "gold", label = "k = 10", linestyle = "--", zorder =2)
    plt.axvline(x= 55, color = "gray",label = "k = 55",linestyle = "--", zorder = 3)
    plt.axvline(x= 100, color = "black", label = "k = 100",linestyle = "--", zorder = 4)
    plt.title('Red Channel Singular Values: Cumulative Sum ({} Logo)'
              .format('UoC' if no == 0 else 'Python'))
    plt.xlabel("k")
    plt.ylabel("% of information captured")
    plt.legend(loc = 'lower right')
    plt.show()
    
    # GREEN
    plt.figure(3)
    plt.axvline(x=10, color = "gold", label = "k = 10", linestyle = "--", zorder =2)
    plt.axvline(x=55, color = "gray",label = "k = 55",linestyle = "--", zorder = 3)
    plt.axvline(x=100, color = "black", label = "k = 100",linestyle = "--", zorder = 4)
    plt.plot(range(len(gS)), gS, color = "darkgreen",linewidth = 3, zorder = 1)
    plt.title('Green Channel Singular Values ({} Logo)'
              .format('UoC' if no == 0 else 'Python'))
    plt.yscale("log")
    plt.xlabel("k")
    plt.ylabel("singular value")
    plt.legend()
    plt.show()
    
    plt.figure(4)
    plt.plot(np.cumsum(gS) / np.sum(gS), linewidth = 3, color = "darkgreen")
    plt.axvline(x= 10, color = "gold", label = "k = 10", linestyle = "--", zorder =2)
    plt.axvline(x= 55, color = "gray",label = "k = 55",linestyle = "--", zorder = 3)
    plt.axvline(x= 100, color = "black", label = "k = 100",linestyle = "--", zorder = 4)
    plt.title('Green Channel Singular Values: Cumulative Sum ({} Logo)'
              .format('UoC' if no == 0 else 'Python'))
    plt.xlabel("k")
    plt.ylabel("% of information captured")
    plt.legend(loc = 'lower right')
    plt.show()
    
    # BLUE
    plt.figure(5)
    plt.axvline(x=10, color = "gold", label = "k = 10", linestyle = "--", zorder =2)
    plt.axvline(x=55, color = "gray",label = "k = 55",linestyle = "--", zorder = 3)
    plt.axvline(x=100, color = "black", label = "k = 100",linestyle = "--", zorder = 4)
    plt.plot(range(len(bS)), bS, color = "royalblue", linewidth = 3, zorder = 1)
    plt.title('Blue Channel Singular Values ({} Logo)'
              .format('UoC' if no == 0 else 'Python'))
    plt.yscale("log")
    plt.xlabel("k")
    plt.ylabel("singular value")
    plt.legend()
    plt.show()
    
    plt.figure(6)
    plt.plot(np.cumsum(bS) / np.sum(bS), linewidth = 3, color = "royalblue")
    plt.axvline(x= 10, color = "gold", label = "k = 10", linestyle = "--", zorder =2)
    plt.axvline(x= 55, color = "gray",label = "k = 55",linestyle = "--", zorder = 3)
    plt.axvline(x= 100, color = "black", label = "k = 100",linestyle = "--", zorder = 4)
    plt.title('Blue Channel Singular Values: Cumulative Sum ({} Logo)'
              .format('UoC' if no == 0 else 'Python'))
    plt.xlabel("k")
    plt.ylabel("% of information captured")
    plt.legend(loc = 'lower right')
    plt.show()
    
    no = 1