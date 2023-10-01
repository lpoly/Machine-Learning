import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

files = ["./python_logo.png"]
# files = ["./uoc_logo.png", "./python_logo.png"]
plt.figure(figsize = (16,9))


no = 1

i = 3

uoriginal = plt.imread('uoc_logo.png')
poriginal = plt.imread('python_logo.png')

u10 = plt.imread('uoc10.png')
u55 = plt.imread('uoc55.png')
u100 = plt.imread('uoc100.png')


p5 = plt.imread('py10.png')
p55 = plt.imread('py55.png')
p100 = plt.imread('py100.png')

plt.subplot(3,3,1)
plt.axis('off')
plt.imshow(poriginal)


plt.subplot(3,3,2)
plt.title("Original")
plt.axis('off')
plt.imshow(poriginal)


plt.subplot(3,3,3)
plt.axis('off')
plt.imshow(poriginal)

# plt.subplot(3,3,4)
# plt.axis('off')
# plt.imshow(u10)


# plt.subplot(3,3,5)
# plt.axis('off')
# plt.imshow(u55)


# plt.subplot(3,3,6)
# plt.axis('off')
# plt.imshow(u100)


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
        if n in [10, 55, 100]:
                        
            plt.subplot(3,3,i+1)
            plt.xticks(color='white')

            plt.axis('off')
            plt.title("components = {0:d}".format(n))
            plt.grid(False)
            plt.imshow(matrix_recon)
            i+=1
    # RED
 
    # plt.figure(2)
    plt.subplot(3,3,i+1)
    plt.plot(np.cumsum(rS) / np.sum(rS),linewidth = 3, color = "maroon")
    plt.axvline(x= 10, color = "gold", label = "comps = 10", linestyle = "--", zorder =2)
    plt.axvline(x= 55, color = "gray",label = "comps = 55",linestyle = "--", zorder = 3)
    plt.axvline(x= 100, color = "black", label = "comps = 100",linestyle = "--", zorder = 4)
    plt.title('Red Ch: CumSum ({})'
              .format('UoC' if no == 0 else 'Python'))
    plt.xlabel("# of components")
    plt.ylabel("% of information captured")
    plt.legend(loc = 'lower right')
    
    # GREEN    
    # plt.figure(4)
    i+=1
    plt.subplot(3,3,i+1)

    plt.plot(np.cumsum(gS) / np.sum(gS), linewidth = 3, color = "darkgreen")
    plt.axvline(x= 10, color = "gold", label = "comps = 10", linestyle = "--", zorder =2)
    plt.axvline(x= 55, color = "gray",label = "comps = 55",linestyle = "--", zorder = 3)
    plt.axvline(x= 100, color = "black", label = "comps = 100",linestyle = "--", zorder = 4)
    plt.title('Green Ch: CumSum ({})'
              .format('UoC' if no == 0 else 'Python'))
    plt.xlabel("# of components")
    plt.ylabel("% of information captured")
    plt.legend(loc = 'lower right')
    
    # BLUE
    # plt.figure(6)
    i+=1
    plt.subplot(3,3,i+1)
    plt.plot(np.cumsum(bS) / np.sum(bS), linewidth = 3, color = "royalblue")
    plt.axvline(x= 10, color = "gold", label = "comps = 10", linestyle = "--", zorder =2)
    plt.axvline(x= 55, color = "gray",label = "comps = 55",linestyle = "--", zorder = 3)
    plt.axvline(x= 100, color = "black", label = "comps = 100",linestyle = "--", zorder = 4)
    plt.title('Blue Ch: CumSum ({})'
              .format('UoC' if no == 0 else 'Python'))
    plt.xlabel("# of components")
    plt.ylabel("% of information captured")
    plt.legend(loc = 'lower right')
    
    i+=1
    no = 1
    
plt.show()
