import numpy as np
import os
import matplotlib.pyplot as plt

def eucl_dist_ts(ts1, ts2):
    """
    Function to calculate euclidean distance between two multivariate time series
    """
    dist = 0
    for t in range(len(ts1)):
        dist = dist + np.linalg.norm(ts1[t, :] - ts2[t, :])
    return dist


# Functions
def Plot_TS(i, tile_dir):
    """
    Plot a TS of 4 elements
    """
    plt.rcParams['figure.figsize'] = (20,10)
    tile0 = np.load(os.path.join(tile_dir, str(i)+'tile_T0.npy'))
    tile1 = np.load(os.path.join(tile_dir, str(i)+'tile_T1.npy'))
    tile2 = np.load(os.path.join(tile_dir, str(i)+'tile_T2.npy'))
    tile3 = np.load(os.path.join(tile_dir, str(i)+'tile_T3.npy'))
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(tile0[:,:,[0,1,2]])
    plt.title('T1')
    plt.subplot(1,4,2)
    plt.imshow(tile1[:,:,[0,1,2]])
    plt.title('T2')
    plt.subplot(1,4,3)
    plt.imshow(tile2[:,:,[0,1,2]])
    plt.title('T3')
    plt.subplot(1,4,4)
    plt.imshow(tile3[:,:,[0,1,2]])
    plt.title('T4')
    
			  
def Plot_TS_long(i, T, tile_dir):
	"""
    Plot a TS of T elements
    """
	plt.rcParams['figure.figsize'] = (10,5)
	plt.figure()
	for t in range(T):
		name= str(i)+'tile_T'+str(t)+'.npy'
		tile= np.load(os.path.join(tile_dir, name))
		plt.subplot(1,T,t+1)
		plt.imshow(tile[:,:,[0,1,2]], vmin=0, vmax=255)
		plt.title('t'+str(t+1))
		plt.axis("off")
		

def num2color(x, min_v, old_range):
    """
    Scale numbers to color 0..255
    """
    x = (((x - min_v) * 255) / old_range)
    return x.astype(int)


def tiff2img(img_file, tile_radius):
    """
    Load tiff image to np array (RGB)
    """
    dataset = gdal.Open(img_file)
    band1 = dataset.GetRasterBand(1)  # Red channel
    band2 = dataset.GetRasterBand(2)  # Green channel
    band3 = dataset.GetRasterBand(3)  # Blue channel

    b1 = band1.ReadAsArray()
    b2 = band2.ReadAsArray()
    b3 = band3.ReadAsArray()

    image = np.dstack((b1, b2, b3))
    image = np.pad(image, pad_width=[(tile_radius, tile_radius),
                                        (tile_radius, tile_radius), (0, 0)], mode='reflect')
    return image


def draw_tiles_map(img_ref, Xa, Ya, ix, tile_radius, color):
    """
    Function to draw a subset of tiles
    """
    row_min = Xa[ix] - tile_radius
    row_max = Xa[ix] + tile_radius
    col_min = Ya[ix] - tile_radius
    col_max = Ya[ix] + tile_radius
    img_ref[row_min:row_min + 10, col_min:col_min + tile_radius*2, :] = color
    img_ref[row_max:row_max + 10, col_min:col_min + tile_radius*2, :] = color
    img_ref[row_min:row_min + tile_radius*2, col_min:col_min + 10, :] = color
    img_ref[row_min:row_min + tile_radius*2, col_max:col_max + 10, :] = color
    return img_ref



def draw_grid_map(img, tile_size, colors):
    """
    Function to draw a grid of tiles within an image
    """
    n_sample = 0
    for i in range(0, img.shape[0] // tile_size):
        for j in range(0, img.shape[1] // tile_size):
            row_min = i*tile_size
            row_max = (i+1)*tile_size
            col_min = j*tile_size
            col_max = (j+1)*tile_size
            img[row_min:row_min + 10, col_min:col_min + tile_size, :] = colors[n_sample]
            img[row_max:row_max + 10, col_min:col_min + tile_size, :] = colors[n_sample]
            img[row_min:row_min + tile_size*2, col_min:col_min + 10, :] = colors[n_sample]
            img[row_min:row_min + tile_size*2, col_max:col_max + 10, :] = colors[n_sample]
            n_sample = n_sample + 1
    return img


def dist_mat_ts(X, n_samples):
    """
    Function to calculat the distance matrix of TS with euclidean distance
    """
    D = np.zeros((n_samples,n_samples))
    for i in range(n_samples - 1):
        for j in range(i, n_samples):
            D[i, j] = eucl_dist_ts(X[i], X[j])
    D = D + D.transpose()
    return D


def dist_mat_vec(Vec, n_samples):
	"""
	Function to calculate matrix of distances between the vectors given in Vec
	"""
	# Para hacer pruebas con un solo vector
	D = np.zeros((n_samples,n_samples))
	for i in range(n_samples - 1):
		for j in range(i, n_samples):
			D[i, j] =  np.linalg.norm(Vec[i] - Vec[j])
	D = D + D.transpose()
	Dist_mat = D


def pca2colors(V, n_clus):
    """
    Function to transform numbers an array if numbers (n_samples x 3) to the range of 0..255 of colors
    """
    min_v = np.ndarray.min(V)
    max_v = np.ndarray.max(V)
    old_range = max_v - min_v
    new_range = 255

    for i in range(n_clus):
        r = num2color(V[i, 0], min_v, old_range)
        g = num2color(V[i, 1], min_v, old_range)
        b = num2color(V[i, 2], min_v, old_range)
        V[i] = [r, g, b]
    return V


def resize_tile_avg(img, tile_size, colors):
    """
    Function to resize the image by averagig the pixels within each tile
    """
    n = 0
    reduced_img = []
    for i in range(0, img.shape[0] // tile_size):
        for j in range(0, img.shape[1] // tile_size):
            row_min = i*tile_size
            row_max = (i+1)*tile_size
            col_min = j*tile_size
            col_max = (j+1)*tile_size
            
            avg_color = np.uint8(np.mean(img[row_min:row_mac, col_min:col_max].reshape(tile_size*tile_size, 3), axis=0))
            reduced_img.append(avg_color)
            n = n + 1
    return reduced_img


def triplet_loss_clustering(z_a, z_n, z_d, margin):
    """
    Function to calculate the triplet loss of three vectors witha given margin
    """
    #l_n = np.sum((z_a - z_n) ** 2)
    #l_d = np.sum((z_a - z_d) ** 2)
    l_n = np.linalg.norm(z_a - z_n)
    l_d = np.linalg.norm(z_a - z_d)
    loss = l_n - l_d + margin
    loss = max(0, loss)
    return loss


def tile2vec_error(D,y,K, margin):
    '''
    D: pairwise distance matrix
    y: cluster labels
    K: number of clusters
    '''
    m= D.shape[0]
    y_k= [[] for k in range(K)]

    # Tomamos los puntos de cada cluster
    for i in range(m):
        y_k[y[i]].append(i)

    # Tomamos los puntos de fuera de cada cluster
    y_l=[[] for k in range(K)]
    for k in range(K):
        for l in range(K):
            if k!= l:
                y_l[k].extend(y_k[l])

    # Para cada punto, obtenemos la distancia mayor dentro de su cluster
    max_k= list()
    for k in range(K):
        max_k.append(np.max(D[y_k[k],:][:,y_k[k]], axis= 1))


    # Para cada punto, obtener las distancias de loss puntos de fuera menores que la mayor a los de su cluster
    Dcand_k= list()
    for k in range(K):
        Dcand_k.append(list())
        for ind,i in enumerate(y_k[k]):
            Dcand_k[k].append(D[i,y_l[k]][D[i,y_l[k]]<max_k[k][ind]] + margin)
            
    # Calcular número de tripletas
    #N = np.sum([len(y_k[k])*(len(y_k[k])-1)/2* (m-len(y_k[k])) for k in range(K)])
    
    # Computar el error para cada par de puntos con respecto al resto
    error= 0
    for k in range(K):
        for ind,i in enumerate(y_k[k]):
            for j in  y_k[k]:
                error+= np.sum(np.clip(D[i,j]-Dcand_k[k][ind] + margin, a_min= 0, a_max= np.inf))
                
    return error


def generateTriplets(tilesCluster, numTriplets=14):
    '''
    Generates a list of triplets (anchor,neighbor,distant) according to a neighborhood defined in terms
    of a clustering with K clusters. The triples are stratified (the proportions of anchors, neighbors and distant
    tiles are obtained proportionally to the size of the clusters)

    tilesCluster: the cluster associated to each tile from a list of tiles, list(int)
    numTriplets: the number of triples generated (anchor, neighbor, distant) given in terms
    of the indices of the tiles in the list tilesCluster, (int,int,int)

    return a list of triplets of indices according to tilesCluster, list((int,int,int))
    '''
	
    n = len(tilesCluster)
    sizeClusters = np.unique(tilesCluster,return_counts=True)[1]
    probCluster = sizeClusters/np.sum(sizeClusters)
    K = len(sizeClusters)
    indClusters = [list() for k in range(K)]
    for ind in range(n):
        indClusters[tilesCluster[ind]].append(ind)

    m_k = np.round(numTriplets*probCluster).astype(int)
    m_k[-1] += numTriplets-np.sum(m_k)
    triplets = list()
    for k in range(K):
        distClust = [l for l in range(K) if l != k]
        distProb = probCluster[distClust]/np.sum(probCluster[distClust])

        for i in range(m_k[k]):
            l = np.random.choice(distClust, 1, p=distProb)[0]
            an = indClusters[k][np.random.randint(0, sizeClusters[k])]
            nb = indClusters[k][np.random.randint(0, sizeClusters[k])]
            while (nb == an):
                nb = indClusters[k][np.random.randint(0, sizeClusters[k])]
            di = indClusters[l][np.random.randint(0, sizeClusters[l])]

            triplets.append((an, nb, di))

    return triplets


# def generateTriplets(tilesCluster, numTriplets=14, centroids=None):
#     '''
#     Generates a list of triplets (anchor,neighbor,distant) according to a neighborhood defined in terms
#     of a clustering with K clusters. The triples are stratified (the proportions of anchors, neighbors and distant
#     tiles are obtained proportionally to the size of the clusters)

#     tilesCluster: the cluster associated to each tile from a list of tiles, list(int)
#     numTriplets: the number of triples generated (anchor, neighbor, distant) given in terms
#     of the indices of the tiles in the list tilesCluster, (int,int,int)
#     centroids: the list of centroids np-array(numClusters,time,embedding dimension)

#     return a list of triplets of indices according to tilesCluster, list((int,int,int))
#     '''

#     n = len(tilesCluster)
#     sizeClusters = np.unique(tilesCluster,return_counts=True)[1]
#     probCluster = sizeClusters/np.sum(sizeClusters)
#     K = len(sizeClusters)
#     indClusters = [list() for k in range(K)]
#     for ind in range(n):
#         indClusters[tilesCluster[ind]].append(ind)

#     if centroids is not None:
#         probInv= np.zeros((K,K))
#         for k in range(K):
#             for l in range(K):
#                 # probabilidad inmensamente proporcional a la distancia
#                 #probInv[k,l]= np.sum(np.linalg.norm(centroids[k,:,:]-centroids[l,:,:],axis=1))
#                 probInv[k,l]=np.sum(np.abs(centroids[k, :, :] - centroids[l, :, :]))
            
#             probInv[k,:]/=np.sum(probInv[k,:])
#     else:
#         probInv = np.ones((K, K))

#     # el numero de tripletas por cada cluster
#     m_k = np.round(numTriplets*probCluster).astype(int)
#     m_k[-1] += numTriplets-np.sum(m_k)
#     triplets = list()
#     for k in range(len(sizeClusters)):
#         # Para cada cluster
#         distClust = [l for l in range(K) if l != k]
#         distProb = probCluster[distClust]/np.sum(probCluster[distClust])
#         for l in range(len(distClust)):
#             distProb[l] *= probInv[k,distClust[l]]
#         distProb /= np.sum(distProb)

#         for i in range(m_k[k]):
#             # Crear tripleta
#             l = np.random.choice(distClust, 1, p=distProb)[0]
#             an = indClusters[k][np.random.randint(0, sizeClusters[k])]
#             nb = indClusters[k][np.random.randint(0, sizeClusters[k])]
#             while (nb == an):
#                 nb = indClusters[k][np.random.randint(0, sizeClusters[k])]
#             di = indClusters[l][np.random.randint(0, sizeClusters[l])]

#             triplets.append((an, nb, di))

#     return triplets


def plot_clustering_map(n_clus, clusters, centroids, pca_model, save=False, res_dir_fig='', name=''):
	''' 
	Function to plot the result of the clustering in the arrangament of the original map.
	Each tile of the map is assigned with the color of its corresponding cluster in the position in which this tile is allocated in the original map.
	'''
	h = 110
	n_samples = len(clusters)
	# Assign components to the centroids
	V_trans = pca_model.transform(centroids)
	# Transform 3 components to colors
	V_trans = pca2colors(V_trans, n_clus)
	# Assign the color to each point according to cluster
	comp_pca = np.zeros((n_samples, 3))
	for i in range(n_samples):
		comp_pca[i, :] = V_trans[clusters[i]]
	Mc = comp_pca.reshape(h, h, 3)
	Mc = Mc.astype(np.uint8)
	plt.figure(figsize=(8,8))
	#plt.imshow(Mc, interpolation='gaussian')
	plt.imshow(Mc)
	plt.axis("off")
	if save:
		plt.savefig(os.path.join(res_dir_fig, name), bbox_inches='tight')


	
	
