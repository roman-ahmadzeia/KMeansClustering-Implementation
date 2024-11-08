import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class KmeansCluster:
    
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_d(x, centroids):
        return np.sqrt(np.sum((centroids - x)**2, axis=1))

    
    def fit(self, X, num_itr = 100):
        self.centroids = np.random.uniform(np.amin(X, axis =0), np.amax(X, axis=0), size=(self.k, X.shape[1]))
        
        for _ in range(num_itr):
            y = []
            
            for i in X:
                distances = KmeansCluster.euclidean_d(i, self.centroids)
                print(f'Distances: {distances}')
                cluster_num = np.argmin(distances)
                y.append(cluster_num)
            
            y = np.array(y)
            
            cluster_indices = []
            
            for i in range(self.k):
            
                cluster_indices.append(np.argwhere(y == i))
            
            cluster_centers = []
            
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])
            
            if np.max(self.centroids - np.array(cluster_centers)) < 0.005:
                break
            
            else:
                self.centroids = np.array(cluster_centers)
          
        return y          
            
            


            
        



# data = pd.read_csv('data.csv')
#
# X = data.drop(['color'],axis=1)
#
# X.astype('int')
#
#
# KCluster = KmeansCluster(k=3)
# labels = KCluster.fit(X)
#
# plt.scatter(X[:,0], X[:,1],c=labels)
# plt.scatter(KCluster.centroids[:,0], KCluster.centroids[:,1], c=range(len(KCluster.centroids)),marker="*",s=200)
# data = pd.read_csv('data.csv')
# X = data.drop(['color'],axis=1)
#
# KCluster = KmeansCluster(k=3)
# labels = KCluster.fit(X)

data = np.random.randint(0,100, (100,2))
KCluster = KmeansCluster(k=3)
labels = KCluster.fit(data)
plt.scatter(data[:,0], data[:,1],c=labels)
plt.scatter(KCluster.centroids[:,0], KCluster.centroids[:,1], c=range(len(KCluster.centroids)),marker="*",s=200)
plt.show()