import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
try:
    import pyarrow.parquet as pq
    import scikitplot as skplt
except:
    print('Instalando Librerias')
    os.system('pip install pyarrow scikit-plot')
    import pyarrow.parquet as pq
    import scikitplot as skplt



def clusterize(data_train, max_clusters):
    clusters = range(2,max_clusters + 1)
    km_entrenados = []
    for cluster in clusters:
        kmeans = KMeans(n_clusters = cluster, init = 'k-means++', random_state = 1)
        kmeans.fit(data_train)
        km_entrenados.append(kmeans)
        
    return km_entrenados

#==================================================================================================================

def summary_sin_entrenar(km_entrenados, nombre_columnas):

    labels = []
    centroides = []
    dfs = []
    num_clusters = 2
    for km_entrenado in km_entrenados:
        centroides.append(km_entrenado.cluster_centers_)
        labels.append(str(num_clusters)+' Clusters')
        dfs.append(pd.DataFrame(km_entrenado.cluster_centers_,columns=nombre_columnas))
        num_clusters += 1
    
    final_df = pd.concat(dfs,keys=labels)
    final_df.to_csv('centroides_features.csv')
    
    return final_df

#=====================================================================================================================

def silhouette_metric_sin_entrenar(km_entrenados,data_train):

    results = {}
    cluster = 2
    clusters = []
    SS = []
    for km_entrenado in km_entrenados:
        cluster_labels = km_entrenado.predict(data_train)
        results[cluster] = cluster_labels
        print('{}_clusters'.format(cluster))
        score = silhouette_score(data_train, labels = cluster_labels, metric='euclidean')
        skplt.metrics.plot_silhouette(data_train, cluster_labels)
        #plt.savefig('SA_WO_{}_clusters.pdf'.format(cluster))
        silueta = plt.show()   
        SS.append(score)
        clusters.append(cluster)
        cluster +=1
  
    plt.plot(clusters, SS, 'bx-')
    plt.title('Silhouette Score for all data')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()
    return silueta
#==================================================================================================================

def get_closest_to_centroids_sin_entrenar(km_entrenados,data_train,data_original):
    closest_dict = {}
    for km_entrenado in km_entrenados:
        m_clusters = km_entrenado.labels_.tolist()
        closest_to_centroids = []
        for clusterid in range(km_entrenado.n_clusters):
            distance = km_entrenado.transform(data_train)[:, clusterid]
            closest_to_centroids.append(data_original.iloc[np.argsort(distance)[::][:1]])
        closest_dict[str(km_entrenado.n_clusters)+' Clusters']=pd.concat([x for x in closest_to_centroids])
    dfs=[]
    for key,value in closest_dict.items():
        dfs.append(value)
    result = pd.concat(dfs,keys=closest_dict.keys())
    
    return result


