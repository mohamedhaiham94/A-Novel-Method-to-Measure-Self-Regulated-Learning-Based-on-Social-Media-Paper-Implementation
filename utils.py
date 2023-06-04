from matplotlib.patches import Ellipse
import pandas as pd
import numpy as np
import collections
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.mixture import GaussianMixture
pd.options.mode.chained_assignment = None  # default='warn'
import os
import warnings

import seaborn as sns
warnings.filterwarnings("ignore")


def Rename_Col(df, check = 1):
  df.rename(columns={df.columns[0]: 'Name'}, inplace=True)
  columns = df.columns
  i = 1

  names = df["Name"]
  gender = df["Gender"]
  ac_year = df["Ac Year"]
  data = [names, gender, ac_year]
  if check:
    for col in columns[1:-2]:
        df.rename(columns={col:'Q'+str(i)}, inplace=True)
        i+=1
    df = df[df.columns[1:-2]]
  else:
    df.dropna(subset=['Gender'], how='all', inplace=True)

    for col in columns[1:-2]:
      df.rename(columns={col:'Q'+str(i)}, inplace=True)
      i+=1
    df = df[df.columns[1:-3]]


  df["response_avg"] = (df.sum(axis=1) / (len(df.columns) * 7) ) * 100


  return df, data

def Drop_Col_Rows(df, rows_numbers):
  
  df.drop(df.tail(rows_numbers).index,inplace=True)

  return df


def Fillna_Normalize(df,filename):
    for col_name in df.columns:
      df[col_name] = df[col_name].astype(float)
      df[col_name].fillna(df[col_name].mode()[0], inplace=True)
    
    df["Average score"] = (df.sum(axis=1) / (len(df.columns) * 7) ) * 100
    if not os.path.isdir("processed"):
      os.makedirs("processed")
    df.to_csv('processed/'+filename+'_processed.csv', index=False)
    return df

def PCA_Data(X):
  distortions = []
  K_to_try = range(1, 9)
  pca = PCA(n_components = 2, random_state=1)
  X_pca = pca.fit_transform(X)
  for i in K_to_try:
      model = KMeans(n_clusters=i)
      model.fit(X_pca)
      distortions.append(model.inertia_)

  plt.plot(K_to_try, distortions, marker='o')
  plt.xlabel('Number of Clusters (k)')
  plt.ylabel('Distortion')
  plt.show()



def Plot_kmeans(X_pca,k, filename):
  model = KMeans(n_clusters=k)
  model = model.fit(X_pca)
  y = model.predict(X_pca)

  colors_c = ['orange','red','green']
  for i in range(k):
    plt.scatter(X_pca[y == int(i), 0], X_pca[y == int(i), 1], s = 50, c = colors_c[i], label = 'Cluster '+str(i))

  plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 80, c = 'blue', label = 'Centroids')
  plt.title('Clusters of '+filename+' Kmeans')
  plt.legend()
  plt.grid()
  plt.show()


def KMeans_Data(X, k, filename, names):
  pca = PCA(n_components = 2, random_state=1)
  X_pca = pca.fit_transform(X)

  model_k = KMeans(n_clusters=k)
  model_k = model_k.fit(X_pca)
  labels = model_k.predict(X_pca)

  frame = X.copy()
  frame['cluster'] = labels

  colors_c = ['orange','red','green']
  for i in range(k):
    plt.scatter(X_pca[labels == int(i), 0], X_pca[labels == int(i), 1], s = 50, c = colors_c[i], label = 'Cluster '+str(i))

  plt.scatter(model_k.cluster_centers_[:, 0], model_k.cluster_centers_[:, 1], s = 80, c = 'blue', label = 'Centroids')
  plt.title('Clusters of '+filename+' Kmeans')
  plt.legend()
  plt.grid()
  plt.show()

  print('Final K Means Result : ')
  print(collections.Counter(labels))

  frame.insert(loc=0, column='Name', value=names['Name'])
  frame.insert(loc=1, column='Group', value=names['Group'])
  frame.insert(loc=2, column='Gender', value=names['Gender'])
  frame.insert(loc=3, column='Ac Year', value=names['Ac Year'])
  frame.insert(loc=4, column='Age', value=names['Age'])
  frame.insert(loc=5, column='N courses', value=names['N courses'])


  if not os.path.isdir("kmeans"):
    os.makedirs("kmeans")
  frame.to_csv('kmeans/'+filename+'_kmeans_result.csv', index=False)

  print('This Results Belongs To : '+filename)
  for i in range(0,k):
    data2 = frame[frame['cluster']==i]
    mean = (data2.shape[0] / frame.shape[0]) * 5
    print('Mean cluster '+str(i)+' : {:.2f}'.format(mean))



def GaussianMixture_Data(X, k, filename, names):
  
  pca = PCA(n_components = 2, random_state=1)
  X_pca = pca.fit_transform(X)

  gmm = GaussianMixture(n_components=k, init_params='kmeans')
  gmm.fit(X_pca)
  print('Final GaussianMixture Result : ')

  #predictions from gmm
  labels = gmm.predict(X_pca)
  print(collections.Counter(labels))

  frame = X.copy()
  frame['cluster'] = labels

  colors_c = ['orange','red','green']
  for i in range(k):
    plt.scatter(X_pca[labels == int(i), 0], X_pca[labels == int(i), 1], s = 50, c = colors_c[i], label = 'Cluster '+str(i))

  plt.title('Clusters of '+filename+' GMM')
  plt.legend()
  plt.grid()
  plt.show()

  print('Final K GMM Result : ')
  print(collections.Counter(labels))
  frame.insert(loc=0, column='Name', value=names['Name'])
  frame.insert(loc=1, column='Gender', value=names['Gender'])
  frame.insert(loc=2, column='Ac Year', value=names['Ac Year'])
  if not os.path.isdir("gmm"):
    os.makedirs("gmm")
  frame.to_csv('gmm/'+filename+'_gmm_result.csv', index=False)

  print('This Results Belongs To : '+filename)
  total = frame.shape[0]
  for i in range(0,k):
    data = frame[frame['cluster']==i]
    data = data['cluster']
    mean = (data.shape[0] / total) * 10
    print('Mean cluster '+str(i)+' : {:.2f}'.format(mean))
  
  print('--------------------------------')
def GaussianMixture_Data(X, k, filename, names):
  
  pca = PCA(n_components = 2, random_state=1)
  X_pca = pca.fit_transform(X)

  gmm = GaussianMixture(n_components=k, init_params='kmeans')
  gmm.fit(X_pca)
  print('Final GaussianMixture Result : ')

  #predictions from gmm
  labels = gmm.predict(X_pca)
  print(collections.Counter(labels))

  frame = X.copy()
  frame['cluster'] = labels

  colors_c = ['orange','red','green']
  for i in range(k):
    plt.scatter(X_pca[labels == int(i), 0], X_pca[labels == int(i), 1], s = 50, c = colors_c[i], label = 'Cluster '+str(i))

  plt.title('Clusters of '+filename+' GMM')
  plt.legend()
  plt.grid()
  plt.show()

  print('Final K GMM Result : ')
  print(collections.Counter(labels))
  frame.insert(loc=0, column='Name', value=names['Name'])
  frame.insert(loc=1, column='Gender', value=names['Gender'])
  frame.insert(loc=2, column='Ac Year', value=names['Ac Year'])
  if not os.path.isdir("gmm"):
    os.makedirs("gmm")
  frame.to_csv('gmm/'+filename+'_gmm_result.csv', index=False)

  print('This Results Belongs To : '+filename)
  total = frame.shape[0]
  for i in range(0,k):
    data = frame[frame['cluster']==i]
    data = data['cluster']
    mean = (data.shape[0] / total) * 10
    print('Mean cluster '+str(i)+' : {:.2f}'.format(mean))
  
  